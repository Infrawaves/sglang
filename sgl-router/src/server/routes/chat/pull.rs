// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::{forward, preparation::PreparedRequest, reorg, ModelRouting};
use crate::discovery::WorkerMode;
use crate::server::{app_context::AppContext, error::ApiError};
use crate::state::kv_events::PrefixLookupResult;
use axum::{
    body::Body,
    http::{HeaderMap, Response},
};
use futures::StreamExt;
use sgl_kv_indexer::PrefixOutcome;
use std::{
    collections::{HashMap, HashSet},
    time::{Duration, Instant},
};

/// How often an unclaimed request checks whether its worker fleet changed.
const FLEET_RECHECK: Duration = Duration::from_secs(1);
/// Recompute eligibility this often even if the fleet looks unchanged, for any
/// input the fingerprint misses. Arbitrary; bounds a missed refresh to 5 s.
const FLEET_REFRESH_MAX: Duration = Duration::from_secs(5);

/// Every worker of the model, prefill and decode, with what eligibility reads
/// from it: whether it admits traffic, its version group and DP ranks. A
/// prefill is eligible only with a compatible decode peer, so a decode coming
/// back must refresh the set as much as a prefill being replaced.
type Fleet = Vec<(String, WorkerMode, bool, Option<String>, u32)>;

fn worker_fleet(ctx: &AppContext, request: &PreparedRequest) -> Fleet {
    let mut fleet: Fleet = ctx
        .registry
        .workers_for(&request.model)
        .into_iter()
        .map(|w| {
            (
                w.url.trim_end_matches('/').to_owned(),
                w.mode(),
                w.breaker.snapshot().admit,
                w.version_group().map(str::to_owned),
                w.dp_ranks(),
            )
        })
        .collect();
    fleet.sort_by(|a, b| (&a.0, a.1 as u8).cmp(&(&b.0, b.1 as u8)));
    fleet
}

/// URLs of the prefill workers that may claim `request` now.
async fn eligible_prefills(
    ctx: &AppContext,
    resolver: &crate::buckets_reorg::BucketResolver,
    request: &PreparedRequest,
    headers: &HeaderMap,
) -> Result<HashSet<String>, ApiError> {
    let mut eligible = HashSet::new();
    for p in ctx
        .registry
        .workers_for_mode(&request.model, WorkerMode::Prefill)
    {
        if p.dp_ranks() != 1 {
            continue;
        }
        match reorg::prefill_is_eligible(ctx, resolver, request, headers, &p).await {
            Ok(()) => {
                eligible.insert(p.url.trim_end_matches('/').to_owned());
            }
            Err(
                ApiError::NoPrefillWorkersAvailable { .. }
                | ApiError::NoDecodeWorkersAvailable { .. }
                | ApiError::NoHealthyWorkers { .. },
            ) => {}
            Err(error) => return Err(error),
        }
    }
    Ok(eligible)
}

/// Prompt tokens each eligible prefill worker holds cached, from the router's
/// local KV-event tree, keyed by the URL the queue matches workers on. Empty
/// without a local tree: the remote KV indexer is not queried on this path.
fn prefix_hits(
    ctx: &AppContext,
    request: &PreparedRequest,
    eligible: &HashSet<String>,
    tokens: usize,
) -> HashMap<String, usize> {
    let mut hits = HashMap::new();
    let (Some(provider), Some(prompt)) = (
        ctx.radix_tree_prefix_provider.as_ref(),
        request.tokens.as_ref(),
    ) else {
        return hits;
    };
    let Some(PrefixLookupResult {
        outcome: PrefixOutcome::Matched { matches, .. },
        query_blocks,
        ..
    }) = provider.match_request_tokens(&prompt.ids, &request.cache_namespace)
    else {
        return hits;
    };
    if query_blocks == 0 {
        return hits;
    }
    for entry in matches {
        let url = entry.address.trim_end_matches('/');
        if !eligible.contains(url) {
            continue;
        }
        let blocks = (entry.matched_prefix_blocks as usize).min(query_blocks);
        let hit = hits.entry(url.to_owned()).or_insert(0);
        *hit = (*hit).max(tokens.saturating_mul(blocks) / query_blocks);
    }
    hits
}

pub(super) async fn dispatch(
    ctx: &AppContext,
    routing: &ModelRouting<'_>,
    mut request: PreparedRequest,
    headers: HeaderMap,
    start: Instant,
) -> Result<Response<Body>, ApiError> {
    let ModelRouting::Reorg(resolver) = routing else {
        return Err(ApiError::BadRequest(
            "pull requires HTTP reorg PD routing".into(),
        ));
    };
    if request.body.len() > 15 * 1024 * 1024 {
        return Err(ApiError::BadRequest("pull request exceeds 15 MiB".into()));
    }
    let value: serde_json::Value =
        serde_json::from_slice(&request.body).map_err(|e| ApiError::BadRequest(e.to_string()))?;
    if request.path != "/generate"
        || request.fans_out
        || ["image_data", "audio_data", "video_data", "input_embeds"]
            .iter()
            .any(|key| value.get(*key).is_some_and(|v| !v.is_null()))
    {
        return Err(ApiError::BadRequest(
            "pull supports one text /generate prompt (or chat converted to /generate), n=1".into(),
        ));
    }
    let ids = value.get("input_ids").and_then(|v| v.as_array());
    if !(value.get("text").is_some_and(|v| v.is_string())
        || ids.is_some_and(|ids| ids.iter().all(|v| v.is_u64())))
    {
        return Err(ApiError::BadRequest(
            "pull requires scalar text or a single input_ids array".into(),
        ));
    }
    // A byte estimate can prevent a valid prompt from ever fitting a P's token
    // grant (especially for non-ASCII text). Preparation already tokenizes text
    // when a model tokenizer is available, even with ID forwarding disabled.
    let tokens = request
        .tokens
        .as_ref()
        .ok_or_else(|| {
            ApiError::BadRequest(
                "pull requires exact prompt tokens; configure a router tokenizer or send input_ids"
                    .into(),
            )
        })?
        .ids
        .len();
    if tokens == 0 {
        return Err(ApiError::BadRequest(
            "pull requires a nonempty prompt".into(),
        ));
    }
    request.forward_prompt_ids()?;
    let mut fleet = worker_fleet(ctx, &request);
    let eligible = eligible_prefills(ctx, resolver, &request, &headers).await?;
    if eligible.is_empty() {
        return Err(ApiError::NoPrefillWorkersAvailable {
            model: request.model.0.clone(),
        });
    }
    let matched = prefix_hits(ctx, &request, &eligible, tokens);
    let (lease, mut claims) =
        ctx.pull_queue
            .enqueue(eligible, matched, tokens, request.body.len())?;
    // The client deadline includes the time spent waiting in the shared queue.
    let deadline = start
        + ctx
            .router_inflight_load
            .stale_request_timeout()
            .min(Duration::from_secs(ctx.config.proxy.request_timeout_secs));
    let duration = forward::RequestDurationGuard::new(ctx, &request.model, start);
    // Kept across claim attempts: a worker that claims and rejects faster than
    // the interval must not keep pushing the next refresh out.
    let mut next_check = Instant::now() + FLEET_RECHECK;
    let mut next_refresh = Instant::now() + FLEET_REFRESH_MAX;
    let dispatch = async {
        loop {
            let assignment = loop {
                if Instant::now() >= next_check {
                    // A rolling replacement or a decode coming back changes who
                    // may claim this unclaimed request: re-derive it instead of
                    // keeping the set fixed at enqueue. Empty keeps it queued.
                    let now = Instant::now();
                    next_check = now + FLEET_RECHECK;
                    let current = worker_fleet(ctx, &request);
                    if current != fleet || now >= next_refresh {
                        fleet = current;
                        next_refresh = now + FLEET_REFRESH_MAX;
                        let eligible = eligible_prefills(ctx, resolver, &request, &headers).await?;
                        let matched = prefix_hits(ctx, &request, &eligible, tokens);
                        ctx.pull_queue.update_candidates(&lease, eligible, matched);
                    }
                }
                tokio::select! {
                    claim = claims.recv() => {
                        break claim.ok_or(ApiError::PrefillFailed { status: None })?;
                    }
                    _ = tokio::time::sleep_until(next_check.into()) => {}
                }
            };
            let p = ctx
                .registry
                .workers_for_mode(&request.model, WorkerMode::Prefill)
                .into_iter()
                .find(|p| p.url.trim_end_matches('/') == assignment.worker_url)
                .ok_or_else(|| ApiError::NoPrefillWorkersAvailable {
                    model: request.model.0.clone(),
                })?;
            // Re-read decode health/admission at acquisition; the P identity stays pinned.
            let workers = match reorg::select_workers_for_prefill(
                ctx, resolver, &request, &headers, &p,
            )
            .await
            {
                Ok(workers) => workers,
                Err(
                    ApiError::NoPrefillWorkersAvailable { .. }
                    | ApiError::NoDecodeWorkersAvailable { .. }
                    | ApiError::NoHealthyWorkers { .. }
                    | ApiError::PolicySelectionFailed { .. },
                ) => {
                    assignment.reject();
                    ctx.pull_queue.requeue(&lease);
                    continue;
                }
                Err(error) => {
                    assignment.reject();
                    return Err(error);
                }
            };
            let result = forward::forward_request(
                ctx,
                &mut request,
                workers,
                headers.clone(),
                start,
                &duration,
                Some(assignment.clone()),
            )
            .await;
            if assignment.expired() {
                ctx.pull_queue.requeue(&lease);
                continue;
            }
            // No automatic retry after commit: execution may have reached the P/D pair.
            return result.map(|r| r.response);
        }
    };
    let response = tokio::time::timeout_at(deadline.into(), dispatch)
        .await
        .map_err(|_| ApiError::StaleRequestExpired {
            model: request.model.0.clone(),
        })??;
    Ok(response.map(|body| {
        Body::from_stream(body.into_data_stream().map(move |frame| {
            let _hold_until_client_finishes = &lease;
            frame
        }))
    }))
}
