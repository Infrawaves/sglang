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
/// An unclaimed request with nobody eligible recomputes this often even if the
/// fleet looks unchanged, for any input the fingerprint misses. Requests that
/// someone can claim skip it: at scale it ran bucket checks for every queued
/// request against every prefill. Arbitrary; bounds a stuck request to 5 s.
const FLEET_REFRESH_EMPTY: Duration = Duration::from_secs(5);
/// Room left in the offer body for the rid, bootstrap fields and sampling
/// defaults dispatch adds after this check. Arbitrary but generous.
const OFFER_BODY_HEADROOM: usize = 64 * 1024;

/// Every worker of the model, prefill and decode, with what eligibility reads
/// from it: whether it admits traffic, its version group and DP ranks. A
/// prefill is eligible only with a compatible decode peer, so a decode coming
/// back must refresh the set as much as a prefill being replaced.
/// Hashed and shared per model for `FLEET_RECHECK` (`PullQueue::fleet_generation`),
/// so queued requests compare one number instead of each rebuilding the fleet.
fn worker_fleet(ctx: &AppContext, request: &PreparedRequest) -> u64 {
    ctx.pull_queue
        .fleet_generation(&request.model.0, FLEET_RECHECK, || {
            let mut fleet: Vec<_> = ctx
                .registry
                .workers_for(&request.model)
                .into_iter()
                .map(|w| {
                    // Everything bucket membership reads off a worker: a
                    // prefill still inside its bootstrap-port grace is skipped
                    // and becomes eligible later with no registry change.
                    (
                        w.url.trim_end_matches('/').to_owned(),
                        w.mode() as u8,
                        w.breaker.would_allow(),
                        w.awaiting_bootstrap_port(),
                        w.version_group().map(str::to_owned),
                        w.dp_ranks(),
                        w.services(),
                    )
                })
                .collect();
            fleet.sort();
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            std::hash::Hash::hash(&fleet, &mut hasher);
            std::hash::Hasher::finish(&hasher)
        })
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
    if request.body.len() > crate::state::work_pull::MAX_OFFER_BODY_BYTES {
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
    // Replacing text with JSON ids can grow the body past what an offer may
    // carry; refuse here rather than after the request was queued and claimed.
    if request.body.len() + OFFER_BODY_HEADROOM > crate::state::work_pull::MAX_OFFER_BODY_BYTES {
        return Err(ApiError::BadRequest(
            "pull request exceeds 15 MiB once its prompt is token ids".into(),
        ));
    }
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
    let mut next_refresh = Instant::now() + FLEET_REFRESH_EMPTY;
    let mut nobody_eligible = false;
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
                    if current != fleet || (nobody_eligible && now >= next_refresh) {
                        fleet = current;
                        next_refresh = now + FLEET_REFRESH_EMPTY;
                        let eligible = eligible_prefills(ctx, resolver, &request, &headers).await?;
                        nobody_eligible = eligible.is_empty();
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
            let Some(p) = ctx
                .registry
                .workers_for_mode(&request.model, WorkerMode::Prefill)
                .into_iter()
                .find(|p| p.url.trim_end_matches('/') == assignment.worker_url)
            else {
                // The claimant left the registry after claiming. Nothing ran
                // yet, so return the request for another prefill to claim.
                assignment.reject();
                ctx.pull_queue.requeue(&lease);
                continue;
            };
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
            // Selection checks breaker readiness. Claim the probe after that
            // check so selecting a half-open P does not exclude itself.
            if !assignment.admit_prefill(&workers.prefill.breaker) {
                assignment.reject();
                ctx.pull_queue.requeue(&lease);
                continue;
            }
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
    // The lease lives as long as the client's body. A successful D response
    // read to its end marks it finished, so P is not cancelled for finishing
    // first; a D error or a client disconnect still cancels P at once.
    let succeeded = response.status().is_success();
    Ok(response.map(|body| {
        // Yields no frame; it only runs once the body is exhausted.
        let end = futures::stream::once(async move {
            if succeeded {
                lease.finish();
            }
            drop(lease);
            None::<Result<bytes::Bytes, axum::Error>>
        })
        .filter_map(futures::future::ready);
        Body::from_stream(body.into_data_stream().chain(end))
    }))
}
