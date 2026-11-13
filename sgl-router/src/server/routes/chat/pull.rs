// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::{forward, preparation::PreparedRequest, reorg, ModelRouting};
use crate::discovery::WorkerMode;
use crate::server::{app_context::AppContext, error::ApiError};
use axum::{
    body::Body,
    http::{HeaderMap, Response},
};
use futures::StreamExt;
use std::{
    collections::HashSet,
    time::{Duration, Instant},
};

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
    let mut eligible = HashSet::new();
    for p in ctx
        .registry
        .workers_for_mode(&request.model, WorkerMode::Prefill)
    {
        if p.dp_ranks() != 1 {
            continue;
        }
        match reorg::prefill_is_eligible(ctx, resolver, &request, &headers, &p).await {
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
    if eligible.is_empty() {
        return Err(ApiError::NoPrefillWorkersAvailable {
            model: request.model.0.clone(),
        });
    }
    let (lease, mut claims) = ctx
        .pull_queue
        .enqueue(eligible, tokens, request.body.len())?;
    // The client deadline includes the time spent waiting in the shared queue.
    let deadline = start
        + ctx
            .router_inflight_load
            .stale_request_timeout()
            .min(Duration::from_secs(ctx.config.proxy.request_timeout_secs));
    let duration = forward::RequestDurationGuard::new(ctx, &request.model, start);
    let dispatch = async {
        loop {
            let assignment = claims
                .recv()
                .await
                .ok_or_else(|| ApiError::PrefillFailed { status: None })?;
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
