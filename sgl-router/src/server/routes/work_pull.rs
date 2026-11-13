// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0
use crate::config::PrefillDispatchMode;
use crate::discovery::WorkerMode;
use crate::server::{app_context::AppContext, error::ApiError};
use crate::state::work_pull::{validate_acquire, validate_outcome, Acquire, Action, Outcome};
use axum::{
    extract::{FromRequestParts, State},
    http::{header::AUTHORIZATION, request::Parts, HeaderMap},
    Json,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use subtle::ConstantTimeEq;

fn enabled(ctx: &AppContext, headers: &HeaderMap) -> Result<(), ApiError> {
    if ctx.config.proxy.pull.mode != PrefillDispatchMode::Pull {
        return Err(ApiError::BadRequest("prefill pull is disabled".into()));
    }
    let expected = ctx
        .config
        .server
        .worker_auth
        .as_ref()
        .filter(|auth| {
            auth.as_bytes()
                .strip_prefix(b"Bearer ")
                .is_some_and(|key| !key.is_empty() && key.iter().all(u8::is_ascii_graphic))
        })
        .ok_or(ApiError::PullAuthenticationRequired)?;
    let mut supplied = headers.get_all(AUTHORIZATION).iter();
    let supplied_value = supplied
        .next()
        .ok_or(ApiError::PullAuthenticationRequired)?;
    if supplied.next().is_some() {
        return Err(ApiError::PullAuthenticationRequired);
    }
    // Fixed-size digests avoid secret-dependent early exits or prefix/length
    // comparisons. The hash cost depends only on the public input length.
    let supplied_digest = Sha256::digest(supplied_value.as_bytes());
    let expected_digest = Sha256::digest(expected.as_bytes());
    if !bool::from(supplied_digest.ct_eq(&expected_digest)) {
        return Err(ApiError::PullAuthenticationRequired);
    }
    Ok(())
}
/// Runs before the body extractor so unauthenticated callers cannot trigger
/// JSON decoding or reach assignment state, even with a malformed payload.
pub struct PullAuthorization;

impl FromRequestParts<Arc<AppContext>> for PullAuthorization {
    type Rejection = ApiError;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &Arc<AppContext>,
    ) -> Result<Self, Self::Rejection> {
        enabled(state, &parts.headers)?;
        Ok(Self)
    }
}

pub async fn acquire(
    State(ctx): State<Arc<AppContext>>,
    _authorization: PullAuthorization,
    Json(mut input): Json<Acquire>,
) -> Result<Json<Value>, ApiError> {
    validate_acquire(&input)?;
    input.worker_url = input.worker_url.trim_end_matches('/').to_owned();
    let worker = ctx
        .registry
        .all()
        .into_iter()
        .find(|p| p.url.trim_end_matches('/') == input.worker_url)
        .ok_or_else(|| ApiError::BadRequest("pull worker is not registered".into()))?;
    if worker.mode() != WorkerMode::Prefill || worker.dp_ranks() != 1 {
        return Err(ApiError::BadRequest(
            "pull requires a registered DP1 prefill worker".into(),
        ));
    }
    Ok(Json(match ctx.pull_queue.acquire(&input).await? {
        Some(offer) => serde_json::to_value(offer).unwrap(),
        None => json!({"status":"empty"}),
    }))
}
pub async fn accept(
    State(ctx): State<Arc<AppContext>>,
    _authorization: PullAuthorization,
    Json(input): Json<Action>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        json!({"status":ctx.pull_queue.assignment(&input)?.map_or("expired", |a| a.accept())}),
    ))
}
pub async fn reject(
    State(ctx): State<Arc<AppContext>>,
    _authorization: PullAuthorization,
    Json(input): Json<Action>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        json!({"status":ctx.pull_queue.assignment(&input)?.map_or("expired", |a| a.reject())}),
    ))
}
pub async fn status(
    State(ctx): State<Arc<AppContext>>,
    _authorization: PullAuthorization,
    Json(input): Json<Action>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        json!({"status":ctx.pull_queue.assignment(&input)?.map_or("expired", |a| a.status())}),
    ))
}
pub async fn outcome(
    State(ctx): State<Arc<AppContext>>,
    _authorization: PullAuthorization,
    Json(input): Json<Outcome>,
) -> Result<Json<Value>, ApiError> {
    validate_outcome(&input)?;
    Ok(Json(
        json!({"status":ctx.pull_queue.assignment(&input.action)?.map_or("expired", |a| a.outcome(input.status_code, input.body))}),
    ))
}
