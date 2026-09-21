// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::chat::{proxy_inference, InferenceEndpoint};
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, Response};
use bytes::Bytes;
use std::sync::Arc;

/// Proxy Responses requests, preserving their body and JSON/SSE response format.
pub async fn responses(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    proxy_inference(ctx, headers, body, InferenceEndpoint::Responses).await
}
