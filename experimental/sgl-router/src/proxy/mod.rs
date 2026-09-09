// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! HTTP proxy — forwards requests to the upstream SGLang worker.

pub mod sse;

use crate::health::circuit_breaker::CircuitBreaker;
use crate::server::error::ApiError;
use crate::server::header_utils::should_forward_request_header;
use anyhow::Context;
use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Response};
use bytes::Bytes;
use reqwest::{Client, Url};
use std::sync::Arc;
use std::time::Duration;

/// Cap on a non-2xx upstream body forwarded to the client. A worker validation
/// error on a `Union` field emits one entry per branch, each echoing `input`,
/// so one bad value can balloon into tens of KB that just repeats the request
/// back. Arbitrary value; it holds the useful prefix of every observed case.
const MAX_UPSTREAM_ERROR_BODY_BYTES: usize = 2048;

/// Truncate an upstream error body to [`MAX_UPSTREAM_ERROR_BODY_BYTES`].
///
/// The result is deliberately not valid JSON when it truncates: the marker
/// tells the client the body was cut here, rather than leaving it to conclude
/// the worker emitted malformed JSON. Cuts on a UTF-8 boundary so the prefix
/// stays decodable.
fn truncate_error_body(bytes: Bytes) -> Bytes {
    if bytes.len() <= MAX_UPSTREAM_ERROR_BODY_BYTES {
        return bytes;
    }
    let total = bytes.len();
    let mut end = MAX_UPSTREAM_ERROR_BODY_BYTES;
    // Back off off a continuation byte (0b10xxxxxx) to the codepoint start.
    while end > 0 && (bytes[end] & 0xC0) == 0x80 {
        end -= 1;
    }
    let mut out = Vec::with_capacity(end + 48);
    out.extend_from_slice(&bytes[..end]);
    out.extend_from_slice(format!("... [truncated, {total} bytes total]").as_bytes());
    Bytes::from(out)
}

/// Parse a worker URL emitted by discovery.  On failure, trip the worker's
/// circuit breaker so the malformed worker drops out of subsequent
/// `healthy_workers_for(...)` selection, then surface the error as
/// `ApiError::WorkerMisconfigured`.
fn parse_worker_url(worker_url: &str, breaker: &CircuitBreaker) -> Result<Url, ApiError> {
    Url::parse(worker_url).map_err(|e| {
        breaker.record_failure();
        ApiError::WorkerMisconfigured {
            worker: worker_url.to_string(),
            source: anyhow::Error::new(e).context("parse worker URL"),
        }
    })
}

#[derive(Debug)]
pub struct Proxy {
    pub client: Client,
    /// Wall-clock timeout applied to non-streaming upstream requests. Streaming
    /// requests deliberately do not use this (long generations are valid).
    pub request_timeout: Duration,
}

impl Proxy {
    /// Build a proxy. `request_timeout` is the per-request wall-clock budget for
    /// non-streaming forwards. Connect timeout is hard-coded to 5 s — even a
    /// streaming request fails fast at TCP setup if the worker is unreachable.
    pub fn new(request_timeout: Duration) -> Result<Self, anyhow::Error> {
        let client = Client::builder()
            .pool_max_idle_per_host(64)
            .connect_timeout(Duration::from_secs(5))
            .build()
            .context("build reqwest client")?;
        Ok(Self {
            client,
            request_timeout,
        })
    }

    /// Classify a reqwest error into the right `ApiError` variant, given an
    /// explicit worker URL. Called from the breaker-gated `forward_*_to`
    /// methods, which carry per-request worker URLs (not a single proxy-level
    /// URL).
    ///
    /// Walks the full source chain to detect timeouts, because reqwest wraps
    /// hyper which wraps `std::io::Error` — a top-level `is_timeout()` check
    /// misses both the wrapped reqwest timeout and the `io::ErrorKind::TimedOut`
    /// cases.
    fn classify_reqwest_error_for(worker: Url, e: reqwest::Error, path: &str) -> ApiError {
        let source = anyhow::Error::new(e).context(format!("worker {worker}: post {path}"));
        let is_timeout = source.chain().any(|c| {
            c.downcast_ref::<reqwest::Error>()
                .is_some_and(|r| r.is_timeout())
        }) || source.chain().any(|c| {
            c.downcast_ref::<std::io::Error>()
                .is_some_and(|io| io.kind() == std::io::ErrorKind::TimedOut)
        });
        if is_timeout {
            ApiError::UpstreamTimeout { worker }
        } else {
            ApiError::UpstreamUnreachable { worker, source }
        }
    }

    /// Breaker-gated JSON POST: checks `breaker.allow()` first, records
    /// success/failure based on response status, and returns
    /// `ApiError::BreakerOpen` immediately when the breaker is Open.
    ///
    /// `worker_url` is the discovery-emitted worker URL string. It's parsed
    /// to [`reqwest::Url`] internally so we can use [`Url::join`] for clean
    /// path concatenation (no double-slash) and pass a typed URL to the
    /// split error variants (`UpstreamUnreachable` / `UpstreamTimeout` /
    /// `UpstreamStatus`).
    pub async fn forward_json_to(
        &self,
        worker_url: &str,
        breaker: &CircuitBreaker,
        path: &str,
        headers: &HeaderMap,
        body: Bytes,
    ) -> Result<Response<Body>, ApiError> {
        if !breaker.allow() {
            return Err(ApiError::BreakerOpen {
                worker: worker_url.to_string(),
            });
        }
        let worker_url = parse_worker_url(worker_url, breaker)?;
        let url = worker_url.join(path).map_err(|e| {
            ApiError::Internal(anyhow::Error::new(e).context(format!("join worker path {path}")))
        })?;
        let mut req = self.client.post(url.clone()).body(body);
        for (k, v) in headers {
            if should_forward_request_header(k) {
                req = req.header(k, v);
            }
        }
        req = req
            .header("content-type", "application/json")
            .timeout(self.request_timeout);
        let resp = req.send().await.map_err(|e| {
            breaker.record_failure();
            Self::classify_reqwest_error_for(worker_url.clone(), e, path)
        })?;
        let status = resp.status();
        // Defer breaker recording until after the body completes — a
        // worker that returns 2xx headers and then drops mid-body is
        // still failing the request, and crediting it as healthy lets
        // a misbehaving worker stay eligible. For 5xx the early bail is
        // safe (no body to consume meaningfully), but we still wait
        // until after the read attempt to record exactly once.
        let bytes = match resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(
                    upstream = %url,
                    status = %status,
                    error = ?e,
                    "upstream dropped connection mid-body",
                );
                breaker.record_failure();
                return Err(ApiError::UpstreamStatus { status });
            }
        };
        if status.is_server_error() {
            breaker.record_failure();
        } else {
            breaker.record_success();
        }
        // Success bodies pass through untouched; only error payloads are capped.
        let bytes = if status.is_success() {
            bytes
        } else {
            truncate_error_body(bytes)
        };
        let mut out = Response::new(Body::from(bytes));
        *out.status_mut() = status;
        out.headers_mut().insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("application/json"),
        );
        Ok(out)
    }

    /// Breaker-gated streaming POST: checks `breaker.allow()` first, records
    /// success/failure, and returns `ApiError::BreakerOpen` when Open.
    ///
    /// `stream_guards` — when `Some`, the value is threaded into the SSE
    /// pump task and held for the entire body lifetime (headers → last byte
    /// / client disconnect).  The proxy does not inspect the boxed value; it
    /// relies entirely on `Drop` semantics, so callers typically pack
    /// `(LoadGuard, ActiveLoadGuard)` here. This keeps both the per-worker
    /// `active_requests` counter and the per-request active-load entry alive
    /// for the full streaming lifetime — without which a long-running SSE
    /// response would under-report load.
    // Each parameter is a distinct, required input to a single upstream
    // forward (target, breaker, path, headers, body, plus the two
    // streaming-lifetime callbacks). Bundling them into a struct purely to
    // satisfy the arg-count heuristic would add indirection without clarity.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward_streaming_to(
        &self,
        worker_url: &str,
        breaker: &Arc<CircuitBreaker>,
        path: &str,
        headers: &HeaderMap,
        body: Bytes,
        stream_guards: Option<Box<dyn Send + 'static>>,
        on_first_byte: Option<Box<dyn FnOnce() + Send + 'static>>,
    ) -> Result<Response<Body>, ApiError> {
        if !breaker.allow() {
            return Err(ApiError::BreakerOpen {
                worker: worker_url.to_string(),
            });
        }
        let worker_url = parse_worker_url(worker_url, breaker)?;
        let url = worker_url.join(path).map_err(|e| {
            ApiError::Internal(anyhow::Error::new(e).context(format!("join worker path {path}")))
        })?;
        let mut req = self.client.post(url.clone()).body(body);
        for (k, v) in headers {
            if should_forward_request_header(k) {
                req = req.header(k, v);
            }
        }
        req = req
            .header("content-type", "application/json")
            .header("accept", "text/event-stream");
        let resp = req.send().await.map_err(|e| {
            breaker.record_failure();
            Self::classify_reqwest_error_for(worker_url.clone(), e, path)
        })?;
        let status = resp.status();
        let upstream_ct = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/json")
            .to_string();
        let content_type = if status.is_success() {
            "text/event-stream".to_string()
        } else {
            upstream_ct
        };
        // A non-2xx body is an error payload, not a generation: collect and
        // truncate it rather than pumping it as SSE. Nothing here needs the
        // pump's hooks — there is no TTFT to record, and the guards have no
        // stream lifetime to stay alive for, so they drop on return.
        if !status.is_success() {
            if status.is_server_error() {
                breaker.record_failure();
            } else {
                breaker.record_success();
            }
            drop(stream_guards);
            let bytes = resp.bytes().await.unwrap_or_else(|e| {
                tracing::warn!(
                    upstream = %url,
                    status = %status,
                    error = ?e,
                    "failed reading upstream error body",
                );
                Bytes::new()
            });
            let mut out = Response::new(Body::from(truncate_error_body(bytes)));
            *out.status_mut() = status;
            out.headers_mut().insert(
                HeaderName::from_static("content-type"),
                HeaderValue::from_str(&content_type)
                    .unwrap_or_else(|_| HeaderValue::from_static("application/json")),
            );
            return Ok(out);
        }
        // Breaker recording is deferred to the pump's completion hook so
        // an upstream that returns 2xx headers and then drops mid-stream
        // is recorded as a failure.
        let on_complete: Option<Box<dyn FnOnce(bool) + Send + 'static>> = {
            let breaker_for_hook = Arc::clone(breaker);
            Some(Box::new(move |ok| {
                if ok {
                    breaker_for_hook.record_success();
                } else {
                    breaker_for_hook.record_failure();
                }
            }))
        };
        let body = sse::bytes_stream_to_body(
            resp.bytes_stream(),
            stream_guards,
            on_complete,
            on_first_byte,
        );
        let mut out = Response::new(body);
        *out.status_mut() = status;
        out.headers_mut().insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_str(&content_type)
                .unwrap_or_else(|_| HeaderValue::from_static("application/json")),
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn new_returns_result_not_panic() {
        let p = Proxy::new(Duration::from_secs(5)).unwrap();
        assert_eq!(p.request_timeout, Duration::from_secs(5));
    }

    /// Bodies up to the cap must round-trip byte-exact; the passthrough tests
    /// in `tests/proxy/chat_routing.rs` depend on it.
    #[test]
    fn error_body_up_to_cap_is_untouched() {
        for body in [
            Bytes::from(r#"{"error":{"message":"bad request"}}"#),
            Bytes::from(vec![b'x'; MAX_UPSTREAM_ERROR_BODY_BYTES]),
        ] {
            assert_eq!(truncate_error_body(body.clone()), body);
        }
    }

    #[test]
    fn oversized_error_body_is_truncated_with_marker() {
        let total = MAX_UPSTREAM_ERROR_BODY_BYTES * 4;
        let out = truncate_error_body(Bytes::from(vec![b'x'; total]));
        let text = String::from_utf8(out.to_vec()).expect("truncated body must stay UTF-8");
        assert!(text.starts_with("xxxx"), "prefix preserved: {text:.32}");
        assert!(
            text.ends_with(&format!("... [truncated, {total} bytes total]")),
            "marker must report the original size; got tail: {}",
            &text[text.len().saturating_sub(48)..],
        );
        assert!(out.len() < total, "must shrink: {} vs {total}", out.len());
    }

    /// A cut landing mid-codepoint must back off, or the client gets an
    /// undecodable tail.
    #[test]
    fn truncation_respects_utf8_boundaries() {
        // 3-byte chars do not divide evenly into the cap, so some cut lands
        // inside a codepoint regardless of alignment.
        let body: String = "错".repeat(MAX_UPSTREAM_ERROR_BODY_BYTES);
        let out = truncate_error_body(Bytes::from(body.into_bytes()));
        String::from_utf8(out.to_vec()).expect("truncated body must stay valid UTF-8");
    }
}
