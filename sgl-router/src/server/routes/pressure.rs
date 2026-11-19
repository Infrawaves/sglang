// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::server::app_context::AppContext;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use std::sync::Arc;
use std::time::Instant;

/// Latest [`ColdPressureSample`](crate::state::load_monitor::cold_pressure::ColdPressureSample)
/// as JSON; 503 whenever it is missing, stale or without data, so a gateway can trust any 200.
pub async fn prefill_pressure(State(ctx): State<Arc<AppContext>>) -> Response {
    let interval_secs = ctx.config.server.cold_pressure.sample_interval_secs;
    match ctx.cold_pressure.latest() {
        Some(sample) if sample.is_stale(interval_secs, Instant::now()) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "cold pressure sample is stale",
                "sampled_at_unix_ms": sample.sampled_at_unix_ms,
            })),
        )
            .into_response(),
        Some(sample) if sample.has_data() => Json(sample).into_response(),
        Some(sample) => {
            let error = if sample.healthy_prefill_workers == 0 {
                "no routable prefill worker"
            } else {
                "no healthy prefill worker has reported load yet"
            };
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": error,
                    "healthy_prefill_workers": sample.healthy_prefill_workers,
                    "unreported_workers": sample.unreported_workers,
                    "overdue_workers": sample.overdue_workers,
                    "sampled_at_unix_ms": sample.sampled_at_unix_ms,
                })),
            )
                .into_response()
        }
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "no cold pressure sample yet"})),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::load_monitor::cold_pressure::unix_ms;
    use crate::state::load_monitor::cold_pressure::{ColdPressureSample, STALE_AFTER_INTERVALS};
    use axum::body::Body;
    use axum::http::Request;
    use std::time::Duration;
    use tower::ServiceExt;

    async fn get(ctx: Arc<AppContext>) -> (StatusCode, serde_json::Value) {
        let res = crate::server::app::build_router(ctx)
            .oneshot(
                Request::builder()
                    .uri("/prefill_pressure")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = res.status();
        let body = axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, serde_json::from_slice(&body).unwrap())
    }

    #[tokio::test]
    async fn unavailable_before_the_first_sample() {
        let (status, body) = get(Arc::new(AppContext::stub())).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(body["error"].is_string(), "{body}");
    }

    #[tokio::test]
    async fn serves_the_latest_sample() {
        let ctx = AppContext::stub();
        let now = unix_ms();
        ctx.cold_pressure.store(ColdPressureSample {
            pressure: 0.5,
            cold_tokens: 100_000,
            capped_cold_tokens: 100_000,
            capacity_tokens: 200_000,
            per_worker_capacity_tokens: 20_000,
            healthy_prefill_workers: 10,
            stale_workers: 1,
            unreported_workers: 2,
            overdue_workers: 1,
            sampled_at_unix_ms: now,
            sampled_at: Instant::now(),
        });
        let (status, body) = get(Arc::new(ctx)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["pressure"], 0.5);
        assert_eq!(body["cold_tokens"], 100_000);
        assert_eq!(body["capped_cold_tokens"], 100_000);
        assert_eq!(body["capacity_tokens"], 200_000);
        assert_eq!(body["per_worker_capacity_tokens"], 20_000);
        assert_eq!(body["healthy_prefill_workers"], 10);
        assert_eq!(body["stale_workers"], 1);
        assert_eq!(body["unreported_workers"], 2);
        assert_eq!(body["overdue_workers"], 1);
        assert_eq!(body["sampled_at_unix_ms"], now);
    }

    #[tokio::test]
    async fn unavailable_while_no_worker_has_reported() {
        let ctx = AppContext::stub();
        let now = unix_ms();
        ctx.cold_pressure.store(ColdPressureSample {
            pressure: 1.0,
            cold_tokens: 0,
            capped_cold_tokens: 0,
            capacity_tokens: 0,
            per_worker_capacity_tokens: 20_000,
            healthy_prefill_workers: 3,
            stale_workers: 0,
            unreported_workers: 3,
            overdue_workers: 0,
            sampled_at_unix_ms: now,
            sampled_at: Instant::now(),
        });
        let (status, body) = get(Arc::new(ctx)).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["unreported_workers"], 3);
        assert!(body.get("pressure").is_none(), "{body}");
    }

    /// A stopped sampler must not keep serving its last value as a 200.
    #[tokio::test]
    async fn unavailable_once_the_sample_is_stale() {
        let ctx = AppContext::stub();
        let interval_ms = ctx.config.server.cold_pressure.sample_interval_secs * 1_000;
        let sampled_at = unix_ms() - (STALE_AFTER_INTERVALS * interval_ms + 1_000);
        ctx.cold_pressure.store(ColdPressureSample {
            pressure: 0.5,
            cold_tokens: 100_000,
            capped_cold_tokens: 100_000,
            capacity_tokens: 200_000,
            per_worker_capacity_tokens: 20_000,
            healthy_prefill_workers: 10,
            stale_workers: 0,
            unreported_workers: 0,
            overdue_workers: 0,
            sampled_at_unix_ms: sampled_at,
            sampled_at: Instant::now()
                - Duration::from_millis(STALE_AFTER_INTERVALS * interval_ms + 1_000),
        });
        let (status, body) = get(Arc::new(ctx)).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["sampled_at_unix_ms"], sampled_at);
        assert!(body.get("pressure").is_none(), "{body}");
    }

    async fn metrics_body(ctx: Arc<AppContext>) -> String {
        let res = crate::server::app::build_router(ctx)
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .unwrap();
        String::from_utf8(body.to_vec()).unwrap()
    }

    /// /metrics must stop exporting exactly when GET /prefill_pressure stops
    /// serving, or Prometheus stamps a frozen value as current.
    #[tokio::test]
    async fn metrics_and_endpoint_agree_on_staleness() {
        let interval_ms = AppContext::stub()
            .config
            .server
            .cold_pressure
            .sample_interval_secs
            * 1_000;
        let max_age_ms = STALE_AFTER_INTERVALS * interval_ms;
        for (age_ms, fresh) in [(0, true), (max_age_ms + 1_000, false)] {
            let ctx = Arc::new(AppContext::stub());
            ctx.cold_pressure.store(ColdPressureSample {
                pressure: 0.25,
                cold_tokens: 5_000,
                capped_cold_tokens: 5_000,
                capacity_tokens: 20_000,
                per_worker_capacity_tokens: 20_000,
                healthy_prefill_workers: 1,
                stale_workers: 0,
                unreported_workers: 0,
                overdue_workers: 0,
                sampled_at_unix_ms: unix_ms() - age_ms,
                sampled_at: Instant::now() - Duration::from_millis(age_ms),
            });
            let (status, _) = get(Arc::clone(&ctx)).await;
            let metrics = metrics_body(ctx).await;
            assert_eq!(status == StatusCode::OK, fresh, "age {age_ms}ms");
            for family in [
                "sgl_router_cold_pressure ",
                "sgl_router_cold_pressure_stale_workers ",
                "sgl_router_cold_pressure_unreported_workers ",
            ] {
                assert_eq!(
                    metrics.contains(family),
                    fresh,
                    "{family} at age {age_ms}ms"
                );
            }
        }
    }

    #[tokio::test]
    async fn unavailable_without_a_routable_prefill_worker() {
        let ctx = AppContext::stub();
        ctx.cold_pressure.store(ColdPressureSample {
            pressure: 1.0,
            cold_tokens: 0,
            capped_cold_tokens: 0,
            capacity_tokens: 0,
            per_worker_capacity_tokens: 20_000,
            healthy_prefill_workers: 0,
            stale_workers: 0,
            unreported_workers: 0,
            overdue_workers: 0,
            sampled_at_unix_ms: unix_ms(),
            sampled_at: Instant::now(),
        });
        let (status, body) = get(Arc::new(ctx)).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"], "no routable prefill worker");
        assert!(body.get("pressure").is_none(), "{body}");
    }

    /// A wall-clock step must not decide staleness; only the monotonic instant does.
    #[tokio::test]
    async fn staleness_ignores_the_wall_clock() {
        let ctx = AppContext::stub();
        ctx.cold_pressure.store(ColdPressureSample {
            pressure: 0.5,
            cold_tokens: 10_000,
            capped_cold_tokens: 10_000,
            capacity_tokens: 20_000,
            per_worker_capacity_tokens: 20_000,
            healthy_prefill_workers: 1,
            stale_workers: 0,
            unreported_workers: 0,
            overdue_workers: 0,
            sampled_at_unix_ms: unix_ms() - 3_600_000,
            sampled_at: Instant::now(),
        });
        let (status, _) = get(Arc::new(ctx)).await;
        assert_eq!(status, StatusCode::OK);
    }
}
