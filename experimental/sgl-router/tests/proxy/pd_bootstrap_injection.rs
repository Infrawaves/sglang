// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! PD-disagg bootstrap-room injection + dual-dispatch — end-to-end
//! at the HTTP layer using MockWorkers.
//!
//! Asserts the router-side contract for SGLang disagg-prefill HTTP mode:
//!
//! * Every PD-mode `/v1/chat/completions` request fans out to BOTH a
//!   prefill and a decode worker (the prefill is `tokio::spawn`'d in
//!   the background; the decode is awaited for the client response).
//! * Both bodies carry the SAME flat top-level fields:
//!     - `bootstrap_host` = the chosen prefill worker's host
//!     - `bootstrap_port` = the chosen prefill worker's bootstrap port
//!     - `bootstrap_room` = a random u64 in `[0, i64::MAX]` (63-bit)
//! * Plain-mode requests do NOT carry any `bootstrap_*` field — the
//!   injection step is gated on `worker.mode() == Prefill`.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use bytes::Bytes;
use serde_json::{json, Value};
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tower::ServiceExt;

fn config() -> Config {
    Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: Some("tests/fixtures/tiny_tokenizer.json".into()),
            policy: PolicyKind::RoundRobin,
            decode_policy: Default::default(),
            bucket_config: None,
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
            affinity: None,
            fused: None,
            eligibility: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
    }
}

fn build_ctx(specs: Vec<WorkerSpec>) -> Arc<AppContext> {
    build_ctx_with_config(specs, config())
}

fn build_ctx_with_config(specs: Vec<WorkerSpec>, cfg: Config) -> Arc<AppContext> {
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    for s in specs {
        let _ = registry.add(s);
    }
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

fn chat_request() -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": "hi"}],
            }))
            .unwrap(),
        ))
        .unwrap()
}

/// Pattern-B dispatch: prefill is `tokio::spawn`'d as a detached task
/// so the client response can return as soon as decode is reachable —
/// the prefill body is captured *eventually* but may not be present
/// when the handler returns. Poll with a short bound rather than
/// sleeping a fixed duration.
async fn await_captured_body(
    mock: &crate::common::mock_worker::MockWorker,
    timeout: Duration,
    label: &str,
) -> Bytes {
    let start = Instant::now();
    loop {
        // Release the `std::sync::Mutex` guard before the sleep.await
        // (clippy: await_holding_lock).
        let captured = mock.captured.lock().unwrap().last_body.clone();
        if let Some(b) = captured {
            return b;
        }
        if start.elapsed() > timeout {
            panic!("{label}: no request body captured within {timeout:?}");
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

fn parse_body(b: &Bytes) -> Value {
    serde_json::from_slice(b).expect("body must be valid JSON")
}

fn assert_internal_rid(value: &Value) -> &str {
    let rid = value["rid"]
        .as_str()
        .expect("PD request must carry a scalar RID");
    let uuid = rid.strip_prefix("sgl-router-").expect("router RID prefix");
    assert_eq!(uuid.len(), 36, "internal RIDs must have fixed length");
    assert_eq!(uuid::Uuid::parse_str(uuid).unwrap().get_version_num(), 4);
    rid
}

/// Helper: extract bootstrap_host as &str.
fn bootstrap_host(v: &Value) -> Option<&str> {
    v.get("bootstrap_host").and_then(|x| x.as_str())
}
/// Helper: extract bootstrap_port as u16.
fn bootstrap_port(v: &Value) -> Option<u16> {
    v.get("bootstrap_port")
        .and_then(|x| x.as_u64())
        .map(|p| p as u16)
}
/// Helper: extract bootstrap_room as u64.
fn bootstrap_room(v: &Value) -> Option<u64> {
    v.get("bootstrap_room").and_then(|x| x.as_u64())
}

/// PD-mode chat fans out to BOTH prefill and decode with identical
/// bootstrap fields injected into both bodies.
#[tokio::test]
async fn pd_mode_chat_injects_bootstrap_fields_into_both_bodies() {
    let prefill = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx(vec![
        WorkerSpec {
            id: WorkerId("p1".into()),
            url: prefill.url.clone(),
            mode: WorkerMode::Prefill,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: Some(8997),
        },
        WorkerSpec {
            id: WorkerId("d1".into()),
            url: decode.url.clone(),
            mode: WorkerMode::Decode,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        },
    ]);
    let app = build_router(ctx);

    let res = app.oneshot(chat_request()).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK, "decode side should 200");

    let prefill_body = await_captured_body(&prefill, Duration::from_secs(2), "prefill").await;
    let decode_body = await_captured_body(&decode, Duration::from_secs(2), "decode").await;
    let pj = parse_body(&prefill_body);
    let dj = parse_body(&decode_body);
    assert_eq!(assert_internal_rid(&pj), assert_internal_rid(&dj));
    assert_eq!(pj["stream"], false);

    // Same bootstrap_room on both sides (one room minted per request).
    let p_room = bootstrap_room(&pj).expect("prefill body missing bootstrap_room");
    let d_room = bootstrap_room(&dj).expect("decode body missing bootstrap_room");
    assert_eq!(
        p_room, d_room,
        "prefill and decode must share the same bootstrap_room"
    );

    // Room must be in [0, i64::MAX]: the SGLang prefill stores it as
    // i64 internally, so values with the top bit set wrap negative.
    assert!(
        p_room <= i64::MAX as u64,
        "bootstrap_room {p_room} exceeds 63-bit range; SGLang would mis-store as negative i64",
    );

    // bootstrap_host on both sides == prefill worker's hostname
    // (MockWorker binds to 127.0.0.1).
    assert_eq!(bootstrap_host(&pj), Some("127.0.0.1"));
    assert_eq!(bootstrap_host(&dj), Some("127.0.0.1"));

    // bootstrap_port on both sides == prefill's configured bootstrap_port.
    assert_eq!(bootstrap_port(&pj), Some(8997));
    assert_eq!(bootstrap_port(&dj), Some(8997));
}

#[tokio::test]
async fn round_robin_pd_prefill_does_not_track_dispatch_timestamps() {
    let prefill =
        crate::common::mock_worker::MockWorker::start_hanging(Duration::from_millis(200)).await;
    let decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx(vec![
        WorkerSpec {
            id: WorkerId("p1".into()),
            url: prefill.url.clone(),
            mode: WorkerMode::Prefill,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: Some(8997),
        },
        WorkerSpec {
            id: WorkerId("d1".into()),
            url: decode.url.clone(),
            mode: WorkerMode::Decode,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        },
    ]);
    let prefill_worker = ctx
        .registry
        .workers_for(&ModelId("tiny".into()))
        .into_iter()
        .find(|worker| worker.id.0 == "p1")
        .expect("prefill worker is registered");
    let cutoff = Instant::now() - Duration::from_secs(1);
    let request = tokio::spawn(build_router(Arc::clone(&ctx)).oneshot(chat_request()));

    await_captured_body(&prefill, Duration::from_secs(2), "prefill").await;
    assert_eq!(prefill_worker.active_load(), 1);
    assert_eq!(prefill_worker.slots_acquired_since(cutoff), 0);

    assert_eq!(request.await.unwrap().unwrap().status(), StatusCode::OK);
}

/// Plain-mode (non-PD) requests do NOT carry any `bootstrap_*` field.
/// The injection step is gated on `worker.mode() == Prefill`; plain
/// workers serve the chat route directly without disagg bootstrapping.
#[tokio::test]
async fn plain_mode_chat_does_not_inject_bootstrap_fields() {
    let plain = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx(vec![WorkerSpec {
        id: WorkerId("w1".into()),
        url: plain.url.clone(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: None,
    }]);
    let app = build_router(ctx);

    let res = app.oneshot(chat_request()).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let body = await_captured_body(&plain, Duration::from_secs(2), "plain").await;
    let v = parse_body(&body);
    assert!(
        v.get("bootstrap_room").is_none(),
        "plain-mode request must not carry bootstrap_room; got {v}"
    );
    assert!(
        v.get("bootstrap_host").is_none(),
        "plain-mode request must not carry bootstrap_host; got {v}"
    );
    assert!(
        v.get("bootstrap_port").is_none(),
        "plain-mode request must not carry bootstrap_port; got {v}"
    );
}

/// PD-mode with multiple prefill workers + different `bootstrap_port`
/// values: the bootstrap_port injected MUST match the actually-chosen
/// prefill (not e.g. the first registered or a global config value).
#[tokio::test]
async fn pd_mode_bootstrap_port_matches_chosen_prefill_worker() {
    let prefill_a = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let prefill_b = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx(vec![
        WorkerSpec {
            id: WorkerId("pA".into()),
            url: prefill_a.url.clone(),
            mode: WorkerMode::Prefill,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: Some(11111),
        },
        WorkerSpec {
            id: WorkerId("pB".into()),
            url: prefill_b.url.clone(),
            mode: WorkerMode::Prefill,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: Some(22222),
        },
        WorkerSpec {
            id: WorkerId("d1".into()),
            url: decode.url.clone(),
            mode: WorkerMode::Decode,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        },
    ]);
    let app = build_router(ctx);

    // Fire enough requests to ensure round-robin hits both prefill workers.
    for _ in 0..6 {
        let res = app.clone().oneshot(chat_request()).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    // Wait until both prefill workers have captured at least one body.
    let body_a = await_captured_body(&prefill_a, Duration::from_secs(2), "prefill_a").await;
    let body_b = await_captured_body(&prefill_b, Duration::from_secs(2), "prefill_b").await;
    let va = parse_body(&body_a);
    let vb = parse_body(&body_b);
    // Each prefill must see its OWN bootstrap_port — never the other's.
    assert_eq!(
        bootstrap_port(&va),
        Some(11111),
        "prefill_a body should carry its own bootstrap_port"
    );
    assert_eq!(
        bootstrap_port(&vb),
        Some(22222),
        "prefill_b body should carry its own bootstrap_port"
    );
}

async fn assert_prefill_failure_cancels_decode(
    prefill_status: StatusCode,
    expected_status: StatusCode,
    streaming: bool,
    scheduler_error: bool,
) {
    let decode =
        crate::common::mock_worker::MockWorker::start_hanging(Duration::from_secs(30)).await;
    let error = json!({"error": {
        "message": "simulated scheduler failure",
        "type": "InternalServerError",
        "code": prefill_status.as_u16(),
    }});
    let prefill = if scheduler_error {
        crate::common::mock_worker::MockWorker::start_scheduler_error_after_peer(
            prefill_status,
            error,
            Some(Arc::clone(&decode.captured)),
        )
        .await
    } else {
        crate::common::mock_worker::MockWorker::start_error_after_peer(
            prefill_status,
            error,
            Some(Arc::clone(&decode.captured)),
        )
        .await
    };
    let mut cfg = config();
    cfg.model.tokenizer_path = None; // Preserve the online --no-tokenizer path.
    let ctx = build_ctx_with_config(
        vec![
            WorkerSpec {
                id: WorkerId("p1".into()),
                url: prefill.url.clone(),
                mode: WorkerMode::Prefill,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: Some(8997),
            },
            WorkerSpec {
                id: WorkerId("d1".into()),
                url: decode.url.clone(),
                mode: WorkerMode::Decode,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: None,
            },
        ],
        cfg,
    );
    let app = build_router(Arc::clone(&ctx));

    // Client must see the prefill failure instead of waiting for decode's
    // bootstrap timeout.
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model":"tiny", "rid":"caller-pd-rid", "stream": streaming,
            "stream_options":{"include_usage":true},
            "messages":[{"role":"user","content":"hi"}]})
            .to_string(),
        ))
        .unwrap();
    let res = tokio::time::timeout(Duration::from_secs(2), app.oneshot(req))
        .await
        .expect("prefill failure must not wait for decode timeout")
        .unwrap();
    assert_eq!(res.status(), expected_status);

    // Decode received its body (proves dual dispatch fired) and then got a
    // targeted cancellation for the same request rid.
    let decode_body = await_captured_body(&decode, Duration::from_secs(2), "decode").await;
    let v = parse_body(&decode_body);
    assert_eq!(bootstrap_port(&v), Some(8997));
    let rid = assert_internal_rid(&v);
    assert_eq!(v["stream"], streaming);
    assert_eq!(v["stream_options"], json!({"include_usage":true}));
    let start = Instant::now();
    loop {
        let aborts = decode.captured.lock().unwrap().abort_rids.clone();
        if aborts.iter().any(|aborted| aborted == rid)
            && ctx
                .metrics
                .render()
                .contains(r#"sgl_router_pd_decode_abort_requests_total{outcome="success"} 1"#)
        {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "decode did not receive abort for rid {rid}"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let metrics = ctx.metrics.render();
    assert!(metrics.contains(r#"sgl_router_pd_decode_abort_requests_total{outcome="success"} 1"#));

    // Prefill also received its body — it just returned 5xx. The
    // bootstrap fields are present so the engine WOULD have honoured
    // the bootstrap_room if the mock had succeeded.
    let prefill_body = await_captured_body(&prefill, Duration::from_secs(2), "prefill").await;
    let pv = parse_body(&prefill_body);
    assert_eq!(bootstrap_port(&pv), Some(8997));
    assert_eq!(pv["rid"], v["rid"]);
    assert_ne!(rid, "caller-pd-rid");
    assert_eq!(pv["stream"], false);
    assert!(pv.get("stream_options").is_none());
    assert_eq!(decode.captured.lock().unwrap().abort_rids.len(), 1);
}

/// A prefill server failure maps to 502 and cancels the paired decode request.
#[tokio::test]
async fn pd_mode_prefill_5xx_cancels_decode() {
    assert_prefill_failure_cancels_decode(
        StatusCode::INTERNAL_SERVER_ERROR,
        StatusCode::BAD_GATEWAY,
        false,
        false,
    )
    .await;
}

/// A prefill client rejection keeps its 4xx status and still cancels decode,
/// because decode cannot complete without a successful prefill.
#[tokio::test]
async fn pd_mode_prefill_4xx_cancels_decode() {
    assert_prefill_failure_cancels_decode(
        StatusCode::BAD_REQUEST,
        StatusCode::BAD_REQUEST,
        false,
        false,
    )
    .await;
}

#[tokio::test]
async fn pd_stream_prefill_5xx_cancels_decode() {
    assert_prefill_failure_cancels_decode(
        StatusCode::INTERNAL_SERVER_ERROR,
        StatusCode::BAD_GATEWAY,
        true,
        false,
    )
    .await;
}

#[tokio::test]
async fn pd_stream_prefill_4xx_cancels_decode() {
    assert_prefill_failure_cancels_decode(
        StatusCode::BAD_REQUEST,
        StatusCode::BAD_REQUEST,
        true,
        false,
    )
    .await;
}

#[tokio::test]
async fn pd_stream_scheduler_500_cancels_decode() {
    assert_prefill_failure_cancels_decode(
        StatusCode::INTERNAL_SERVER_ERROR,
        StatusCode::BAD_GATEWAY,
        true,
        true,
    )
    .await;
}

#[tokio::test]
async fn pd_stream_scheduler_400_cancels_decode() {
    assert_prefill_failure_cancels_decode(
        StatusCode::BAD_REQUEST,
        StatusCode::BAD_REQUEST,
        true,
        true,
    )
    .await;
}

#[tokio::test]
async fn pd_prefill_non_streaming_preserves_decode_sse() {
    const CHUNK: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n";
    const DONE: &str = "data: [DONE]\n\n";
    let prefill = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode = crate::common::mock_worker::MockWorker::start(vec![CHUNK, DONE]).await;
    let ctx = build_ctx(vec![
        WorkerSpec {
            id: WorkerId("p1".into()),
            url: prefill.url.clone(),
            mode: WorkerMode::Prefill,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: Some(8997),
        },
        WorkerSpec {
            id: WorkerId("d1".into()),
            url: decode.url.clone(),
            mode: WorkerMode::Decode,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        },
    ]);
    let app = build_router(Arc::clone(&ctx));
    let mut previous_rid = None;
    for _ in 0..2 {
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model":"tiny", "rid":"same-caller-rid", "stream":true,
                    "stream_options":{"include_usage":true},
                    "messages":[{"role":"user","content":"hi"}],
                })
                .to_string(),
            ))
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["content-type"], "text/event-stream");
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(body, format!("{CHUNK}{DONE}"));
        let p = parse_body(&await_captured_body(&prefill, Duration::from_secs(2), "prefill").await);
        let d = parse_body(&await_captured_body(&decode, Duration::from_secs(2), "decode").await);
        assert_eq!(p["stream"], false);
        assert!(p.get("stream_options").is_none());
        assert_eq!(d["stream"], true);
        assert_eq!(d["stream_options"], json!({"include_usage":true}));
        assert_eq!(assert_internal_rid(&p), assert_internal_rid(&d));
        assert_ne!(d["rid"], "same-caller-rid");
        assert_ne!(previous_rid.as_ref(), Some(&d["rid"]));
        previous_rid = Some(d["rid"].clone());
    }
    assert!(decode.captured.lock().unwrap().abort_rids.is_empty());

    // PD dispatch must retain the base router's stream-end accounting for
    // the Decode worker that actually sends the client-visible SSE stream.
    let expected = format!(
        r#"sgl_router_stream_outcome_total{{worker_url="{}",model_id="tiny",outcome="ok"}} 2"#,
        decode.url,
    );
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if ctx.metrics.render().lines().any(|line| line == expected) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("both Decode streams must be counted");
}

/// Match SGLang's prefix-based abort while two Decode requests are active.
/// The sibling must remain pending until explicitly completed by the test.
#[tokio::test]
async fn pd_abort_does_not_cancel_another_callers_rid_prefix() {
    use axum::{extract::State, response::IntoResponse, routing::post, Json, Router};
    use std::collections::HashMap;
    use std::sync::Mutex;
    use tokio::sync::oneshot;

    #[derive(Default)]
    struct DecodeState {
        pending: HashMap<String, oneshot::Sender<StatusCode>>,
        bodies: Vec<Value>,
        aborted: Vec<String>,
    }
    let state = Arc::new(Mutex::new(DecodeState::default()));
    let decode_app = Router::new()
        .route(
            "/v1/chat/completions",
            post(
                |State(state): State<Arc<Mutex<DecodeState>>>, Json(body): Json<Value>| async move {
                    let rid = body["rid"].as_str().unwrap().to_string();
                    let (tx, rx) = oneshot::channel();
                    {
                        let mut state = state.lock().unwrap();
                        state.bodies.push(body);
                        assert!(state.pending.insert(rid, tx).is_none());
                    }
                    let status = rx.await.unwrap();
                    (
                        status,
                        Json(json!({"choices":[{"message":{"content":"ok"}}]})),
                    )
                },
            ),
        )
        .route(
            "/abort_request",
            post(
                |State(state): State<Arc<Mutex<DecodeState>>>, Json(body): Json<Value>| async move {
                    let prefix = body["rid"].as_str().unwrap();
                    let mut state = state.lock().unwrap();
                    let matches: Vec<_> = state
                        .pending
                        .keys()
                        .filter(|rid| rid.starts_with(prefix))
                        .cloned()
                        .collect();
                    for rid in matches {
                        state.aborted.push(rid.clone());
                        let _ = state
                            .pending
                            .remove(&rid)
                            .unwrap()
                            .send(StatusCode::INTERNAL_SERVER_ERROR);
                    }
                    StatusCode::OK
                },
            ),
        )
        .with_state(Arc::clone(&state));
    let prefill_app = Router::new()
        .route(
            "/v1/chat/completions",
            post(
                |State(state): State<Arc<Mutex<DecodeState>>>, Json(body): Json<Value>| async move {
                    // Both decode requests must have arrived before the failure.
                    tokio::time::timeout(Duration::from_secs(2), async {
                        loop {
                            if state.lock().unwrap().bodies.len() == 2 {
                                break;
                            }
                            tokio::time::sleep(Duration::from_millis(5)).await;
                        }
                    })
                    .await
                    .unwrap();
                    let decode_body = state
                        .lock()
                        .unwrap()
                        .bodies
                        .iter()
                        .find(|value| value["messages"] == body["messages"])
                        .unwrap()
                        .clone();
                    assert_eq!(body["rid"], decode_body["rid"]);
                    if body["messages"][0]["content"] == "fail" {
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(json!({"error":"prefill failed"})),
                        )
                            .into_response()
                    } else {
                        Json(json!({})).into_response()
                    }
                },
            ),
        )
        .with_state(Arc::clone(&state));
    let p_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let d_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let p_url = format!("http://{}", p_listener.local_addr().unwrap());
    let d_url = format!("http://{}", d_listener.local_addr().unwrap());
    let p_task = tokio::spawn(async move { axum::serve(p_listener, prefill_app).await.unwrap() });
    let d_task = tokio::spawn(async move { axum::serve(d_listener, decode_app).await.unwrap() });
    // Ensure the listeners stop even if an assertion panics.
    struct TestServers([tokio::task::JoinHandle<()>; 2]);
    impl Drop for TestServers {
        fn drop(&mut self) {
            for task in &self.0 {
                task.abort();
            }
        }
    }
    let _servers = TestServers([p_task, d_task]);
    let mut cfg = config();
    cfg.model.tokenizer_path = None;
    let ctx = build_ctx_with_config(
        vec![
            WorkerSpec {
                id: WorkerId("p1".into()),
                url: p_url,
                mode: WorkerMode::Prefill,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: Some(8997),
            },
            WorkerSpec {
                id: WorkerId("d1".into()),
                url: d_url,
                mode: WorkerMode::Decode,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: None,
            },
        ],
        cfg,
    );
    let app = build_router(Arc::clone(&ctx));
    let request = |rid: &str, content: &str| {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({"model":"tiny", "rid":rid, "stream":false,
            "messages":[{"role":"user","content":content}]})
                .to_string(),
            ))
            .unwrap()
    };
    let failed = tokio::spawn(app.clone().oneshot(request("request-1", "fail")));
    let sibling = tokio::spawn(app.oneshot(request("request-10", "succeed")));
    let response = tokio::time::timeout(Duration::from_secs(2), failed)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if ctx
                .metrics
                .render()
                .contains(r#"sgl_router_pd_decode_abort_requests_total{outcome="success"} 1"#)
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    {
        let mut state = state.lock().unwrap();
        assert_eq!(state.bodies.len(), 2);
        assert_eq!(
            state.aborted.len(),
            1,
            "abort must only match the failed request"
        );
        assert_eq!(
            state.pending.len(),
            1,
            "sibling must remain active after abort"
        );
        let failed_body = state
            .bodies
            .iter()
            .find(|b| b["messages"][0]["content"] == "fail")
            .unwrap();
        let failed_rid = assert_internal_rid(failed_body).to_string();
        let sibling_body = state
            .bodies
            .iter()
            .find(|b| b["messages"][0]["content"] == "succeed")
            .unwrap();
        let sibling_rid = assert_internal_rid(sibling_body).to_string();
        assert_ne!(failed_rid, sibling_rid);
        assert_eq!(state.aborted, vec![failed_rid]);
        state
            .pending
            .remove(&sibling_rid)
            .unwrap()
            .send(StatusCode::OK)
            .unwrap();
    }
    let response = tokio::time::timeout(Duration::from_secs(2), sibling)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn responses_pd_dispatch_uses_request_id_and_preserves_decode_streaming() {
    use http_body_util::BodyExt;

    for streaming in [false, true] {
        let prefill = crate::common::mock_worker::MockWorker::start(vec![]).await;
        let decode = crate::common::mock_worker::MockWorker::start(vec![
            "event: response.completed\ndata: {\"type\":\"response.completed\"}\n\n",
        ]).await;
        let ctx = build_ctx(vec![
            WorkerSpec {
                id: WorkerId("p1".into()), url: prefill.url.clone(), mode: WorkerMode::Prefill,
                model_ids: vec![ModelId("tiny".into())], bootstrap_port: Some(8997),
            },
            WorkerSpec {
                id: WorkerId("d1".into()), url: decode.url.clone(), mode: WorkerMode::Decode,
                model_ids: vec![ModelId("tiny".into())], bootstrap_port: None,
            },
        ]);
        let req = Request::builder().method("POST").uri("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(json!({
                "model": "tiny", "input": "hi", "stream": streaming, "request_id": "caller-id"
            }).to_string())).unwrap();
        let res = build_router(ctx).oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let _ = res.into_body().collect().await.unwrap();
        let p = parse_body(&await_captured_body(&prefill, Duration::from_secs(2), "prefill").await);
        let d = parse_body(&await_captured_body(&decode, Duration::from_secs(2), "decode").await);
        assert_eq!(p["bootstrap_room"], d["bootstrap_room"]);
        assert_eq!(p["bootstrap_host"], "127.0.0.1");
        assert_eq!(p["bootstrap_port"], 8997);
        assert_eq!(p["request_id"], d["request_id"]);
        assert!(p["request_id"].as_str().unwrap().starts_with("sgl-router-"));
        assert_ne!(p["request_id"], "caller-id");
        assert_eq!(p["stream"], false);
        assert_eq!(d["stream"], streaming);
        for value in [&p, &d] {
            assert_eq!(value["input"], "hi");
            assert!(value.get("rid").is_none());
            assert!(value.get("input_ids").is_none());
        }
    }
}
