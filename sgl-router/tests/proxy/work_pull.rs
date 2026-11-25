// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::common::mock_worker::MockWorker;
use axum::{body::Body, http::Request, Router};
use clap::Parser;
use serde_json::{json, Value};
use sgl_router::{
    buckets_reorg::{Bucket, BucketGroups, BucketResolver, EngineGroup},
    config::Cli,
    discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec},
    health::circuit_breaker::CircuitBreakerConfig,
    policies::factory::build_registry_with_defaults,
    policies_reorg::admission::{Decision, EngineAdmission, EngineMetrics},
    policies_reorg::power_of_two::PowerOfTwoPolicy,
    policies_reorg::PickError,
    proxy::Proxy,
    server::{
        app::build_router,
        app_context::{AppContext, ChatRouting},
    },
    tokenizer::TokenizerRegistry,
    workers::{Worker, WorkerRegistry},
};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use tower::ServiceExt;

struct Fixture {
    app: Router,
    prefill: MockWorker,
    decode: MockWorker,
    registry: Arc<WorkerRegistry>,
}
impl Fixture {
    async fn new(lease_ms: u64, max_requests: usize) -> Self {
        Self::with_decode_admission(lease_ms, max_requests, None).await
    }
    async fn with_decode_admission(
        lease_ms: u64,
        max_requests: usize,
        admission: Option<Arc<dyn EngineAdmission>>,
    ) -> Self {
        Self::configured(lease_ms, max_requests, admission, None).await
    }
    async fn configured(
        lease_ms: u64,
        max_requests: usize,
        admission: Option<Arc<dyn EngineAdmission>>,
        tokenizer: Option<&str>,
    ) -> Self {
        Self::with_decode(
            MockWorker::start(vec![]).await,
            true,
            lease_ms,
            max_requests,
            admission,
            tokenizer,
            CircuitBreakerConfig::default(),
        )
        .await
    }
    async fn with_decode(
        decode: MockWorker,
        stream_after_prefill: bool,
        lease_ms: u64,
        max_requests: usize,
        admission: Option<Arc<dyn EngineAdmission>>,
        tokenizer: Option<&str>,
        prefill_breaker: CircuitBreakerConfig,
    ) -> Self {
        let prefill = MockWorker::start(vec![]).await;
        let mut cfg = Cli::try_parse_from([
            "sgl-router",
            "--model-id",
            "tiny",
            "--worker-urls",
            &prefill.url,
            &decode.url,
            "--no-tokenizer",
            "--chat-routing",
            "reorg",
            "--prefill-dispatch-mode",
            "pull",
            "--pull-control-api-key",
            "pull-test-secret",
        ])
        .unwrap()
        .into_config()
        .unwrap();
        cfg.proxy.pull.offer_timeout_ms = lease_ms;
        cfg.proxy.pull.max_requests = max_requests;
        cfg.proxy.stream_after_prefill = stream_after_prefill;
        cfg.proxy.request_timeout_secs = 3;
        if let Some(path) = tokenizer {
            cfg.model.tokenizer_path = Some(path.to_owned());
            // Pull must forward ids even when text forwarding is disabled.
            cfg.model.disable_input_ids_forwarding = true;
        }
        let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
        let registry = Arc::new(WorkerRegistry::default());
        for (id, url, mode) in [
            ("p", &prefill.url, WorkerMode::Prefill),
            ("d", &decode.url, WorkerMode::Decode),
        ] {
            registry
                .add_with_cb(
                    WorkerSpec {
                        id: WorkerId(id.into()),
                        url: url.clone(),
                        mode,
                        model_ids: vec![ModelId("tiny".into())],
                        bootstrap_port: Some(8997),
                        ..Default::default()
                    },
                    Some(if mode == WorkerMode::Prefill {
                        prefill_breaker.clone()
                    } else {
                        CircuitBreakerConfig::default()
                    }),
                    sgl_router::workers::WireProtocol::default(),
                )
                .unwrap();
        }
        let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
        let mut ctx = AppContext::new(
            cfg,
            tokenizers,
            Arc::new(Proxy::new(Duration::from_secs(3)).unwrap()),
            registry.clone(),
            policies,
        );
        let policy = Arc::new(PowerOfTwoPolicy::new(ctx.engine_reported_load.clone()));
        let mut decode_policy = PowerOfTwoPolicy::new(ctx.engine_reported_load.clone());
        if let Some(admission) = admission {
            decode_policy.admission = admission;
        }
        ctx.chat_routing = ChatRouting::Reorg(HashMap::from([(
            ModelId("tiny".into()),
            BucketResolver::new(vec![Bucket::new(
                "pd",
                BucketGroups::Pd {
                    prefill: EngineGroup::new(policy.clone()),
                    decode: EngineGroup::new(Arc::new(decode_policy)),
                },
            )])
            .unwrap(),
        )]));
        Self {
            app: build_router(Arc::new(ctx)),
            prefill,
            decode,
            registry,
        }
    }
    fn generate(&self) -> tokio::task::JoinHandle<axum::response::Response> {
        let app = self.app.clone();
        tokio::spawn(async move {
            app.oneshot(request("/generate", json!({"input_ids":[1,2,3], "sampling_params":{"max_new_tokens":1}, "stream":false}))).await.unwrap()
        })
    }
    async fn acquire(&self, poll: &str) -> Value {
        self.acquire_with_limit(poll, 1).await
    }
    async fn acquire_with_limit(&self, poll: &str, max_offered: usize) -> Value {
        self.post(
            "acquire",
            json!({
                "worker_url": self.prefill.url, "worker_epoch": "epoch-1", "dp_rank": 0,
                "poll_id": poll, "max_tokens": 100, "max_requests": 1,
                "max_offered": max_offered, "wait_ms": 500
            }),
        )
        .await
    }
    async fn post(&self, route: &str, body: Value) -> Value {
        let response = self
            .app
            .clone()
            .oneshot(request(&format!("/internal/pull/{route}"), body))
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 16 * 1024 * 1024)
            .await
            .unwrap();
        assert!(
            status.is_success(),
            "{status}: {}",
            String::from_utf8_lossy(&bytes)
        );
        serde_json::from_slice(&bytes).unwrap()
    }
    fn action(offer: &Value) -> Value {
        json!({"assignment_id":offer["assignment_id"],"worker_epoch":"epoch-1"})
    }
    async fn outcome(&self, offer: &Value, status: u16, body: &str) -> Value {
        let mut value = Self::action(offer);
        value["status_code"] = json!(status);
        value["body"] = json!(body);
        self.post("outcome", value).await
    }
}

#[derive(Debug)]
struct DecodeAdmission(AtomicBool);

impl EngineAdmission for DecodeAdmission {
    fn check(&self, _: &Worker, _: &EngineMetrics) -> Result<Decision, PickError> {
        Ok(if self.0.load(Ordering::Relaxed) {
            Decision::Allow
        } else {
            Decision::Reject("decode full".into())
        })
    }
}

#[tokio::test]
async fn pull_waits_in_router_until_decode_can_admit_the_assignment() {
    let admission = Arc::new(DecodeAdmission(AtomicBool::new(false)));
    let f = Fixture::with_decode_admission(1000, 10, Some(admission.clone())).await;
    let mut client = f.generate();
    assert!(tokio::time::timeout(Duration::from_millis(30), &mut client)
        .await
        .is_err());
    // A full D cannot consume the P's offer: acquisition returns empty and the
    // client request remains queued, with no execution on either worker.
    assert_eq!(f.acquire("full").await["status"], "empty");
    assert!(!client.is_finished());
    assert!(f.prefill.captured.lock().unwrap().last_body.is_none());
    assert!(f.decode.captured.lock().unwrap().last_body.is_none());
    admission.0.store(true, Ordering::Relaxed);
    let offer = f.acquire("available").await;
    assert_eq!(offer["status"], "offer");
    f.post("accept", Fixture::action(&offer)).await;
    f.outcome(&offer, 200, "{}").await;
    assert_eq!(client.await.unwrap().status(), 200);
}
fn request(path: &str, value: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .header("authorization", "Bearer pull-test-secret")
        .body(Body::from(value.to_string()))
        .unwrap()
}

#[tokio::test]
async fn pull_decode_failure_still_cancels_its_committed_prefill() {
    // A read-to-the-end D error kept P committed: nobody would take its KV,
    // and P held its budget until the bootstrap timeout.
    let decode = MockWorker::start_returning_error(
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        json!({"error": "decode failed"}),
    )
    .await;
    // Without stream_after_prefill, D's error reaches the client while P
    // still waits for D to take its KV.
    let f = Fixture::with_decode(
        decode,
        false,
        1000,
        10,
        None,
        None,
        CircuitBreakerConfig::default(),
    )
    .await;
    let client = f.generate();
    let offer = f.acquire("poll-1").await;
    f.post("accept", Fixture::action(&offer)).await;
    let response = client.await.unwrap();
    assert_eq!(response.status(), 500);
    axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(
        f.post("status", Fixture::action(&offer)).await["status"],
        "cancelled"
    );
}

#[tokio::test]
async fn pull_prefill_failures_open_its_breaker_and_stop_its_claims() {
    // Pull outcomes bypassed the proxy's breaker accounting: a P failing
    // every request stayed closed and kept claiming.
    let f = Fixture::new(1000, 10).await;
    for i in 0..3 {
        let client = f.generate();
        let offer = f.acquire(&format!("poll-{i}")).await;
        f.post("accept", Fixture::action(&offer)).await;
        f.outcome(&offer, 502, "{}").await;
        assert_eq!(client.await.unwrap().status(), 502);
    }
    let _queued = f.generate();
    assert_eq!(f.acquire("after").await["status"], "empty");
}

#[tokio::test]
async fn pull_half_open_prefill_replays_its_probe_until_the_outcome() {
    let decode = MockWorker::start(vec![]).await;
    let cooldown = Duration::from_millis(5);
    let f = Fixture::with_decode(
        decode,
        false,
        1000,
        10,
        None,
        None,
        CircuitBreakerConfig {
            cool_down: cooldown,
            ..Default::default()
        },
    )
    .await;
    let prefill = f.registry.get(&WorkerId("p".into())).unwrap();
    let breaker = prefill.breaker.config().unwrap();
    for _ in 0..breaker.threshold.get() {
        prefill.breaker.record_failure();
    }
    tokio::time::sleep(cooldown).await;

    let mut first = f.generate();
    let mut second = f.generate();
    // Both requests must be eligible for P before its probe is in flight.
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let response = f
                .app
                .clone()
                .oneshot(Request::get("/metrics").body(Body::empty()).unwrap())
                .await
                .unwrap();
            let metrics = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            if std::str::from_utf8(&metrics)
                .unwrap()
                .contains("sgl_router_pull_requests{state=\"waiting\"} 2\n")
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("both clients should be queued");

    // A half-open P may publish one offer and replay it after a lost response.
    let probe = f.acquire_with_limit("probe", 2).await;
    assert_eq!(probe["status"], "offer");
    let replay = f.acquire_with_limit("probe", 2).await;
    assert_eq!(replay["assignment_id"], probe["assignment_id"]);
    assert_eq!(f.acquire_with_limit("other", 2).await["status"], "empty");
    assert_eq!(
        f.post("accept", Fixture::action(&probe)).await["status"],
        "committed"
    );

    // The permit remains held even after the probe's client reads D's response.
    let (response, queued) = tokio::select! {
        response = &mut first => (response.unwrap(), second),
        response = &mut second => (response.unwrap(), first),
    };
    assert_eq!(response.status(), 200);
    axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert!(!prefill.breaker.would_allow());
    assert_eq!(f.acquire_with_limit("other", 2).await["status"], "empty");
    assert_eq!(f.outcome(&probe, 200, "{}").await["status"], "completed");
    assert_eq!(prefill.breaker.snapshot().state_code, 0);

    let offer = f.acquire("after-recovery").await;
    assert_eq!(offer["status"], "offer");
    f.post("accept", Fixture::action(&offer)).await;
    f.outcome(&offer, 200, "{}").await;
    assert_eq!(queued.await.unwrap().status(), 200);
}

#[tokio::test]
async fn pull_late_prefill_outcomes_count_once_after_decode_response() {
    let decode = MockWorker::start(vec![]).await;
    let f = Fixture::with_decode(
        decode,
        false,
        1000,
        10,
        None,
        None,
        CircuitBreakerConfig::default(),
    )
    .await;
    let prefill = f.registry.get(&WorkerId("p".into())).unwrap();
    let threshold = prefill.breaker.config().unwrap().threshold.get();
    for count in 1..=threshold {
        let client = f.generate();
        let offer = f.acquire(&format!("poll-{count}")).await;
        assert_eq!(offer["status"], "offer");
        f.post("accept", Fixture::action(&offer)).await;
        let response = client.await.unwrap();
        assert_eq!(response.status(), 200);
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();

        // An accepted report still updates health after the client is gone.
        // Retrying it must not count the same upstream failure twice.
        for _ in 0..2 {
            assert_eq!(f.outcome(&offer, 502, "{}").await["status"], "completed");
            assert_eq!(
                prefill.breaker.snapshot().state_code,
                u8::from(count == threshold)
            );
        }
    }
}

#[tokio::test]
async fn pull_late_prefill_outcomes_preserve_worker_metrics() {
    for (status, body, outcome) in [
        (200, "{}", "success"),
        (502, "{}", "error"),
        (429, "{}", "backpressure"),
        (503, "{}", "backpressure"),
        (422, "{}", "client_error"),
        (
            200,
            "data: {\"error\":{\"message\":\"failed\"}}\n\n",
            "error",
        ),
    ] {
        let f = Fixture::with_decode(
            MockWorker::start(vec![]).await,
            false,
            1000,
            10,
            None,
            None,
            CircuitBreakerConfig::default(),
        )
        .await;
        let client = f.generate();
        let offer = f.acquire("late-outcome").await;
        f.post("accept", Fixture::action(&offer)).await;
        let response = client.await.unwrap();
        assert_eq!(response.status(), 200);
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        for _ in 0..2 {
            assert_eq!(f.outcome(&offer, status, body).await["status"], "completed");
        }

        // The detached task records one result even after the client body ends.
        // A lost outcome body used to turn every late response into an error.
        let prefix = format!(
            "sgl_router_worker_requests_total{{worker_url=\"{}\",model_id=\"tiny\",mode=\"prefill\",",
            f.prefill.url
        );
        let expected = format!("{prefix}outcome=\"{outcome}\"}} 1");
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let response = f
                    .app
                    .clone()
                    .oneshot(Request::get("/metrics").body(Body::empty()).unwrap())
                    .await
                    .unwrap();
                let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap();
                let metrics = std::str::from_utf8(&body).unwrap();
                let recorded: Vec<_> = metrics
                    .lines()
                    .filter(|line| line.starts_with(&prefix))
                    .collect();
                if !recorded.is_empty() {
                    assert_eq!(recorded, vec![expected.as_str()], "P status {status}");
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("the detached prefill task should record its outcome");
    }
}

#[tokio::test]
async fn pull_acquire_from_a_not_yet_registered_worker_is_retryable() {
    // A 400 sent a newly scaled-out prefill into the agent's long 4xx
    // backoff before discovery had registered it.
    let f = Fixture::new(1000, 10).await;
    let response = f
        .app
        .clone()
        .oneshot(request(
            "/internal/pull/acquire",
            json!({"worker_url":"http://unknown:1", "worker_epoch":"epoch-1", "dp_rank":0, "poll_id":"p", "max_tokens":100, "max_requests":1, "wait_ms":0}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), 503);
}

#[tokio::test]
async fn pull_dispatches_only_decode_after_commit_and_preserves_pairing() {
    let f = Fixture::new(1000, 10).await;
    let client = f.generate();
    let offer = f.acquire("poll-1").await;
    assert_eq!(offer["status"], "offer");
    let duplicate = f.acquire("poll-1").await;
    assert_eq!(offer["assignment_id"], duplicate["assignment_id"]);
    assert_eq!(offer["body"], duplicate["body"]);
    assert!(f.prefill.captured.lock().unwrap().last_body.is_none());
    assert!(f.decode.captured.lock().unwrap().last_body.is_none());
    assert_eq!(
        f.post("accept", Fixture::action(&offer)).await["status"],
        "committed"
    );
    assert_eq!(
        f.post("accept", Fixture::action(&offer)).await["status"],
        "committed"
    );
    let decode = f.decode.captured_json().await;
    assert_eq!(decode["rid"], offer["request_id"]);
    assert_eq!(decode["bootstrap_room"], offer["body"]["bootstrap_room"]);
    assert_eq!(offer["body"]["stream"], false);
    assert_eq!(offer["body"]["routed_dp_rank"], 0);
    assert_eq!(decode["routed_dp_rank"], 0);
    assert_eq!(decode["disagg_prefill_dp_rank"], 0);
    assert_eq!(f.outcome(&offer, 200, "{}").await["status"], "completed");
    let result = client.await.unwrap();
    assert_eq!(result.status(), 200);
    axum::body::to_bytes(result.into_body(), 1024 * 1024)
        .await
        .unwrap();
    assert_eq!(f.outcome(&offer, 200, "{}").await["status"], "completed");
    assert!(
        f.prefill.captured.lock().unwrap().last_body.is_none(),
        "router must never POST work to P"
    );
}

#[tokio::test]
async fn pull_prefill_client_error_reaches_client_verbatim() {
    let f = Fixture::new(1000, 10).await;
    let client = f.generate();
    let offer = f.acquire("p").await;
    f.post("accept", Fixture::action(&offer)).await;
    f.outcome(&offer, 422, "{\"error\":\"invalid prompt\"}")
        .await;
    let response = client.await.unwrap();
    assert_eq!(response.status(), 422);
    assert_eq!(
        response.headers()["x-router-error-code"],
        "prefill_request_rejected"
    );
    assert_eq!(
        axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap(),
        "{\"error\":\"invalid prompt\"}"
    );
}

#[tokio::test]
async fn pull_outcome_wire_limit_accounts_for_json_escaping() {
    let f = Fixture::new(1000, 10).await;
    let client = f.generate();
    let offer = f.acquire("escaped").await;
    f.post("accept", Fixture::action(&offer)).await;
    // This valid 3 MiB JSON response becomes >6 MiB when nested in the control
    // message, but still fits the 4 MiB raw outcome contract.
    let body = serde_json::to_string(&"\\".repeat(1500 * 1024)).unwrap();
    assert_eq!(f.outcome(&offer, 200, &body).await["status"], "completed");
    assert_eq!(client.await.unwrap().status(), 200);
}

#[tokio::test]
async fn pull_expired_offer_requeues_with_new_request_and_room() {
    let f = Fixture::new(30, 10).await;
    let client = f.generate();
    let first = f.acquire("first").await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        f.post("accept", Fixture::action(&first)).await["status"],
        "expired"
    );
    let second = f.acquire("second").await;
    assert_eq!(second["status"], "offer");
    assert_ne!(first["request_id"], second["request_id"]);
    assert_ne!(
        first["body"]["bootstrap_room"],
        second["body"]["bootstrap_room"]
    );
    assert!(f.decode.captured.lock().unwrap().last_body.is_none());
    f.post("accept", Fixture::action(&second)).await;
    f.outcome(&second, 200, "{}").await;
    assert_eq!(client.await.unwrap().status(), 200);
}

#[tokio::test]
async fn pull_epoch_rotation_reassigns_offer_without_failing_the_client() {
    let f = Fixture::new(10_000, 10).await;
    let client = f.generate();
    let first = f.acquire("old-agent").await;
    assert_eq!(first["status"], "offer");
    let second = f
        .post(
            "acquire",
            json!({
                "worker_url": f.prefill.url, "worker_epoch": "epoch-2",
                "dp_rank": 0, "poll_id": "new-agent", "max_tokens": 100,
                "max_requests": 1, "wait_ms": 500
            }),
        )
        .await;
    assert_eq!(second["status"], "offer");
    assert_eq!(
        f.post("accept", Fixture::action(&first)).await["status"],
        "expired"
    );
    assert_ne!(first["assignment_id"], second["assignment_id"]);
    assert_ne!(first["request_id"], second["request_id"]);
    assert_ne!(
        first["body"]["bootstrap_room"],
        second["body"]["bootstrap_room"]
    );
    assert!(!client.is_finished());
    assert!(f.decode.captured.lock().unwrap().last_body.is_none());
    let action = json!({
        "assignment_id": second["assignment_id"], "worker_epoch": "epoch-2"
    });
    assert_eq!(
        f.post("accept", action.clone()).await["status"],
        "committed"
    );
    let decode = f.decode.captured_json().await;
    assert_eq!(decode["rid"], second["request_id"]);
    let mut outcome = action;
    outcome["status_code"] = json!(200);
    outcome["body"] = json!("{}");
    f.post("outcome", outcome).await;
    let response = client.await.unwrap();
    assert_eq!(response.status(), 200);
    axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
}

#[tokio::test]
async fn pull_cancellation_revokes_offer_and_reclaims_queue_budget() {
    let f = Fixture::new(1000, 1).await;
    let client = f.generate();
    let offer = f.acquire("first").await;
    let saturated = f.generate().await.unwrap();
    assert_eq!(saturated.status(), 503);
    assert_eq!(
        saturated.headers()["x-router-error-code"],
        "pull_queue_full"
    );
    let error: Value = serde_json::from_slice(
        &axum::body::to_bytes(saturated.into_body(), 1024)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(error["error"]["message"], "pull queue capacity exhausted");
    client.abort();
    let _ = client.await;
    assert_eq!(
        f.post("accept", Fixture::action(&offer)).await["status"],
        "cancelled"
    );
    let next = f.generate();
    let offer = f.acquire("second").await;
    assert_eq!(offer["status"], "offer");
    next.abort();
}

#[tokio::test]
async fn pull_does_not_replay_committed_work_when_client_is_cancelled() {
    let f = Fixture::new(30, 10).await;
    let client = f.generate();
    let offer = f.acquire("first").await;
    f.post("accept", Fixture::action(&offer)).await;
    client.abort();
    let _ = client.await;
    assert_eq!(
        f.post("status", Fixture::action(&offer)).await["status"],
        "cancelled"
    );
    assert_eq!(f.acquire("second").await["status"], "empty");
}

#[tokio::test]
async fn pull_rejects_batch_and_multisample_before_queueing() {
    let f = Fixture::new(1000, 10).await;
    for body in [
        json!({"input_ids":[[1],[2]]}),
        json!({"input_ids":[1],"sampling_params":{"n":2}}),
    ] {
        let response = f
            .app
            .clone()
            .oneshot(request("/generate", body))
            .await
            .unwrap();
        assert_eq!(response.status(), 400);
    }
    assert_eq!(f.acquire("empty").await["status"], "empty");
}

#[tokio::test]
async fn pull_requires_exact_text_tokens_before_queueing() {
    let f = Fixture::new(1000, 10).await;
    let response = tokio::time::timeout(
        Duration::from_millis(100),
        f.app
            .clone()
            .oneshot(request("/generate", json!({"text":"中文"}))),
    )
    .await
    .expect("un-tokenizable text must fail before waiting for a worker")
    .unwrap();
    assert_eq!(response.status(), 400);
    let body = axum::body::to_bytes(response.into_body(), 1024)
        .await
        .unwrap();
    assert!(String::from_utf8_lossy(&body).contains("input_ids"));
    assert_eq!(f.acquire("empty").await["status"], "empty");
}

#[tokio::test]
async fn pull_forwards_router_ids_even_when_text_forwarding_is_disabled() {
    // P rejects an engine token count above the router's; forwarding text let a
    // P-side tokenization difference (e.g. one more special token) fail every request.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.json");
    let mut tokenizer: Value =
        serde_json::from_slice(&std::fs::read("tests/fixtures/tiny_tokenizer.json").unwrap())
            .unwrap();
    tokenizer["added_tokens"]
        .as_array_mut()
        .unwrap()
        .push(json!({
            "id":257, "special":false, "content":"中文", "single_word":false,
            "lstrip":false, "rstrip":false, "normalized":true
        }));
    std::fs::write(&path, tokenizer.to_string()).unwrap();
    let f = Fixture::configured(1000, 10, None, Some(path.to_str().unwrap())).await;
    let app = f.app.clone();
    let client = tokio::spawn(async move {
        app.oneshot(request("/generate", json!({"text":"中文中文"})))
            .await
            .unwrap()
    });
    let offer = f
        .post(
            "acquire",
            json!({
                "worker_url":f.prefill.url, "worker_epoch":"epoch-1", "dp_rank":0,
                "poll_id":"exact-text", "max_tokens":2, "max_requests":1, "wait_ms":200
            }),
        )
        .await;
    assert_eq!(offer["status"], "offer");
    assert_eq!(offer["input_tokens"], 2);
    assert_eq!(offer["body"]["input_ids"], json!([257, 257]));
    assert!(offer["body"].get("text").is_none());
    f.post("accept", Fixture::action(&offer)).await;
    f.outcome(&offer, 200, "{}").await;
    assert_eq!(client.await.unwrap().status(), 200);
}

#[tokio::test]
async fn pull_client_deadline_includes_offer_waiting_time() {
    let f = Fixture::new(10_000, 10).await;
    let client = f.generate();
    let offer = f.acquire("first").await;
    let response = tokio::time::timeout(Duration::from_secs(4), client)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(response.status(), 504);
    assert_eq!(
        f.post("accept", Fixture::action(&offer)).await["status"],
        "cancelled"
    );
    assert!(f.decode.captured.lock().unwrap().last_body.is_none());
}

#[tokio::test]
async fn pull_wire_contract_works_over_http() {
    let f = Fixture::new(1000, 10).await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let app = f.app.clone();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let http = reqwest::Client::new();
    let client = {
        let http = http.clone();
        let base = base.clone();
        tokio::spawn(async move {
            http.post(format!("{base}/generate"))
                .json(&json!({"input_ids":[1,2,3]}))
                .send()
                .await
                .unwrap()
        })
    };
    let offer: Value = http.post(format!("{base}/internal/pull/acquire"))
        .bearer_auth("pull-test-secret")
        .json(&json!({"worker_url":f.prefill.url,"worker_epoch":"epoch-1","dp_rank":0,"poll_id":"wire","max_tokens":100,"max_requests":1,"wait_ms":500}))
        .send().await.unwrap().json().await.unwrap();
    assert_eq!(offer["status"], "offer");
    let commit: Value = http
        .post(format!("{base}/internal/pull/accept"))
        .bearer_auth("pull-test-secret")
        .json(&Fixture::action(&offer))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(commit["status"], "committed");
    // The worker executes the acquired payload locally and reports its response.
    let p_response = http
        .post(format!("{}/generate", f.prefill.url))
        .json(&offer["body"])
        .send()
        .await
        .unwrap();
    let mut outcome = Fixture::action(&offer);
    outcome["status_code"] = json!(p_response.status().as_u16());
    outcome["body"] = json!(p_response.text().await.unwrap());
    assert!(http
        .post(format!("{base}/internal/pull/outcome"))
        .bearer_auth("pull-test-secret")
        .json(&outcome)
        .send()
        .await
        .unwrap()
        .status()
        .is_success());
    assert_eq!(client.await.unwrap().status(), 200);
    server.abort();
}

#[test]
fn pull_configuration_rejects_unsupported_combinations() {
    for (extra, expected) in [
        (
            vec!["--chat-routing", "legacy"],
            "requires --chat-routing reorg",
        ),
        (
            vec!["--chat-routing", "reorg", "--grpc-port", "5000"],
            "does not support gRPC",
        ),
        (
            vec!["--chat-routing", "reorg", "--decode-dispatch-limit", "1"],
            "decode-dispatch-limit",
        ),
        (
            vec!["--chat-routing", "reorg", "--pull-max-bytes", "0"],
            "limits must be positive",
        ),
    ] {
        let mut args = vec![
            "sgl-router",
            "--model-id",
            "tiny",
            "--worker-urls",
            "http://p:3000",
            "--no-tokenizer",
            "--prefill-dispatch-mode",
            "pull",
            "--pull-control-api-key",
            "pull-test-secret",
        ];
        args.extend(extra);
        let error = Cli::try_parse_from(args)
            .unwrap()
            .into_config()
            .unwrap_err()
            .to_string();
        assert!(error.contains(expected), "{error}");
    }
}

#[tokio::test]
async fn pull_replacement_prefill_claims_requests_queued_before_it_joined() {
    // The eligible set was fixed at enqueue, so after a rolling replacement the
    // new prefill's acquire returned empty and the queued request waited out
    // its deadline.
    let f = Fixture::new(1000, 10).await;
    let client = f.generate();
    tokio::time::sleep(Duration::from_millis(100)).await;
    let replacement = MockWorker::start(vec![]).await;
    f.registry
        .add(WorkerSpec {
            id: WorkerId("p2".into()),
            url: replacement.url.clone(),
            mode: WorkerMode::Prefill,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: Some(8997),
            ..Default::default()
        })
        .unwrap();
    f.registry.remove(&WorkerId("p".into()));
    let offer = f
        .post(
            "acquire",
            json!({
                "worker_url": replacement.url, "worker_epoch": "epoch-1", "dp_rank": 0,
                "poll_id": "replacement", "max_tokens": 100, "max_requests": 1,
                "wait_ms": 3000
            }),
        )
        .await;
    assert_eq!(offer["status"], "offer");
    f.post("accept", Fixture::action(&offer)).await;
    f.outcome(&offer, 200, "{}").await;
    assert_eq!(client.await.unwrap().status(), 200);
}

fn prefill_spec(id: &str, url: &str, mode: WorkerMode) -> WorkerSpec {
    WorkerSpec {
        id: WorkerId(id.into()),
        url: url.to_owned(),
        mode,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: Some(8997),
        ..Default::default()
    }
}

#[tokio::test]
async fn pull_candidates_refresh_when_decode_returns() {
    // Replacing P while D was down refreshed to an empty set; D coming back
    // left the prefill fleet unchanged, so the new P never became eligible.
    let f = Fixture::new(1000, 10).await;
    let client = f.generate();
    tokio::time::sleep(Duration::from_millis(100)).await;
    let replacement = MockWorker::start(vec![]).await;
    f.registry.remove(&WorkerId("d".into()));
    f.registry
        .add(prefill_spec("p2", &replacement.url, WorkerMode::Prefill))
        .unwrap();
    f.registry.remove(&WorkerId("p".into()));
    tokio::time::sleep(Duration::from_millis(1500)).await;
    f.registry
        .add(prefill_spec("d", &f.decode.url, WorkerMode::Decode))
        .unwrap();
    let offer = f
        .post(
            "acquire",
            json!({
                "worker_url": replacement.url, "worker_epoch": "epoch-1", "dp_rank": 0,
                "poll_id": "after-decode", "max_tokens": 100, "max_requests": 1,
                "wait_ms": 3000
            }),
        )
        .await;
    assert_eq!(offer["status"], "offer");
    f.post("accept", Fixture::action(&offer)).await;
    f.outcome(&offer, 200, "{}").await;
    assert_eq!(client.await.unwrap().status(), 200);
}

#[tokio::test]
async fn pull_candidate_refresh_is_not_postponed_by_repeated_rejections() {
    // Each reject restarted the one-second refresh timer, so an old P that
    // claimed and rejected every 50 ms kept a new P from ever becoming eligible.
    let f = Arc::new(Fixture::new(1000, 10).await);
    let client = f.generate();
    tokio::time::sleep(Duration::from_millis(100)).await;
    let replacement = MockWorker::start(vec![]).await;
    f.registry
        .add(prefill_spec("p2", &replacement.url, WorkerMode::Prefill))
        .unwrap();
    let churn = {
        let f = f.clone();
        tokio::spawn(async move {
            for i in 0.. {
                let offer = f
                    .post(
                        "acquire",
                        json!({
                            "worker_url": f.prefill.url, "worker_epoch": "epoch-1",
                            "dp_rank": 0, "poll_id": format!("churn-{i}"),
                            "max_tokens": 100, "max_requests": 1, "wait_ms": 50
                        }),
                    )
                    .await;
                if offer["status"] == "offer" {
                    f.post("reject", Fixture::action(&offer)).await;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
    };
    let offer = f
        .post(
            "acquire",
            json!({
                "worker_url": replacement.url, "worker_epoch": "epoch-1", "dp_rank": 0,
                "poll_id": "new-p", "max_tokens": 100, "max_requests": 1,
                "wait_ms": 3000
            }),
        )
        .await;
    churn.abort();
    assert_eq!(offer["status"], "offer");
    f.post("accept", Fixture::action(&offer)).await;
    f.outcome(&offer, 200, "{}").await;
    assert_eq!(client.await.unwrap().status(), 200);
}

#[tokio::test]
#[ignore = "slow: tokenizes 8 MiB of text in a debug build (~25 s)"]
async fn pull_refuses_at_ingress_a_prompt_too_large_once_it_is_token_ids() {
    // The size check ran on the text body; JSON ids are several bytes per
    // token, so the offer outgrew 15 MiB and failed only after a P claimed it.
    let f = Fixture::configured(1000, 10, None, Some("tests/fixtures/tiny_tokenizer.json")).await;
    let text = "a ".repeat(4 * 1024 * 1024);
    let response = f
        .app
        .clone()
        .oneshot(request("/generate", json!({"text": text})))
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let empty = f
        .post(
            "acquire",
            json!({
                "worker_url": f.prefill.url, "worker_epoch": "epoch-1", "dp_rank": 0,
                "poll_id": "never", "max_tokens": 100_000_000, "max_requests": 1,
                "wait_ms": 100
            }),
        )
        .await;
    assert_eq!(empty["status"], "empty");
}

#[tokio::test]
async fn pull_candidates_refresh_when_a_prefill_gets_its_bootstrap_port() {
    // A prefill registered before its bootstrap port is known is skipped by
    // bucket membership. Filling the port changed nothing the fleet
    // fingerprint saw, and the old prefill kept the set non-empty, so the
    // newly routable prefill never received the queued request.
    let f = Fixture::new(1000, 10).await;
    let late = MockWorker::start(vec![]).await;
    let mut portless = prefill_spec("p2", &late.url, WorkerMode::Prefill);
    portless.bootstrap_port = None;
    f.registry.add(portless).unwrap();
    let client = f.generate();
    tokio::time::sleep(Duration::from_millis(100)).await;
    f.registry
        .add(prefill_spec("p2", &late.url, WorkerMode::Prefill))
        .unwrap();
    let offer = f
        .post(
            "acquire",
            json!({
                "worker_url": late.url, "worker_epoch": "epoch-1", "dp_rank": 0,
                "poll_id": "late", "max_tokens": 100, "max_requests": 1,
                "wait_ms": 3000
            }),
        )
        .await;
    assert_eq!(offer["status"], "offer");
    f.post("accept", Fixture::action(&offer)).await;
    f.outcome(&offer, 200, "{}").await;
    assert_eq!(client.await.unwrap().status(), 200);
}
