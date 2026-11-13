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
        let prefill = MockWorker::start(vec![]).await;
        let decode = MockWorker::start(vec![]).await;
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
            "--worker-api-key",
            "pull-test-secret",
        ])
        .unwrap()
        .into_config()
        .unwrap();
        cfg.proxy.pull.offer_timeout_ms = lease_ms;
        cfg.proxy.pull.max_requests = max_requests;
        cfg.proxy.stream_after_prefill = true;
        cfg.proxy.request_timeout_secs = 3;
        if let Some(path) = tokenizer {
            cfg.model.tokenizer_path = Some(path.to_owned());
            // Cover exact routing counts when the execution body remains text.
            cfg.model.disable_input_ids_forwarding = true;
        }
        let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
        let registry = Arc::new(WorkerRegistry::default());
        for (id, url, mode) in [
            ("p", &prefill.url, WorkerMode::Prefill),
            ("d", &decode.url, WorkerMode::Decode),
        ] {
            registry
                .add(WorkerSpec {
                    id: WorkerId(id.into()),
                    url: url.clone(),
                    mode,
                    model_ids: vec![ModelId("tiny".into())],
                    bootstrap_port: Some(8997),
                    ..Default::default()
                })
                .unwrap();
        }
        let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
        let mut ctx = AppContext::new(
            cfg,
            tokenizers,
            Arc::new(Proxy::new(Duration::from_secs(3)).unwrap()),
            registry,
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
        }
    }
    fn generate(&self) -> tokio::task::JoinHandle<axum::response::Response> {
        let app = self.app.clone();
        tokio::spawn(async move {
            app.oneshot(request("/generate", json!({"input_ids":[1,2,3], "sampling_params":{"max_new_tokens":1}, "stream":false}))).await.unwrap()
        })
    }
    async fn acquire(&self, poll: &str) -> Value {
        self.post("acquire", json!({"worker_url":self.prefill.url, "worker_epoch":"epoch-1", "dp_rank":0, "poll_id":poll, "max_tokens":100, "max_requests":1, "wait_ms":500})).await
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
async fn pull_text_budget_uses_tokenizer_ids_even_when_forwarding_text() {
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
    assert_eq!(offer["body"]["text"], "中文中文");
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
            "--worker-api-key",
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
