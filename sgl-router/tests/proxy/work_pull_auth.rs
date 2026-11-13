// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use axum::{body::Body, http::Request, Router};
use clap::Parser;
use serde_json::{json, Value};
use sgl_router::{
    config::{Cli, PrefillDispatchMode},
    discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec},
    policies::factory::build_registry_with_defaults,
    proxy::Proxy,
    server::{app::build_router, app_context::AppContext},
    tokenizer::TokenizerRegistry,
    workers::WorkerRegistry,
};
use std::{sync::Arc, time::Duration};
use tower::ServiceExt;

const WORKER: &str = "http://127.0.0.1:39999";
const KEY: &str = "Bearer pull-test-secret";

fn cli(mode: &str, key: Option<&str>) -> Cli {
    let mut args = vec![
        "sgl-router",
        "--model-id",
        "tiny",
        "--worker-urls",
        WORKER,
        "--no-tokenizer",
        "--chat-routing",
        "reorg",
        "--prefill-dispatch-mode",
        mode,
    ];
    if let Some(key) = key {
        args.extend(["--worker-api-key", key]);
    }
    Cli::try_parse_from(args).unwrap()
}

fn app(key: Option<&str>) -> Router {
    let mut cfg = cli("push", key).into_config().unwrap();
    // Exercise a programmatically constructed config too: runtime auth must
    // remain closed even if a caller bypasses CLI/startup validation.
    cfg.proxy.pull.mode = PrefillDispatchMode::Pull;
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    registry
        .add(WorkerSpec {
            id: WorkerId("p".into()),
            url: WORKER.into(),
            mode: WorkerMode::Prefill,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: Some(8998),
            ..Default::default()
        })
        .unwrap();
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    build_router(Arc::new(AppContext::new(
        cfg,
        tokenizers,
        Arc::new(Proxy::new(Duration::from_secs(1)).unwrap()),
        registry,
        policies,
    )))
}

fn body(route: &str) -> Value {
    match route {
        "acquire" => json!({"worker_url":WORKER,"worker_epoch":"e","dp_rank":0,
            "poll_id":"p","max_tokens":1,"max_requests":1,"wait_ms":0}),
        "outcome" => json!({"assignment_id":"unknown","worker_epoch":"e",
            "status_code":200,"body":"{}"}),
        _ => json!({"assignment_id":"unknown","worker_epoch":"e"}),
    }
}

async fn post(
    app: &Router,
    route: &str,
    headers: &[&str],
    body: String,
) -> axum::response::Response {
    let mut request = Request::builder()
        .method("POST")
        .uri(format!("/internal/pull/{route}"))
        .header("content-type", "application/json");
    for value in headers {
        request = request.header("authorization", *value);
    }
    app.clone()
        .oneshot(request.body(Body::from(body)).unwrap())
        .await
        .unwrap()
}

#[tokio::test]
async fn every_pull_control_requires_auth_before_reading_the_payload() {
    let app = app(Some("pull-test-secret"));
    for route in ["acquire", "accept", "reject", "status", "outcome"] {
        for headers in [
            vec![],
            vec!["Bearer wrong"],
            vec!["Basic pull-test-secret"],
            vec![KEY, KEY],
        ] {
            for payload in [body(route).to_string(), "not JSON".into()] {
                let response = post(&app, route, &headers, payload).await;
                assert_eq!(response.status(), 401, "{route}: {headers:?}");
                assert_eq!(
                    response.headers()["x-router-error-code"],
                    "pull_authentication_required"
                );
                assert_eq!(response.headers()["www-authenticate"], "Bearer");
            }
        }
        let response = post(&app, route, &[KEY], body(route).to_string()).await;
        assert_eq!(response.status(), 200, "authenticated {route}");
    }
}

#[tokio::test]
async fn missing_runtime_key_never_opens_pull_control() {
    let app = app(None);
    for route in ["acquire", "accept", "reject", "status", "outcome"] {
        for headers in [vec![], vec![KEY]] {
            let response = post(&app, route, &headers, body(route).to_string()).await;
            assert_eq!(response.status(), 401, "{route}");
        }
    }
}

#[test]
fn pull_requires_a_key_but_default_push_does_not() {
    for key in [None, Some("")] {
        let error = cli("pull", key).into_config().unwrap_err().to_string();
        assert!(error.contains("worker-api-key"), "{error}");
    }
    assert!(cli("pull", Some("pull-test-secret")).into_config().is_ok());
    assert!(cli("push", None).into_config().is_ok());
}
