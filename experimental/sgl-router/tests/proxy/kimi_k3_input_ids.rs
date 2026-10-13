// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Exercise the K3 process boundary and P/D fan-out without model downloads.
//! The Python helper's normalization is tested separately against serving.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use sgl_router::config::{
    Config, DiscoveryBackend, InflightLoadConfig, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::policies::prefix_provider::RadixTreePrefixProvider;
use sgl_router::policies::{Policy, SelectionContext};
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::state::kv_events::{compute_block_hashes, HashTree, KvWorkerId};
use sgl_router::tokenizer::options::{ChatEncoderKind, ChatEncodingOptions};
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::{Worker, WorkerRegistry};
use std::fs;
use std::num::NonZeroUsize;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tower::ServiceExt;

use crate::common::mock_worker::MockWorker;

const MODEL: &str = "kimi-k3-test";
const GENERATED_IDS: &[u32] = &[100279, 781, 321, 100282];

struct FakeTokenizer {
    directory: TempDir,
    executable: PathBuf,
    requests: PathBuf,
    process_ids: PathBuf,
}

impl FakeTokenizer {
    fn new(reply: &str) -> Self {
        Self::with_replies(&[reply])
    }

    fn with_replies(replies: &[&str]) -> Self {
        Self::with_replies_and_restart_delay(replies, 0)
    }

    fn with_replies_and_restart_delay(replies: &[&str], restart_delay_millis: u64) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let executable = directory.path().join("fake-tokenizer");
        let requests = directory.path().join("requests.jsonl");
        let process_ids = directory.path().join("processes.txt");
        let process_count = directory.path().join("process-count");
        let counter = directory.path().join("reply-count");
        fs::write(&requests, "").unwrap();
        // JSON strings are valid Python string literals for these paths and
        // response bytes; no shell interpolation or environment mutation.
        let script = format!(
            r#"#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path

process_count = Path({process_count})
started_count = int(process_count.read_text()) if process_count.exists() else 0
process_count.write_text(str(started_count + 1))
with open({process_ids}, "a", encoding="utf-8") as started:
    started.write(str(os.getpid()) + "\n")
json.loads(sys.stdin.readline())
if started_count:
    time.sleep({restart_delay_seconds})
print(json.dumps({{"ready": True}}), flush=True)
replies = {replies}
counter = Path({counter})
for line in sys.stdin:
    request = json.loads(line)
    with open({requests}, "a", encoding="utf-8") as captured:
        captured.write(json.dumps(request) + "\n")
    count = int(counter.read_text()) if counter.exists() else 0
    counter.write_text(str(count + 1))
    reply = replies[min(count, len(replies) - 1)]
    if reply == "__stall__":
        time.sleep(30)
        reply = '{{"input_ids": [999]}}'
    print(reply, flush=True)
"#,
            requests = serde_json::to_string(&requests.to_str().unwrap()).unwrap(),
            process_ids = serde_json::to_string(&process_ids.to_str().unwrap()).unwrap(),
            process_count = serde_json::to_string(&process_count.to_str().unwrap()).unwrap(),
            counter = serde_json::to_string(&counter.to_str().unwrap()).unwrap(),
            replies = serde_json::to_string(replies).unwrap(),
            restart_delay_seconds = (restart_delay_millis as f64) / 1000.0,
        );
        fs::write(&executable, script).unwrap();
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o700)).unwrap();
        Self {
            directory,
            executable,
            requests,
            process_ids,
        }
    }

    fn successful() -> Self {
        Self::new(&json!({"input_ids": GENERATED_IDS}).to_string())
    }

    fn captured_requests(&self) -> Vec<Value> {
        fs::read_to_string(&self.requests)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    fn started_processes(&self) -> Vec<u32> {
        // The file is created by the first process; none means none started.
        fs::read_to_string(&self.process_ids)
            .unwrap_or_default()
            .lines()
            .map(|line| line.parse().unwrap())
            .collect()
    }

    fn config(&self) -> Config {
        Config {
            server: ServerConfig {
                host: "0".into(),
                port: 0,
                ..Default::default()
            },
            observability: ObservabilityConfig::default(),
            model: ModelConfig {
                id: MODEL.into(),
                // Deliberately no tokenizer.json: K3 uses its Python tokenizer.
                tokenizer_path: self.directory.path().to_str().unwrap().into(),
                disable_input_ids_forwarding: false,
                policy: PolicyKind::RoundRobin,
                decode_policy: Default::default(),
                bucket_config: None,
                circuit_breaker: None,
                cache_aware: None,
                sticky: None,
                affinity: None,
                fused: None,
                eligibility: None,
                sampling_overrides: Default::default(),
                default_chat_template_kwargs: Default::default(),
            },
            discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
                urls: vec!["http://placeholder:0".into()],
            }),
            proxy: ProxyConfig::default(),
            router_inflight_load: InflightLoadConfig::default(),
        }
    }

    fn context(&self, prefill: &MockWorker, decode: &MockWorker) -> Arc<AppContext> {
        self.context_with_timeout(prefill, decode, 5)
    }

    fn context_with_timeout(
        &self,
        prefill: &MockWorker,
        decode: &MockWorker,
        tokenizer_timeout_secs: u64,
    ) -> Arc<AppContext> {
        self.context_with_timeouts(prefill, decode, tokenizer_timeout_secs, 300)
    }

    fn context_with_timeouts(
        &self,
        prefill: &MockWorker,
        decode: &MockWorker,
        tokenizer_timeout_secs: u64,
        tokenizer_startup_timeout_secs: u64,
    ) -> Arc<AppContext> {
        let cfg = self.config();
        let options = ChatEncodingOptions {
            chat_encoder: ChatEncoderKind::KimiK3,
            tokenizer_python: self.executable.to_str().unwrap().into(),
            tokenizer_workers: NonZeroUsize::new(1),
            tokenizer_timeout_secs,
            tokenizer_startup_timeout_secs,
            ..Default::default()
        };
        let tokenizers = Arc::new(TokenizerRegistry::load_with_options(&cfg, options).unwrap());
        assert!(tokenizers.has_kimi_encoder(MODEL));
        assert!(tokenizers.get(MODEL).is_none());
        let registry = Arc::new(WorkerRegistry::default());
        for (worker, mode, port) in [
            (prefill, WorkerMode::Prefill, Some(8997)),
            (decode, WorkerMode::Decode, None),
        ] {
            registry
                .add(WorkerSpec {
                    id: WorkerId(worker.url.clone()),
                    url: worker.url.clone(),
                    mode,
                    model_ids: vec![ModelId(MODEL.into())],
                    bootstrap_port: port,
                })
                .unwrap();
        }
        let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
        let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
        Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
    }
}

async fn send(ctx: Arc<AppContext>, body: &Value) -> StatusCode {
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(body).unwrap()))
        .unwrap();
    build_router(ctx).oneshot(request).await.unwrap().status()
}

async fn captured_body(worker: &MockWorker) -> Value {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let body = worker.captured.lock().unwrap().last_body.clone();
            if let Some(body) = body {
                return serde_json::from_slice(&body).unwrap();
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("P/D worker must receive the request")
}

fn text_request() -> Value {
    json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "你好，please help me"}],
    })
}

#[test]
fn kimi_k3_pool_is_opt_in_and_loads_without_rust_tokenizer() {
    for model in [MODEL, "custom-served-model"] {
        let tokenizer = FakeTokenizer::successful();
        let mut cfg = tokenizer.config();
        cfg.model.id = model.into();
        let options = ChatEncodingOptions {
            chat_encoder: ChatEncoderKind::KimiK3,
            tokenizer_python: tokenizer.executable.to_str().unwrap().into(),
            tokenizer_workers: NonZeroUsize::new(1),
            tokenizer_timeout_secs: 5,
            ..Default::default()
        };
        let registry = TokenizerRegistry::load_with_options(&cfg, options).unwrap();
        assert!(registry.has_kimi_encoder(model));
        assert!(registry.get(model).is_none());
        assert!(tokenizer.captured_requests().is_empty());
    }
    // Without --chat-encoder kimi-k3 a K3 model name keeps the native
    // dynamo-render path, so no Python process is started.
    let tokenizer = FakeTokenizer::successful();
    let mut cfg = tokenizer.config();
    cfg.model.tokenizer_path = "tests/fixtures/tiny_tokenizer.json".into();
    let registry =
        TokenizerRegistry::load_with_options(&cfg, ChatEncodingOptions::default()).unwrap();
    assert!(!registry.has_kimi_encoder(MODEL));
    assert!(tokenizer.started_processes().is_empty());
}

#[tokio::test]
async fn kimi_k3_raw_tokenize_endpoints_report_not_supported() {
    let tokenizer = FakeTokenizer::successful();
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let ctx = tokenizer.context(&prefill, &decode);
    for (uri, body) in [
        ("/v1/tokenize", json!({"model": MODEL, "prompt": "hi"})),
        ("/v1/detokenize", json!({"model": MODEL, "tokens": [1, 2]})),
        ("/v1/tokenize", json!({"model": "unknown", "prompt": "hi"})),
    ] {
        let request = Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let status = build_router(Arc::clone(&ctx))
            .oneshot(request)
            .await
            .unwrap()
            .status();
        let expected = if body["model"] == MODEL {
            StatusCode::NOT_IMPLEMENTED
        } else {
            StatusCode::NOT_FOUND
        };
        assert_eq!(status, expected, "{uri} {body}");
    }
    assert!(tokenizer.captured_requests().is_empty());
}

#[tokio::test]
async fn kimi_k3_tokenizes_complete_request_once_and_sends_same_ids_to_pd() {
    let tokenizer = FakeTokenizer::successful();
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let ctx = tokenizer.context(&prefill, &decode);
    let request = json!({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": [{"type": "text", "text": "北京天气"}]},
            {"role": "assistant", "content": "", "reasoning_content": "Check weather.",
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "weather", "arguments": "{\"city\":\"北京\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            {"role": "assistant", "content": "The weather is"}
        ],
        "tools": [{"type": "function", "function": {
            "name": "weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }}],
        "tool_choice": "auto",
        "reasoning_effort": "high",
        "reasoning": {"enabled": true},
        "chat_template_kwargs": {"custom_flag": "retained"},
        "response_format": {"type": "json_object"},
        "continue_final_message": true,
        "stream": true,
        "max_completion_tokens": 100,
    });
    assert_eq!(send(ctx, &request).await, StatusCode::OK);

    let p = captured_body(&prefill).await;
    let d = captured_body(&decode).await;
    assert_eq!(p["input_ids"], json!(GENERATED_IDS));
    assert_eq!(d["input_ids"], p["input_ids"]);
    assert_eq!(p["bootstrap_room"], d["bootstrap_room"]);
    assert_eq!(p["rid"], d["rid"]);
    for (key, expected) in request.as_object().unwrap() {
        if key == "stream" {
            assert_eq!(p[key], false);
            assert_eq!(d[key], expected.clone());
        } else {
            assert_eq!(&p[key], expected, "prefill changed {key}");
            assert_eq!(&d[key], expected, "decode changed {key}");
        }
    }
    assert_eq!(tokenizer.captured_requests(), vec![request]);
}

#[tokio::test]
async fn kimi_k3_caller_input_ids_bypass_tokenization_and_are_preserved() {
    let tokenizer = FakeTokenizer::successful();
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let ctx = tokenizer.context(&prefill, &decode);
    let mut request = text_request();
    request["input_ids"] = json!([7, 8, 9]);
    assert_eq!(send(ctx, &request).await, StatusCode::OK);
    for body in [captured_body(&prefill).await, captured_body(&decode).await] {
        assert_eq!(body["input_ids"], request["input_ids"]);
        assert_eq!(body["messages"], request["messages"]);
    }
    assert!(tokenizer.captured_requests().is_empty());
}

#[tokio::test]
async fn kimi_k3_malformed_caller_input_ids_are_rejected_at_router() {
    let tokenizer = FakeTokenizer::successful();
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let ctx = tokenizer.context(&prefill, &decode);
    let mut request = text_request();
    request["input_ids"] = json!("not an array of token IDs");

    assert_eq!(send(ctx, &request).await, StatusCode::BAD_REQUEST);
    assert!(tokenizer.captured_requests().is_empty());
    assert!(prefill.captured.lock().unwrap().last_body.is_none());
    assert!(decode.captured.lock().unwrap().last_body.is_none());
}

#[tokio::test]
async fn kimi_k3_images_encode_once_and_preserve_image_payloads_for_pd() {
    let tokenizer = FakeTokenizer::successful();
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let ctx = tokenizer.context(&prefill, &decode);
    let request = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What is in this picture?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png", "detail": "high"}},
            {"type": "text", "text": "Compare it with this image."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}, "modalities": "image"}
        ]}],
        "reasoning_effort": "high",
    });
    assert_eq!(send(ctx, &request).await, StatusCode::OK);
    for body in [captured_body(&prefill).await, captured_body(&decode).await] {
        assert_eq!(body["input_ids"], json!(GENERATED_IDS));
        assert_eq!(body["messages"], request["messages"]);
        assert_eq!(body["reasoning_effort"], request["reasoning_effort"]);
    }
    assert_eq!(tokenizer.captured_requests(), vec![request]);
}

#[tokio::test]
async fn kimi_k3_audio_and_video_requests_bypass_encoder() {
    for part in [
        json!({"type": "input_audio", "input_audio": {"data": "YXVkaW8=", "format": "wav"}}),
        json!({"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}),
        json!({"type": "image_url", "image_url": {"url": "https://example.com/frame.png"}, "modalities": "video"}),
    ] {
        let tokenizer = FakeTokenizer::successful();
        let prefill = MockWorker::start(vec![]).await;
        let decode = MockWorker::start(vec![]).await;
        let ctx = tokenizer.context(&prefill, &decode);
        let request = json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Describe this media."}, part
            ]}],
        });
        assert_eq!(send(ctx, &request).await, StatusCode::OK);
        for body in [captured_body(&prefill).await, captured_body(&decode).await] {
            assert!(body.get("input_ids").is_none());
            assert_eq!(body["messages"], request["messages"]);
        }
        assert!(tokenizer.captured_requests().is_empty());
    }
}

#[derive(Debug)]
struct ImageRoutingProbe {
    expected_estimate: u64,
    selections: Arc<AtomicUsize>,
}

impl Policy for ImageRoutingProbe {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        assert!(
            ctx.request_tokens().is_none(),
            "unexpanded image IDs entered prefix scoring"
        );
        assert!(
            ctx.external_prefix().is_none(),
            "unexpanded image IDs produced a cache signal"
        );
        assert_eq!(ctx.input_tokens(), Some(self.expected_estimate));
        self.selections.fetch_add(1, Ordering::Relaxed);
        workers.first().cloned()
    }
}

struct CountingPrefixIndex(Arc<AtomicUsize>);

#[tonic::async_trait]
impl sgl_kv_indexer::PrefixIndex for CountingPrefixIndex {
    async fn match_prefix(
        &self,
        _: Vec<i64>,
    ) -> Result<sgl_kv_indexer::PrefixOutcome, sgl_kv_indexer::PrefixIndexError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(sgl_kv_indexer::PrefixOutcome::Empty)
    }
}

#[tokio::test]
async fn kimi_k3_image_ids_are_forwarded_but_excluded_from_cache_and_exact_length() {
    for caller_ids in [false, true] {
        for external_index in [false, true] {
            let tokenizer = FakeTokenizer::successful();
            let prefill = MockWorker::start(vec![]).await;
            let decode = MockWorker::start(vec![]).await;
            let mut ctx = tokenizer.context(&prefill, &decode);
            let mut request = json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "Describe the scene."},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
                ]}],
            });
            if caller_ids {
                request["input_ids"] = json!(GENERATED_IDS);
            }
            let selections = Arc::new(AtomicUsize::new(0));
            let queries = Arc::new(AtomicUsize::new(0));
            let context = Arc::get_mut(&mut ctx).unwrap();
            context.policies.insert(
                ModelId(MODEL.into()),
                Arc::new(ImageRoutingProbe {
                    expected_estimate: (serde_json::to_vec(&request).unwrap().len() / 4).max(1)
                        as u64,
                    selections: Arc::clone(&selections),
                }),
            );
            context.block_size_oracle.try_set(1).unwrap();
            if external_index {
                context.prefix_index = Some(Arc::new(CountingPrefixIndex(Arc::clone(&queries))));
            } else {
                // Deliberately cache the placeholder IDs. If the router hashes
                // these, the probe would see a false match for any image URL.
                let tree = Arc::new(HashTree::new());
                tree.insert(
                    &KvWorkerId::new(prefill.url.clone(), 0),
                    None,
                    &compute_block_hashes(GENERATED_IDS, 1),
                );
                context.radix_tree_prefix_provider = Some(RadixTreePrefixProvider::new(
                    tree,
                    Arc::clone(&context.block_size_oracle),
                ));
            }
            assert_eq!(send(ctx, &request).await, StatusCode::OK);
            for body in [captured_body(&prefill).await, captured_body(&decode).await] {
                assert_eq!(body["input_ids"], json!(GENERATED_IDS));
                assert_eq!(body["messages"], request["messages"]);
            }
            assert!(selections.load(Ordering::Relaxed) > 0);
            assert_eq!(queries.load(Ordering::Relaxed), 0);
            assert_eq!(
                tokenizer.captured_requests(),
                if caller_ids { vec![] } else { vec![request] }
            );
        }
    }
}

#[tokio::test]
async fn kimi_k3_invalid_encoder_results_never_dispatch_partial_ids_and_can_recover() {
    let success = json!({"input_ids": GENERATED_IDS}).to_string();
    for (reply, expected_status, keep_worker) in [
        (
            r#"{"error":"tokenizer rejected the request","error_type":"request"}"#,
            StatusCode::BAD_REQUEST,
            true,
        ),
        (
            r#"{"error":"tokenizer failed","error_type":"internal"}"#,
            StatusCode::SERVICE_UNAVAILABLE,
            true,
        ),
        (
            r#"{"input_ids":[]}"#,
            StatusCode::SERVICE_UNAVAILABLE,
            false,
        ),
        (
            r#"{"input_ids":[10,"invalid",20]}"#,
            StatusCode::SERVICE_UNAVAILABLE,
            false,
        ),
        ("invalid-json", StatusCode::SERVICE_UNAVAILABLE, false),
    ] {
        let tokenizer = FakeTokenizer::with_replies(&[reply, &success]);
        let prefill = MockWorker::start(vec![]).await;
        let decode = MockWorker::start(vec![]).await;
        let ctx = tokenizer.context(&prefill, &decode);
        let request = text_request();
        assert_eq!(
            send(Arc::clone(&ctx), &request).await,
            expected_status,
            "{reply}"
        );
        assert!(
            prefill.captured.lock().unwrap().last_body.is_none(),
            "{reply}"
        );
        assert!(
            decode.captured.lock().unwrap().last_body.is_none(),
            "{reply}"
        );
        assert_eq!(
            tokenizer.captured_requests(),
            vec![request.clone()],
            "{reply}"
        );

        // A well-formed worker error response leaves the JSONL protocol in
        // sync, so the worker should remain alive. Transport/protocol failures
        // retire it and start a replacement for the next request.
        let expected_process_count = if keep_worker { 1 } else { 2 };

        // A later request must be able to obtain a healthy encoder instead of
        // keeping the pool wedged after an invalid response or process failure.
        assert_eq!(send(ctx, &request).await, StatusCode::OK, "{reply}");
        assert_eq!(
            tokenizer.started_processes().len(),
            expected_process_count,
            "unexpected worker restart behavior for {reply}"
        );
        for body in [captured_body(&prefill).await, captured_body(&decode).await] {
            assert_eq!(body["input_ids"], json!(GENERATED_IDS), "{reply}");
            assert_eq!(body["messages"], request["messages"]);
        }
        assert_eq!(
            tokenizer.captured_requests(),
            vec![request.clone(), request],
            "{reply}"
        );
    }
}

#[tokio::test]
async fn kimi_k3_timeout_reaps_stalled_process_and_restarts_without_stale_ids() {
    let success = json!({"input_ids": GENERATED_IDS}).to_string();
    let tokenizer = FakeTokenizer::with_replies(&["__stall__", &success]);
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let ctx = tokenizer.context_with_timeout(&prefill, &decode, 1);
    let first_pid = tokenizer.started_processes()[0];
    let request = text_request();

    // A stuck encoder must honor the deadline even if stdout never produces
    // a line. Bound the test too, so a missing timeout cannot hang the suite.
    let status = tokio::time::timeout(Duration::from_secs(5), send(Arc::clone(&ctx), &request))
        .await
        .expect("stalled tokenizer request must finish within its deadline");
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(prefill.captured.lock().unwrap().last_body.is_none());
    assert!(decode.captured.lock().unwrap().last_body.is_none());
    #[cfg(target_os = "linux")]
    assert!(
        !std::path::Path::new(&format!("/proc/{first_pid}")).exists(),
        "timed-out tokenizer process must be killed and reaped"
    );

    assert_eq!(send(ctx, &request).await, StatusCode::OK);
    let process_ids = tokenizer.started_processes();
    assert_eq!(
        process_ids.len(),
        2,
        "next request must restart the process"
    );
    assert_ne!(process_ids[1], first_pid);
    for body in [captured_body(&prefill).await, captured_body(&decode).await] {
        assert_eq!(body["input_ids"], json!(GENERATED_IDS));
        assert_eq!(body["messages"], request["messages"]);
    }
    assert_eq!(
        tokenizer.captured_requests(),
        vec![request.clone(), request]
    );
}

#[tokio::test]
async fn kimi_k3_pre_dispatch_timeout_preserves_healthy_process() {
    let tokenizer = FakeTokenizer::successful();
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let ctx = tokenizer.context(&prefill, &decode);
    let request = text_request();
    let expired = std::time::Instant::now() - Duration::from_millis(1);

    assert!(ctx
        .tokenizers
        .encode_kimi_request(MODEL, &request, expired)
        .is_err());
    assert_eq!(tokenizer.started_processes().len(), 1);

    assert_eq!(send(ctx, &request).await, StatusCode::OK);
    assert_eq!(tokenizer.started_processes().len(), 1);
    assert_eq!(tokenizer.captured_requests(), vec![request]);
}

#[tokio::test]
async fn kimi_k3_restart_finishes_after_request_timeout_for_later_request() {
    let success = json!({"input_ids": GENERATED_IDS}).to_string();
    let tokenizer =
        FakeTokenizer::with_replies_and_restart_delay(&[r#"{"input_ids":[]}"#, &success], 1_500);
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let ctx = tokenizer.context_with_timeouts(&prefill, &decode, 1, 5);
    let request = text_request();

    // The malformed first reply retires the initial process and starts an
    // independent background restart. This request's one-second budget ends
    // before the replacement finishes its 1.5-second startup delay.
    assert_eq!(
        send(Arc::clone(&ctx), &request).await,
        StatusCode::SERVICE_UNAVAILABLE
    );
    let started = std::time::Instant::now();
    let status = tokio::time::timeout(Duration::from_secs(2), send(Arc::clone(&ctx), &request))
        .await
        .expect("request must honor its own deadline while restart continues");
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "request exceeded its one-second tokenizer budget"
    );

    tokio::time::timeout(Duration::from_secs(1), async {
        while tokenizer.started_processes().len() < 2 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("background restart must start independently of the request");
    tokio::time::sleep(Duration::from_millis(700)).await;
    assert_eq!(send(Arc::clone(&ctx), &request).await, StatusCode::OK);
    assert_eq!(tokenizer.started_processes().len(), 2);
    assert_eq!(tokenizer.captured_requests().len(), 2);
    for body in [captured_body(&prefill).await, captured_body(&decode).await] {
        assert_eq!(body["input_ids"], json!(GENERATED_IDS));
        assert_eq!(body["messages"], request["messages"]);
    }
}

#[tokio::test]
async fn kimi_k3_caller_ids_and_audio_bypass_a_busy_encoder() {
    let tokenizer = FakeTokenizer::new("__stall__");
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let ctx = tokenizer.context_with_timeout(&prefill, &decode, 1);
    let stalled_context = Arc::clone(&ctx);
    let stalled = tokio::spawn(async move { send(stalled_context, &text_request()).await });

    // Wait until the sole encoder slot is actually occupied before sending
    // requests that should not need it.
    tokio::time::timeout(Duration::from_millis(500), async {
        while tokenizer.captured_requests().is_empty() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("first request must enter the fake encoder");

    tokio::time::timeout(Duration::from_millis(750), async {
        let mut caller_ids = text_request();
        caller_ids["input_ids"] = json!([7, 8, 9]);
        assert_eq!(send(Arc::clone(&ctx), &caller_ids).await, StatusCode::OK);
        for body in [captured_body(&prefill).await, captured_body(&decode).await] {
            assert_eq!(body["input_ids"], caller_ids["input_ids"]);
        }
        assert!(!stalled.is_finished(), "caller IDs waited for the encoder");

        prefill.captured.lock().unwrap().last_body = None;
        decode.captured.lock().unwrap().last_body = None;
        let media = json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": "YXVkaW8=", "format": "wav"}}
            ]}],
        });
        assert_eq!(send(Arc::clone(&ctx), &media).await, StatusCode::OK);
        for body in [captured_body(&prefill).await, captured_body(&decode).await] {
            assert!(body.get("input_ids").is_none());
            assert_eq!(body["messages"], media["messages"]);
        }
        assert!(!stalled.is_finished(), "media waited for the encoder");
    })
    .await
    .expect("caller IDs and media must bypass the busy encoder slot");

    assert_eq!(stalled.await.unwrap(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(tokenizer.captured_requests(), vec![text_request()]);
    tokio::time::timeout(Duration::from_secs(1), async {
        while tokenizer.started_processes().len() < 2 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("background recovery must finish before the fake tokenizer is removed");
    tokio::time::sleep(Duration::from_millis(50)).await;
}
