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
use sgl_router::policies::{Policy, SelectionContext};
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::{AppContext, ChatRouting};
use sgl_router::state::kv_events::RadixTreePrefixProvider;
use sgl_router::state::kv_events::{compute_block_hashes, HashTree, KvWorkerId};
use sgl_router::tokenizer::options::{ChatEncoderKind, ChatEncodingOptions};
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::{
    EngineProfile, WireProtocol, Worker, WorkerCapabilities, WorkerRegistry,
};
use std::fs;
use std::num::{NonZeroU32, NonZeroUsize};
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
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
        fs::write(&requests, "").unwrap();
        fs::write(
            directory.path().join("fixture.json"),
            json!({
                "replies": replies,
                "restart_delay_millis": restart_delay_millis,
                "generated_ids": GENERATED_IDS,
            })
            .to_string(),
        )
        .unwrap();
        // Execute a stable script. A freshly written executable can be busy
        // when another test's fork temporarily inherits its open writer.
        symlink(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/kimi_k3/fake_tokenizer.py"),
            &executable,
        )
        .unwrap();
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
                aliases: Vec::new(),
                tokenizer: Default::default(),
                dp_aware: false,
                dp_rank_policy: Default::default(),
                load_based_metric: Default::default(),
                load_based_token_gate: Default::default(),
                reorg_buckets: None,
                reorg_admission: Default::default(),
                // Deliberately no tokenizer.json: K3 uses its Python tokenizer.
                tokenizer_path: Some(self.directory.path().to_str().unwrap().into()),
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
        self.context_with_config(
            prefill,
            decode,
            self.config(),
            tokenizer_timeout_secs,
            tokenizer_startup_timeout_secs,
        )
    }

    fn context_with_config(
        &self,
        prefill: &MockWorker,
        decode: &MockWorker,
        cfg: Config,
        tokenizer_timeout_secs: u64,
        tokenizer_startup_timeout_secs: u64,
    ) -> Arc<AppContext> {
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
            register_capabilities(&registry, worker, mode, Some(1), Some(1), port);
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

async fn metrics(ctx: &Arc<AppContext>) -> String {
    let response = build_router(Arc::clone(ctx))
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    String::from_utf8(
        axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap()
}

fn assert_encoder_outcome(metrics: &str, outcome: &str, count: u64) {
    let sample = format!(
        "sgl_router_encoder_requests_total{{model_id=\"{MODEL}\",outcome=\"{outcome}\"}} {count}"
    );
    assert!(
        metrics.lines().any(|line| line == sample),
        "missing {sample}\n{metrics}"
    );
}

fn assert_encoder_idle(metrics: &str) {
    for sample in [
        format!("sgl_router_encoder_waiting_requests{{model_id=\"{MODEL}\"}} 0"),
        format!("sgl_router_encoder_processes{{model_id=\"{MODEL}\",state=\"ready\"}} 1"),
        format!("sgl_router_encoder_processes{{model_id=\"{MODEL}\",state=\"in_use\"}} 0"),
    ] {
        assert!(
            metrics.lines().any(|line| line == sample),
            "missing {sample}\n{metrics}"
        );
    }
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
    cfg.model.tokenizer_path = Some("tests/fixtures/tiny_tokenizer.json".into());
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
    assert_eq!(p["kimi_k3_input_ids"], json!(GENERATED_IDS));
    assert_eq!(d["kimi_k3_input_ids"], p["kimi_k3_input_ids"]);
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
        assert_eq!(body["kimi_k3_media_input_ids"], json!(GENERATED_IDS));
        // Old schemas ignore the dedicated field and encode the messages.
        assert!(body.get("input_ids").is_none());
        assert!(body.get("kimi_k3_input_ids").is_none());
        assert_eq!(body["messages"], request["messages"]);
        assert_eq!(body["reasoning_effort"], request["reasoning_effort"]);
    }
    assert_eq!(tokenizer.captured_requests(), vec![request]);
}

#[tokio::test]
async fn kimi_k3_media_ids_require_capability_on_both_pd_workers() {
    for (prefill_version, decode_version) in [
        (None, Some(1)),
        (Some(1), None),
        (None, None),
        (Some(0), Some(1)),
        (Some(1), Some(2)),
    ] {
        let tokenizer = FakeTokenizer::successful();
        let prefill = MockWorker::start(vec![]).await;
        let decode = MockWorker::start(vec![]).await;
        let ctx = tokenizer.context(&prefill, &decode);
        for (worker, mode, version) in [
            (&prefill, WorkerMode::Prefill, prefill_version),
            (&decode, WorkerMode::Decode, decode_version),
        ] {
            register_with_media_capability(&ctx.registry, worker, mode, version);
        }
        let mut request = text_request();
        request["messages"][0]["content"] = json!([
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U=", "detail": "high"}}
        ]);
        assert_eq!(send(Arc::clone(&ctx), &request).await, StatusCode::OK);
        // The whole PD pair must use local encoding if either side is old.
        for body in [captured_body(&prefill).await, captured_body(&decode).await] {
            assert!(body.get("input_ids").is_none());
            assert!(body.get("kimi_k3_input_ids").is_none());
            assert!(body.get("kimi_k3_media_input_ids").is_none());
            assert_eq!(body["messages"], request["messages"]);
        }
        let exposed = metrics(&ctx).await;
        assert_encoder_outcome(&exposed, "encoded", 0);
        assert_encoder_outcome(&exposed, "bypassed", 1);
        assert!(tokenizer.captured_requests().is_empty());
        assert!(exposed.contains(&format!(
            "sgl_router_input_ids_forwarding_total{{model_id=\"{MODEL}\",outcome=\"ineligible_worker_capability\"}} 1"
        )));
    }
}

#[tokio::test]
async fn kimi_k3_media_capability_gate_covers_plain_workers_and_preserves_caller_ids() {
    for version in [None, Some(1), Some(2)] {
        for caller_ids in [false, true] {
            let tokenizer = FakeTokenizer::successful();
            let plain = MockWorker::start(vec![]).await;
            let unused = MockWorker::start(vec![]).await;
            let ctx = tokenizer.context(&plain, &unused);
            ctx.registry.remove(&WorkerId(plain.url.clone()));
            ctx.registry.remove(&WorkerId(unused.url.clone()));
            register_with_media_capability(&ctx.registry, &plain, WorkerMode::Plain, version);
            let mut request = text_request();
            request["messages"][0]["content"] = json!([
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
            ]);
            if caller_ids {
                request["input_ids"] = json!([17, 18, 19]);
            }
            assert_eq!(send(ctx, &request).await, StatusCode::OK);
            let body = captured_body(&plain).await;
            assert_eq!(body["messages"], request["messages"]);
            if caller_ids {
                assert_eq!(body["input_ids"], request["input_ids"]);
                assert!(body.get("kimi_k3_media_input_ids").is_none());
                assert!(tokenizer.captured_requests().is_empty());
            } else if version == Some(1) {
                assert_eq!(body["kimi_k3_media_input_ids"], json!(GENERATED_IDS));
                assert!(body.get("input_ids").is_none());
                assert!(body.get("kimi_k3_input_ids").is_none());
            } else {
                assert!(body.get("input_ids").is_none());
                assert!(body.get("kimi_k3_input_ids").is_none());
                assert!(body.get("kimi_k3_media_input_ids").is_none());
                assert!(tokenizer.captured_requests().is_empty());
            }
            assert!(unused.captured.lock().unwrap().last_body.is_none());
        }
    }
}

#[tokio::test]
async fn kimi_k3_media_capability_is_rechecked_after_encoding() {
    let tokenizer = FakeTokenizer::new("__wait_for_release__");
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let ctx = tokenizer.context_with_timeout(&prefill, &decode, 30);
    let mut request = text_request();
    request["messages"][0]["content"] = json!([
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
    ]);
    let sending = {
        let ctx = Arc::clone(&ctx);
        let request = request.clone();
        tokio::spawn(async move { send(ctx, &request).await })
    };
    tokio::time::timeout(Duration::from_secs(2), async {
        while tokenizer.captured_requests().is_empty() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("supported workers should allow image encoding to start");

    // A replacement can lose support while the prompt is being encoded.
    register_with_media_capability(&ctx.registry, &prefill, WorkerMode::Prefill, None);
    fs::write(tokenizer.directory.path().join("release"), "").unwrap();
    assert_eq!(sending.await.unwrap(), StatusCode::OK);
    for body in [captured_body(&prefill).await, captured_body(&decode).await] {
        assert!(body.get("input_ids").is_none());
        assert!(body.get("kimi_k3_input_ids").is_none());
        assert!(body.get("kimi_k3_media_input_ids").is_none());
        assert_eq!(body["messages"], request["messages"]);
    }
    let exposed = metrics(&ctx).await;
    // Encoding succeeded and consumed CPU even though its result was discarded.
    assert_encoder_outcome(&exposed, "encoded", 1);
    assert_encoder_outcome(&exposed, "bypassed", 0);
    assert!(exposed.contains(&format!(
        "sgl_router_input_ids_forwarding_total{{model_id=\"{MODEL}\",outcome=\"ineligible_worker_capability\"}} 1"
    )));
}

fn register_with_media_capability(
    registry: &WorkerRegistry,
    worker: &MockWorker,
    mode: WorkerMode,
    version: Option<u32>,
) {
    register_capabilities(
        registry,
        worker,
        mode,
        Some(1),
        version,
        (mode == WorkerMode::Prefill).then_some(8997),
    );
}

fn register_capabilities(
    registry: &WorkerRegistry,
    worker: &MockWorker,
    mode: WorkerMode,
    text_version: Option<u32>,
    media_version: Option<u32>,
    bootstrap_port: Option<u16>,
) {
    registry
        .add_with_cb(
            WorkerSpec {
                id: WorkerId(worker.url.clone()),
                url: worker.url.clone(),
                mode,
                model_ids: vec![ModelId(MODEL.into())],
                bootstrap_port,
                ..Default::default()
            },
            None,
            EngineProfile {
                capabilities: WorkerCapabilities {
                    kimi_k3_input_ids_version: text_version,
                    kimi_k3_media_input_ids_version: media_version,
                },
                ..WireProtocol::default().into()
            },
        )
        .unwrap();
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
            assert!(body.get("kimi_k3_input_ids").is_none());
            assert_eq!(body["messages"], request["messages"]);
        }
        assert!(tokenizer.captured_requests().is_empty());
    }
}

#[tokio::test]
async fn kimi_k3_request_chat_templates_bypass_encoder() {
    for extra in [
        json!({"chat_template_kwargs": {"chat_template": "{{ custom }}"}}),
        json!({"chat_template": "{{ custom }}"}),
    ] {
        let tokenizer = FakeTokenizer::successful();
        let prefill = MockWorker::start(vec![]).await;
        let decode = MockWorker::start(vec![]).await;
        let ctx = tokenizer.context(&prefill, &decode);
        let mut request = text_request();
        request
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        assert_eq!(send(ctx, &request).await, StatusCode::OK, "{extra}");
        for body in [captured_body(&prefill).await, captured_body(&decode).await] {
            assert!(body.get("input_ids").is_none(), "{extra}");
            assert!(body.get("kimi_k3_input_ids").is_none(), "{extra}");
            assert_eq!(body["messages"], request["messages"]);
            assert_eq!(
                body["chat_template_kwargs"],
                request["chat_template_kwargs"]
            );
        }
        assert!(tokenizer.captured_requests().is_empty(), "{extra}");
    }
}

#[tokio::test]
async fn kimi_k3_disabled_forwarding_skips_image_encoding() {
    let tokenizer = FakeTokenizer::successful();
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let mut cfg = tokenizer.config();
    cfg.model.disable_input_ids_forwarding = true;
    let ctx = tokenizer.context_with_config(&prefill, &decode, cfg, 5, 300);
    let request = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What is in this picture?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
        ]}],
    });
    assert_eq!(send(ctx, &request).await, StatusCode::OK);
    for body in [captured_body(&prefill).await, captured_body(&decode).await] {
        assert!(body.get("input_ids").is_none());
        assert!(body.get("kimi_k3_input_ids").is_none());
        assert_eq!(body["messages"], request["messages"]);
    }
    // Image IDs never drive routing and are not forwarded here, so encoding is waste.
    assert!(tokenizer.captured_requests().is_empty());
}

#[tokio::test]
async fn kimi_k3_workers_are_selected_after_encoding_finishes() {
    let tokenizer = FakeTokenizer::new("__slow__");
    let old_prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let new_prefill = MockWorker::start(vec![]).await;
    let ctx = tokenizer.context(&old_prefill, &decode);
    let sending = {
        let ctx = Arc::clone(&ctx);
        tokio::spawn(async move { send(ctx, &text_request()).await })
    };
    tokio::time::timeout(Duration::from_secs(5), async {
        while tokenizer.captured_requests().is_empty() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("encoding must start");

    // Rolling replacement while the request is still encoding.
    register_capabilities(
        &ctx.registry,
        &new_prefill,
        WorkerMode::Prefill,
        Some(1),
        Some(1),
        Some(8998),
    );
    ctx.registry.remove(&WorkerId(old_prefill.url.clone()));

    assert_eq!(sending.await.unwrap(), StatusCode::OK);
    assert_eq!(
        captured_body(&new_prefill).await["kimi_k3_input_ids"],
        json!(GENERATED_IDS)
    );
    assert!(old_prefill.captured.lock().unwrap().last_body.is_none());
}

#[test]
fn kimi_k3_preparation_deadline_bounds_a_congested_blocking_pool() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .max_blocking_threads(1)
        .enable_all()
        .build()
        .unwrap();
    let tokenizer = FakeTokenizer::successful();
    runtime.block_on(async {
        let prefill = MockWorker::start(vec![]).await;
        let decode = MockWorker::start(vec![]).await;
        let ctx = tokenizer.context_with_timeout(&prefill, &decode, 1);
        // Hold the only blocking thread well past the 1s request deadline.
        let _busy = tokio::task::spawn_blocking(|| std::thread::sleep(Duration::from_secs(3)));
        let start = std::time::Instant::now();
        assert_eq!(
            send(ctx, &text_request()).await,
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert!(
            start.elapsed() < Duration::from_millis(2500),
            "request outlived its deadline: {:?}",
            start.elapsed()
        );
        assert!(prefill.captured.lock().unwrap().last_body.is_none());
        assert!(decode.captured.lock().unwrap().last_body.is_none());
    });
    runtime.shutdown_timeout(Duration::from_secs(5));
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
                if caller_ids {
                    assert_eq!(body["input_ids"], json!(GENERATED_IDS));
                    assert!(body.get("kimi_k3_media_input_ids").is_none());
                } else {
                    assert_eq!(body["kimi_k3_media_input_ids"], json!(GENERATED_IDS));
                    assert!(body.get("input_ids").is_none());
                    assert!(body.get("kimi_k3_input_ids").is_none());
                }
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
    for (reply, expected_status, keep_worker, outcome) in [
        (
            r#"{"error":"tokenizer rejected the request","error_type":"request"}"#,
            StatusCode::BAD_REQUEST,
            true,
            "request_error",
        ),
        (
            r#"{"error":"tokenizer failed","error_type":"internal"}"#,
            StatusCode::SERVICE_UNAVAILABLE,
            true,
            "internal_error",
        ),
        (
            r#"{"input_ids":[]}"#,
            StatusCode::SERVICE_UNAVAILABLE,
            false,
            "unavailable",
        ),
        (
            r#"{"input_ids":[10,"invalid",20]}"#,
            StatusCode::SERVICE_UNAVAILABLE,
            false,
            "unavailable",
        ),
        (
            "invalid-json",
            StatusCode::SERVICE_UNAVAILABLE,
            false,
            "unavailable",
        ),
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
        let exposed = metrics(&ctx).await;
        assert_encoder_outcome(&exposed, outcome, 1);
        assert_encoder_outcome(&exposed, "encoded", 0);
        assert!(!exposed.contains("sgl_router_ingress_tokenize_errors_total{"));
        assert!(!exposed.contains("sgl_router_input_ids_forwarding_total{"));

        // A well-formed worker error response leaves the JSONL protocol in
        // sync, so the worker should remain alive. Transport/protocol failures
        // retire it and start a replacement for the next request.
        let expected_process_count = if keep_worker { 1 } else { 2 };

        // A later request must be able to obtain a healthy encoder instead of
        // keeping the pool wedged after an invalid response or process failure.
        assert_eq!(
            send(Arc::clone(&ctx), &request).await,
            StatusCode::OK,
            "{reply}"
        );
        let exposed = metrics(&ctx).await;
        assert_encoder_outcome(&exposed, outcome, 1);
        assert_encoder_outcome(&exposed, "encoded", 1);
        assert_encoder_idle(&exposed);
        for name in ["queue_duration_seconds", "encode_duration_seconds"] {
            assert!(exposed.contains(&format!(
                "sgl_router_encoder_{name}_count{{model_id=\"{MODEL}\"}} 2"
            )));
        }
        assert_eq!(
            tokenizer.started_processes().len(),
            expected_process_count,
            "unexpected worker restart behavior for {reply}"
        );
        for body in [captured_body(&prefill).await, captured_body(&decode).await] {
            assert_eq!(body["kimi_k3_input_ids"], json!(GENERATED_IDS), "{reply}");
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
    assert_encoder_outcome(&metrics(&ctx).await, "encode_timeout", 1);
    #[cfg(target_os = "linux")]
    tokio::time::timeout(Duration::from_secs(1), async {
        while Path::new(&format!("/proc/{first_pid}")).exists() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("timed-out tokenizer process must be killed and reaped");

    assert_eq!(send(Arc::clone(&ctx), &request).await, StatusCode::OK);
    let exposed = metrics(&ctx).await;
    assert_encoder_outcome(&exposed, "encode_timeout", 1);
    assert_encoder_outcome(&exposed, "encoded", 1);
    assert_encoder_idle(&exposed);
    let process_ids = tokenizer.started_processes();
    assert_eq!(
        process_ids.len(),
        2,
        "next request must restart the process"
    );
    assert_ne!(process_ids[1], first_pid);
    for body in [captured_body(&prefill).await, captured_body(&decode).await] {
        assert_eq!(body["kimi_k3_input_ids"], json!(GENERATED_IDS));
        assert_eq!(body["messages"], request["messages"]);
    }
    assert_eq!(
        tokenizer.captured_requests(),
        vec![request.clone(), request]
    );
}

#[tokio::test]
async fn kimi_k3_cancelled_request_keeps_process_in_use_until_encoding_exits() {
    let success = json!({"input_ids": GENERATED_IDS}).to_string();
    let tokenizer = FakeTokenizer::with_replies(&["__wait_for_release__", &success]);
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let ctx = tokenizer.context_with_timeout(&prefill, &decode, 30);
    let sending = {
        let ctx = Arc::clone(&ctx);
        tokio::spawn(async move { send(ctx, &text_request()).await })
    };
    tokio::time::timeout(Duration::from_secs(2), async {
        while tokenizer.captured_requests().is_empty() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("encoding must start before cancellation");

    sending.abort();
    assert!(sending.await.unwrap_err().is_cancelled());
    let exposed = metrics(&ctx).await;
    assert_encoder_outcome(&exposed, "cancelled", 1);
    assert_encoder_outcome(&exposed, "encoded", 0);
    assert!(exposed.contains(&format!(
        "sgl_router_encoder_processes{{model_id=\"{MODEL}\",state=\"in_use\"}} 1"
    )));
    assert!(!exposed.contains("sgl_router_input_ids_forwarding_total{"));
    assert!(prefill.captured.lock().unwrap().last_body.is_none());
    assert!(decode.captured.lock().unwrap().last_body.is_none());

    // The blocking job still owns its slot after the HTTP future is gone.
    // A later request waits for it and reuses the now-synchronized process.
    let retrying = {
        let ctx = Arc::clone(&ctx);
        tokio::spawn(async move { send(ctx, &text_request()).await })
    };
    let waiting = format!("sgl_router_encoder_waiting_requests{{model_id=\"{MODEL}\"}} 1");
    tokio::time::timeout(Duration::from_secs(2), async {
        while !metrics(&ctx).await.lines().any(|line| line == waiting) {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("later request must wait for the cancelled job's process");
    fs::write(tokenizer.directory.path().join("release"), "").unwrap();
    assert_eq!(retrying.await.unwrap(), StatusCode::OK);
    let exposed = metrics(&ctx).await;
    assert_encoder_outcome(&exposed, "cancelled", 1);
    assert_encoder_outcome(&exposed, "encoded", 1);
    assert_encoder_idle(&exposed);
    assert_eq!(tokenizer.started_processes().len(), 1);
    assert!(exposed.contains(&format!(
        "sgl_router_encoder_encode_duration_seconds_count{{model_id=\"{MODEL}\"}} 2"
    )));
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
    let exposed = metrics(&ctx).await;
    assert_encoder_outcome(&exposed, "unavailable", 1);
    assert_encoder_outcome(&exposed, "queue_timeout", 1);
    assert_encoder_outcome(&exposed, "encoded", 1);
    assert_encoder_idle(&exposed);
    for body in [captured_body(&prefill).await, captured_body(&decode).await] {
        assert_eq!(body["kimi_k3_input_ids"], json!(GENERATED_IDS));
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
            assert!(body.get("kimi_k3_input_ids").is_none());
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

/// A worker supporting the legacy media field alone cannot apply the new text
/// accounting semantics. Both text and images must stay on the worker then.
#[tokio::test]
async fn kimi_k3_generic_capability_is_required_for_text_and_images() {
    for media in [false, true] {
        for plain in [false, true] {
            for version in [None, Some(0), Some(2)] {
                let tokenizer = FakeTokenizer::successful();
                let prefill = MockWorker::start(vec![]).await;
                let decode = MockWorker::start(vec![]).await;
                let ctx = tokenizer.context(&prefill, &decode);
                if plain {
                    ctx.registry.remove(&WorkerId(prefill.url.clone()));
                    ctx.registry.remove(&WorkerId(decode.url.clone()));
                    register_capabilities(
                        &ctx.registry,
                        &prefill,
                        WorkerMode::Plain,
                        version,
                        Some(1),
                        None,
                    );
                } else {
                    register_capabilities(
                        &ctx.registry,
                        &decode,
                        WorkerMode::Decode,
                        version,
                        Some(1),
                        None,
                    );
                }
                let request = request_with_image(media);
                assert_eq!(send(Arc::clone(&ctx), &request).await, StatusCode::OK);
                for worker in [&prefill, &decode]
                    .into_iter()
                    .take(if plain { 1 } else { 2 })
                {
                    assert_local_encoding(&captured_body(worker).await, &request);
                }
                assert!(tokenizer.captured_requests().is_empty());
                assert_encoder_outcome(&metrics(&ctx).await, "bypassed", 1);
            }
        }
    }
}

fn request_with_image(media: bool) -> Value {
    let mut request = text_request();
    if media {
        request["messages"][0]["content"] = json!([
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
        ]);
    }
    request
}

fn assert_local_encoding(body: &Value, request: &Value) {
    assert!(body.get("input_ids").is_none());
    assert!(body.get("kimi_k3_input_ids").is_none());
    assert!(body.get("kimi_k3_media_input_ids").is_none());
    assert_eq!(body["messages"], request["messages"]);
}

#[derive(Debug)]
struct FirstWorkerPolicy {
    preferred_url: String,
    routing_tokens: bool,
}

impl Policy for FirstWorkerPolicy {
    fn needs_request_tokens(&self) -> bool {
        self.routing_tokens
    }

    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        if self.routing_tokens {
            assert_eq!(ctx.request_tokens(), Some(GENERATED_IDS));
            assert_eq!(ctx.input_tokens(), Some(GENERATED_IDS.len() as u64));
        }
        workers
            .iter()
            .find(|worker| worker.url == self.preferred_url)
            .or_else(|| workers.first())
            .cloned()
    }
}

/// Retry selection can move in either direction across a rolling upgrade. The
/// encoder runs once, but every dispatch must independently gate the IDs.
#[tokio::test]
async fn kimi_k3_retries_recheck_capabilities_without_losing_encoded_ids() {
    for media in [false, true] {
        for first_supports in [false, true] {
            let tokenizer = FakeTokenizer::successful();
            let failed = MockWorker::start_returning_error(
                StatusCode::SERVICE_UNAVAILABLE,
                json!({"error": "retry"}),
            )
            .await;
            let ok = MockWorker::start(vec![]).await;
            let mut cfg = tokenizer.config();
            cfg.proxy.max_attempts = NonZeroU32::new(2).unwrap();
            let ctx = tokenizer.context_with_config(&failed, &ok, cfg, 5, 300);
            ctx.registry.remove(&WorkerId(failed.url.clone()));
            ctx.registry.remove(&WorkerId(ok.url.clone()));
            for (worker, supports) in [(&failed, first_supports), (&ok, !first_supports)] {
                register_capabilities(
                    &ctx.registry,
                    worker,
                    WorkerMode::Plain,
                    supports.then_some(1),
                    Some(1),
                    None,
                );
            }
            ctx.policies.insert(
                ModelId(MODEL.into()),
                Arc::new(FirstWorkerPolicy {
                    preferred_url: failed.url.clone(),
                    routing_tokens: false,
                }),
            );
            let request = request_with_image(media);
            assert_eq!(send(Arc::clone(&ctx), &request).await, StatusCode::OK);
            for (worker, supports) in [(&failed, first_supports), (&ok, !first_supports)] {
                let body = captured_body(worker).await;
                if supports {
                    let field = if media {
                        "kimi_k3_media_input_ids"
                    } else {
                        "kimi_k3_input_ids"
                    };
                    assert_eq!(body[field], json!(GENERATED_IDS));
                    assert!(body.get("input_ids").is_none());
                } else {
                    assert_local_encoding(&body, &request);
                }
            }
            assert_eq!(tokenizer.captured_requests(), vec![request]);
            assert!(metrics(&ctx).await.contains(&format!(
                "sgl_router_retries_total{{model_id=\"{MODEL}\"}} 1"
            )));
        }
    }
}

/// Turning off forwarding must still supply exact text tokens to a policy
/// that needs them, even when workers have no dedicated-ID capability.
#[tokio::test]
async fn kimi_k3_disabled_forwarding_keeps_text_tokens_for_routing() {
    let tokenizer = FakeTokenizer::successful();
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let mut cfg = tokenizer.config();
    cfg.model.disable_input_ids_forwarding = true;
    let ctx = tokenizer.context_with_config(&prefill, &decode, cfg, 5, 300);
    for (worker, mode) in [
        (&prefill, WorkerMode::Prefill),
        (&decode, WorkerMode::Decode),
    ] {
        register_capabilities(
            &ctx.registry,
            worker,
            mode,
            None,
            None,
            (mode == WorkerMode::Prefill).then_some(8997),
        );
    }
    ctx.policies.insert(
        ModelId(MODEL.into()),
        Arc::new(FirstWorkerPolicy {
            preferred_url: prefill.url.clone(),
            routing_tokens: true,
        }),
    );
    let request = text_request();
    assert_eq!(send(ctx, &request).await, StatusCode::OK);
    for body in [captured_body(&prefill).await, captured_body(&decode).await] {
        assert_local_encoding(&body, &request);
    }
    assert_eq!(tokenizer.captured_requests(), vec![request]);
}

/// Both policy implementations share asynchronous preparation and PD body
/// construction. Keep the new pool offload covered through the reorg path.
#[tokio::test]
async fn kimi_k3_reorg_routes_text_and_images_with_one_encoding() {
    for media in [false, true] {
        let tokenizer = FakeTokenizer::successful();
        let prefill = MockWorker::start(vec![]).await;
        let decode = MockWorker::start(vec![]).await;
        let mut cfg = tokenizer.config();
        cfg.model.policy = PolicyKind::PowerOfTwo;
        let mut ctx = tokenizer.context_with_config(&prefill, &decode, cfg, 5, 300);
        let context = Arc::get_mut(&mut ctx).unwrap();
        let state = sgl_router::state::kv_events::KvEventIndex::new();
        let (resolver, _) = sgl_router::policies_reorg::factory::build_resolver(
            &context.config.model,
            &state,
            None,
        )
        .unwrap();
        context.chat_routing = ChatRouting::Reorg([(ModelId(MODEL.into()), resolver)].into());
        let request = request_with_image(media);
        assert_eq!(send(ctx, &request).await, StatusCode::OK);
        let field = if media {
            "kimi_k3_media_input_ids"
        } else {
            "kimi_k3_input_ids"
        };
        let p = captured_body(&prefill).await;
        let d = captured_body(&decode).await;
        assert_eq!(p[field], json!(GENERATED_IDS));
        assert_eq!(d[field], p[field]);
        assert_eq!(p["bootstrap_room"], d["bootstrap_room"]);
        assert_eq!(p["rid"], d["rid"]);
        assert_eq!(tokenizer.captured_requests(), vec![request]);
    }
}
