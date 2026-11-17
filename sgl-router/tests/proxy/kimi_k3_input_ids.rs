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
use sglang_processor::openai::OpenAiSettings;
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
/// The fake encoder's rendering digest (`tests/fixtures/kimi_k3/fake_tokenizer.py`).
const FINGERPRINT: &str = "fake-fingerprint";

struct FakeTokenizer {
    directory: TempDir,
    /// The Rust tokenizer the router loads beside the fake Python encoder.
    model: TempDir,
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
            model: crate::kimi_fixture::tokenizer(),
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
                tokenizer_path: Some(
                    self.model
                        .path()
                        .join("tiktoken.model")
                        .to_str()
                        .unwrap()
                        .into(),
                ),
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
        assert!(tokenizers.get(MODEL).is_some());
        let registry = Arc::new(WorkerRegistry::default());
        for (worker, mode, port) in [
            (prefill, WorkerMode::Prefill, Some(8997)),
            (decode, WorkerMode::Decode, None),
        ] {
            register_capabilities(&registry, worker, mode, Some(1), port);
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

/// The pool starts beside the Rust tokenizer, from the directory of
/// `--tokenizer-path`, whose `config.json` the encoder reads; it is opt-in.
#[test]
fn kimi_k3_pool_loads_beside_the_rust_tokenizer() {
    let tokenizer = FakeTokenizer::successful();
    let options = ChatEncodingOptions {
        chat_encoder: ChatEncoderKind::KimiK3,
        tokenizer_python: tokenizer.executable.to_str().unwrap().into(),
        tokenizer_timeout_secs: 5,
        ..Default::default()
    };
    let registry = TokenizerRegistry::load_with_options(&tokenizer.config(), options).unwrap();
    assert!(registry.has_kimi_encoder(MODEL));
    assert!(registry.get(MODEL).is_some());
    let init: Value = serde_json::from_str(
        &fs::read_to_string(tokenizer.directory.path().join("init.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        init["tokenizer_path"],
        json!(tokenizer.model.path().to_str().unwrap())
    );

    let tokenizer = FakeTokenizer::successful();
    let registry =
        TokenizerRegistry::load_with_options(&tokenizer.config(), ChatEncodingOptions::default())
            .unwrap();
    assert!(!registry.has_kimi_encoder(MODEL));
    assert!(tokenizer.started_processes().is_empty());
}

/// Endpoints that need the Rust tokenizer keep working beside the encoder.
#[tokio::test]
async fn kimi_k3_rust_tokenizer_serves_the_other_endpoints() {
    let tokenizer = FakeTokenizer::successful();
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let ctx = tokenizer.context(&prefill, &decode);
    let tokenize = Request::post("/v1/tokenize")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": MODEL, "prompt": "hi"}).to_string(),
        ))
        .unwrap();
    let response = build_router(Arc::clone(&ctx))
        .oneshot(tokenize)
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let generate = Request::post("/generate")
        .header("content-type", "application/json")
        .body(Body::from(json!({"text": "hi"}).to_string()))
        .unwrap();
    let response = build_router(Arc::clone(&ctx))
        .oneshot(generate)
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = captured_body(&decode).await;
    assert!(body.get("text").is_none(), "{body}");
    assert_eq!(
        body["input_ids"],
        json!(ctx.tokenizers.encode_prompt(MODEL, "hi").unwrap())
    );
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
        assert_eq!(body["kimi_k3_input_ids"], json!(GENERATED_IDS));
        // Old schemas ignore the dedicated field and encode the messages.
        assert!(body.get("input_ids").is_none());
        assert_eq!(body["messages"], request["messages"]);
        assert_eq!(body["reasoning_effort"], request["reasoning_effort"]);
    }
    assert_eq!(tokenizer.captured_requests(), vec![request]);
}

/// Every selected worker must advertise `kimi_k3_input_ids_version: 1`;
/// otherwise text and image chats stay on the workers, unencoded.
#[tokio::test]
async fn kimi_k3_ids_need_every_selected_worker_to_advertise_version_1() {
    let pd = [
        (None, Some(1)),
        (Some(1), None),
        (Some(0), Some(1)),
        (Some(1), Some(2)),
    ];
    let cases = pd
        .into_iter()
        .map(|(p, d)| (false, p, d))
        .chain([None, Some(2)].map(|version| (true, version, None)));
    for (plain, prefill_version, decode_version) in cases {
        for media in [false, true] {
            let tokenizer = FakeTokenizer::successful();
            let prefill = MockWorker::start(vec![]).await;
            let decode = MockWorker::start(vec![]).await;
            let ctx = tokenizer.context(&prefill, &decode);
            if plain {
                ctx.registry.remove(&WorkerId(decode.url.clone()));
                ctx.registry.remove(&WorkerId(prefill.url.clone()));
                register_capabilities(
                    &ctx.registry,
                    &prefill,
                    WorkerMode::Plain,
                    prefill_version,
                    None,
                );
            } else {
                for (worker, mode, version) in [
                    (&prefill, WorkerMode::Prefill, prefill_version),
                    (&decode, WorkerMode::Decode, decode_version),
                ] {
                    let port = (mode == WorkerMode::Prefill).then_some(8997);
                    register_capabilities(&ctx.registry, worker, mode, version, port);
                }
            }
            let request = request_with_image(media);
            assert_eq!(send(Arc::clone(&ctx), &request).await, StatusCode::OK);
            let workers = if plain {
                vec![&prefill]
            } else {
                vec![&prefill, &decode]
            };
            for worker in workers {
                assert_not_router_encoded(&captured_body(worker).await);
            }
            assert!(tokenizer.captured_requests().is_empty());
            assert_encoder_outcome(&metrics(&ctx).await, "bypassed", 1);
        }
    }
}

#[tokio::test]
async fn kimi_k3_capability_is_rechecked_after_encoding() {
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
    register_capabilities(
        &ctx.registry,
        &prefill,
        WorkerMode::Prefill,
        None,
        Some(8997),
    );
    fs::write(tokenizer.directory.path().join("release"), "").unwrap();
    assert_eq!(sending.await.unwrap(), StatusCode::OK);
    for body in [captured_body(&prefill).await, captured_body(&decode).await] {
        assert_not_router_encoded(&body);
    }
    let exposed = metrics(&ctx).await;
    // Encoding succeeded and consumed CPU even though its result was discarded.
    assert_encoder_outcome(&exposed, "encoded", 1);
    assert_encoder_outcome(&exposed, "bypassed", 0);
    assert!(exposed.contains(&format!(
        "sgl_router_input_ids_forwarding_total{{model_id=\"{MODEL}\",outcome=\"ineligible_worker_capability\"}} 1"
    )));
}

fn register_capabilities(
    registry: &WorkerRegistry,
    worker: &MockWorker,
    mode: WorkerMode,
    version: Option<u32>,
    bootstrap_port: Option<u16>,
) {
    register_worker(
        registry,
        worker,
        mode,
        version,
        bootstrap_port,
        OpenAiSettings::default(),
    );
}

/// What a worker running the fake encoder's checkpoint advertises at `version`.
fn capabilities(version: Option<u32>) -> WorkerCapabilities {
    WorkerCapabilities {
        kimi_k3_input_ids_version: version,
        kimi_k3_input_ids_fingerprint: Some(FINGERPRINT.into()),
        kimi_k3_input_ids_media: true,
    }
}

/// A worker whose `/server_info` reported `settings`, which carry its
/// `--default-chat-template-kwargs`.
fn register_worker(
    registry: &WorkerRegistry,
    worker: &MockWorker,
    mode: WorkerMode,
    version: Option<u32>,
    bootstrap_port: Option<u16>,
    settings: OpenAiSettings,
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
                capabilities: capabilities(version),
                openai: Some(Arc::new(settings)),
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

/// A request template replaces the checkpoint's encoder, so neither the Python
/// encoder nor the Rust renderer renders the chat; the worker renders it, or
/// rejects it without `--trust-request-chat-template`.
#[tokio::test]
async fn kimi_k3_request_chat_templates_render_on_the_worker() {
    let tokenizer = FakeTokenizer::successful();
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let ctx = tokenizer.context(&prefill, &decode);
    let mut request = text_request();
    request["chat_template_kwargs"] = json!({"chat_template": "{{ custom }}"});
    assert_eq!(send(Arc::clone(&ctx), &request).await, StatusCode::OK);
    for body in [captured_body(&prefill).await, captured_body(&decode).await] {
        assert!(body.get("input_ids").is_none(), "{body}");
        assert_not_router_encoded(&body);
        assert_eq!(body["messages"], request["messages"]);
        assert_eq!(
            body["chat_template_kwargs"],
            request["chat_template_kwargs"]
        );
    }
    assert!(tokenizer.captured_requests().is_empty());
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
    for external_index in [false, true] {
        let tokenizer = FakeTokenizer::successful();
        let prefill = MockWorker::start(vec![]).await;
        let decode = MockWorker::start(vec![]).await;
        let mut ctx = tokenizer.context(&prefill, &decode);
        let request = json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Describe the scene."},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
            ]}],
        });
        let selections = Arc::new(AtomicUsize::new(0));
        let queries = Arc::new(AtomicUsize::new(0));
        let context = Arc::get_mut(&mut ctx).unwrap();
        context.policies.insert(
            ModelId(MODEL.into()),
            Arc::new(ImageRoutingProbe {
                expected_estimate: (serde_json::to_vec(&request).unwrap().len() / 4).max(1) as u64,
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
            assert_eq!(body["kimi_k3_input_ids"], json!(GENERATED_IDS));
            assert!(body.get("input_ids").is_none());
            assert_eq!(body["messages"], request["messages"]);
        }
        assert!(selections.load(Ordering::Relaxed) > 0);
        assert_eq!(queries.load(Ordering::Relaxed), 0);
        assert_eq!(tokenizer.captured_requests(), vec![request]);
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

/// The chat took the Rust path, as without the Python encoder.
fn assert_not_router_encoded(body: &Value) {
    assert!(body.get("kimi_k3_input_ids").is_none(), "{body}");
}

#[derive(Debug)]
struct FirstWorkerPolicy {
    preferred_url: String,
    /// The routing tokens the policy must see, when it asks for them.
    expected_tokens: Option<Vec<u32>>,
}

impl Policy for FirstWorkerPolicy {
    fn needs_request_tokens(&self) -> bool {
        self.expected_tokens.is_some()
    }

    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        if let Some(expected) = &self.expected_tokens {
            assert_eq!(ctx.request_tokens(), Some(expected.as_slice()));
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
                    None,
                );
            }
            ctx.policies.insert(
                ModelId(MODEL.into()),
                Arc::new(FirstWorkerPolicy {
                    preferred_url: failed.url.clone(),
                    expected_tokens: None,
                }),
            );
            let request = request_with_image(media);
            assert_eq!(send(Arc::clone(&ctx), &request).await, StatusCode::OK);
            for (worker, supports) in [(&failed, first_supports), (&ok, !first_supports)] {
                let body = captured_body(worker).await;
                if supports {
                    assert_eq!(body["kimi_k3_input_ids"], json!(GENERATED_IDS));
                    assert!(body.get("input_ids").is_none());
                } else {
                    assert_not_router_encoded(&body);
                }
            }
            assert_eq!(tokenizer.captured_requests(), vec![request]);
            assert!(metrics(&ctx).await.contains(&format!(
                "sgl_router_retries_total{{model_id=\"{MODEL}\"}} 1"
            )));
        }
    }
}

/// A chat the encoder skips takes the Rust path, as without the encoder: the
/// Rust tokenizer's tokens drive routing, and the IDs it forwards, if any, go
/// in `input_ids`.
#[tokio::test]
async fn kimi_k3_skipped_chats_route_on_rust_tokens() {
    for disable_forwarding in [true, false] {
        let tokenizer = FakeTokenizer::successful();
        let prefill = MockWorker::start(vec![]).await;
        let decode = MockWorker::start(vec![]).await;
        let mut cfg = tokenizer.config();
        cfg.model.disable_input_ids_forwarding = disable_forwarding;
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
                (mode == WorkerMode::Prefill).then_some(8997),
            );
        }
        let request = text_request();
        let rust_ids = ctx.tokenizers.encode_chat(MODEL, &request).unwrap();
        ctx.policies.insert(
            ModelId(MODEL.into()),
            Arc::new(FirstWorkerPolicy {
                preferred_url: prefill.url.clone(),
                expected_tokens: Some(rust_ids),
            }),
        );
        assert_eq!(send(Arc::clone(&ctx), &request).await, StatusCode::OK);
        for body in [captured_body(&prefill).await, captured_body(&decode).await] {
            assert_not_router_encoded(&body);
        }
        assert!(tokenizer.captured_requests().is_empty());
        let outcome = if disable_forwarding {
            "disabled"
        } else {
            "forwarded"
        };
        assert!(metrics(&ctx).await.contains(&format!(
            "sgl_router_input_ids_forwarding_total{{model_id=\"{MODEL}\",outcome=\"{outcome}\"}} 1"
        )));
    }
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
        let p = captured_body(&prefill).await;
        let d = captured_body(&decode).await;
        assert_eq!(p["kimi_k3_input_ids"], json!(GENERATED_IDS));
        assert_eq!(d["kimi_k3_input_ids"], p["kimi_k3_input_ids"]);
        assert_eq!(p["bootstrap_room"], d["bootstrap_room"]);
        assert_eq!(p["rid"], d["rid"]);
        assert_eq!(tokenizer.captured_requests(), vec![request]);
    }
}

/// The encoder applies the router's `--default-chat-template-kwargs`; a worker
/// reporting other defaults, or none, would take IDs that differ from its own
/// rendering, so it renders locally.
#[tokio::test]
async fn kimi_k3_ids_need_the_workers_default_chat_template_kwargs() {
    let other_defaults = OpenAiSettings {
        default_chat_template_kwargs: json!({"thinking": false}).as_object().cloned(),
        ..Default::default()
    };
    // `None`: the worker's /server_info carried no OpenAI settings to compare.
    for settings in [Some(other_defaults), None] {
        let reports_other_defaults = settings.is_some();
        let tokenizer = FakeTokenizer::successful();
        let prefill = MockWorker::start(vec![]).await;
        let decode = MockWorker::start(vec![]).await;
        let ctx = tokenizer.context(&prefill, &decode);
        let spec = WorkerSpec {
            id: WorkerId(decode.url.clone()),
            url: decode.url.clone(),
            mode: WorkerMode::Decode,
            model_ids: vec![ModelId(MODEL.into())],
            ..Default::default()
        };
        let profile = EngineProfile {
            capabilities: capabilities(Some(1)),
            openai: settings.map(Arc::new),
            ..WireProtocol::default().into()
        };
        ctx.registry.add_with_cb(spec, None, profile).unwrap();
        let request = text_request();
        assert_eq!(send(Arc::clone(&ctx), &request).await, StatusCode::OK);
        for body in [captured_body(&prefill).await, captured_body(&decode).await] {
            assert_not_router_encoded(&body);
            // The Rust path renders with the router's defaults too.
            if reports_other_defaults {
                assert!(body.get("input_ids").is_none(), "{body}");
                assert_eq!(body["messages"], request["messages"]);
            }
        }
        assert!(tokenizer.captured_requests().is_empty());
    }
}

/// Workers take `kimi_k3_input_ids` as the router's own rendering, so a client
/// must not send it.
#[tokio::test]
async fn kimi_k3_router_input_ids_from_clients_are_rejected() {
    let tokenizer = FakeTokenizer::successful();
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let ctx = tokenizer.context(&prefill, &decode);
    let mut request = text_request();
    request["kimi_k3_input_ids"] = json!([1, 2, 3]);
    assert_eq!(
        send(Arc::clone(&ctx), &request).await,
        StatusCode::BAD_REQUEST
    );
    request["kimi_k3_input_ids"] = Value::Null;
    assert_eq!(send(Arc::clone(&ctx), &request).await, StatusCode::OK);
    assert_eq!(
        captured_body(&decode).await["kimi_k3_input_ids"],
        json!(GENERATED_IDS)
    );
}

/// The encoder validates the request the workers receive, so a null that the
/// sampling contract fills must reach it filled; the Python schema rejects it.
#[tokio::test]
async fn kimi_k3_sampling_defaults_reach_the_encoder() {
    let tokenizer = FakeTokenizer::successful();
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let mut cfg = tokenizer.config();
    let argv = [
        "sgl-router",
        "--model-id",
        MODEL,
        "--worker-urls",
        "http://placeholder:0",
        "--override-sampling-params",
        r#"{"n": 1, "frequency_penalty": 0.5}"#,
    ];
    cfg.model.sampling_overrides = <sgl_router::config::Cli as clap::Parser>::parse_from(argv)
        .into_config()
        .unwrap()
        .model
        .sampling_overrides;
    let ctx = tokenizer.context_with_config(&prefill, &decode, cfg, 5, 300);
    let mut request = text_request();
    request["n"] = Value::Null;
    request["frequency_penalty"] = Value::Null;
    assert_eq!(send(ctx, &request).await, StatusCode::OK);
    let encoded = tokenizer.captured_requests();
    assert_eq!(encoded.len(), 1);
    assert_eq!(encoded[0]["n"], json!(1));
    assert_eq!(encoded[0]["frequency_penalty"], json!(0.5));
    assert_eq!(
        captured_body(&decode).await["kimi_k3_input_ids"],
        json!(GENERATED_IDS)
    );
}

/// dynamo-tokenizers has no tiktoken pattern for `kimi_k31`, which uses K3's
/// tokenizer; the router still loads it and encodes as K3, on both backends,
/// and removes the copy it staged to do so.
#[test]
fn kimi_k31_tiktoken_loads_as_kimi_k3() {
    use sgl_router::config::{TokenizerBackend, TokenizerConfig};
    use sgl_router::tokenizer::adapter;
    let k3 = crate::kimi_fixture::tokenizer();
    let k31 = crate::kimi_fixture::tokenizer();
    fs::write(
        k31.path().join("config.json"),
        r#"{"model_type":"kimi_k31"}"#,
    )
    .unwrap();
    let path = |dir: &TempDir| dir.path().join("tiktoken.model").display().to_string();
    let staged = || -> std::collections::HashSet<_> {
        fs::read_dir(std::env::temp_dir())
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name())
            .filter(|name| name.to_string_lossy().starts_with("sgl-router-kimi-k31-"))
            .collect()
    };
    let before = staged();
    for backend in [TokenizerBackend::Hf, TokenizerBackend::Fast] {
        let cfg = TokenizerConfig {
            backend,
            ..Default::default()
        };
        let (k3, _) = adapter::load_with(&path(&k3), cfg).unwrap();
        let (k31, _) = adapter::load_with(&path(&k31), cfg).unwrap();
        let text = "hello <|open|>message world";
        assert_eq!(
            adapter::encode(&k31, text).unwrap(),
            adapter::encode(&k3, text).unwrap(),
            "{backend:?}"
        );
    }
    // Only this test stages here; other processes' stages are left alone.
    assert!(staged().is_subset(&before));
}

/// A worker whose rendering digest differs from the encoder's (another
/// checkpoint, such as K3 beside K3.1, or tokenizer snapshot) gets no router
/// IDs; one without a K3 image processor gets text IDs but not image IDs.
#[tokio::test]
async fn kimi_k3_ids_need_the_encoders_rendering_and_image_support() {
    let other_rendering = WorkerCapabilities {
        kimi_k3_input_ids_fingerprint: Some("other-checkpoint".into()),
        ..capabilities(Some(1))
    };
    let no_images = WorkerCapabilities {
        kimi_k3_input_ids_media: false,
        ..capabilities(Some(1))
    };
    for (capabilities, media, encoded) in [
        (other_rendering.clone(), false, false),
        (other_rendering, true, false),
        (no_images.clone(), false, true),
        (no_images, true, false),
    ] {
        let tokenizer = FakeTokenizer::successful();
        let prefill = MockWorker::start(vec![]).await;
        let decode = MockWorker::start(vec![]).await;
        let ctx = tokenizer.context(&prefill, &decode);
        let spec = WorkerSpec {
            id: WorkerId(decode.url.clone()),
            url: decode.url.clone(),
            mode: WorkerMode::Decode,
            model_ids: vec![ModelId(MODEL.into())],
            ..Default::default()
        };
        let profile = EngineProfile {
            capabilities,
            openai: Some(Arc::new(OpenAiSettings::default())),
            ..WireProtocol::default().into()
        };
        ctx.registry.add_with_cb(spec, None, profile).unwrap();
        let request = request_with_image(media);
        assert_eq!(send(Arc::clone(&ctx), &request).await, StatusCode::OK);
        let renders_alike = capabilities_renders_alike(&ctx, &decode);
        for body in [captured_body(&prefill).await, captured_body(&decode).await] {
            if encoded {
                assert_eq!(body["kimi_k3_input_ids"], json!(GENERATED_IDS));
            } else {
                assert_not_router_encoded(&body);
            }
            // Neither does the Rust path forward IDs to another checkpoint.
            if !renders_alike {
                assert!(body.get("input_ids").is_none(), "{body}");
            }
        }
        assert_eq!(tokenizer.captured_requests().len(), usize::from(encoded));
        if !renders_alike {
            let generate = Request::post("/generate")
                .header("content-type", "application/json")
                .body(Body::from(json!({"text": "hi"}).to_string()))
                .unwrap();
            let response = build_router(Arc::clone(&ctx))
                .oneshot(generate)
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body = captured_body(&decode).await;
            assert_eq!(body["text"], "hi");
            assert!(body.get("input_ids").is_none(), "{body}");
        }
    }
}

fn capabilities_renders_alike(ctx: &AppContext, worker: &MockWorker) -> bool {
    ctx.registry
        .get(&WorkerId(worker.url.clone()))
        .and_then(|w| w.kimi_k3_input_ids_fingerprint().map(str::to_owned))
        .as_deref()
        == Some(FINGERPRINT)
}

/// The pool takes the encoder's digest from its ready reply, and a restart
/// that renders differently is not accepted.
#[test]
fn kimi_k3_pool_requires_an_encoder_fingerprint() {
    let tokenizer = FakeTokenizer::successful();
    let mut scenario: Value = serde_json::from_str(
        &fs::read_to_string(tokenizer.directory.path().join("fixture.json")).unwrap(),
    )
    .unwrap();
    scenario["fingerprint"] = json!("");
    fs::write(
        tokenizer.directory.path().join("fixture.json"),
        scenario.to_string(),
    )
    .unwrap();
    let options = ChatEncodingOptions {
        chat_encoder: ChatEncoderKind::KimiK3,
        tokenizer_python: tokenizer.executable.to_str().unwrap().into(),
        ..Default::default()
    };
    let error = TokenizerRegistry::load_with_options(&tokenizer.config(), options).unwrap_err();
    assert!(format!("{error:#}").contains("fingerprint"), "{error:#}");
}

/// K3.1 has no Rust chat formatter, so the Anthropic token count uses the
/// Python encoder, as the chat route does.
#[tokio::test]
async fn kimi_k31_messages_count_tokens_uses_the_encoder() {
    let tokenizer = FakeTokenizer::successful();
    fs::write(
        tokenizer.model.path().join("config.json"),
        r#"{"model_type":"kimi_k31"}"#,
    )
    .unwrap();
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let ctx = tokenizer.context(&prefill, &decode);
    assert!(!ctx.tokenizers.has_chat_formatter(MODEL));
    let request = Request::post("/v1/messages/count_tokens")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}).to_string(),
        ))
        .unwrap();
    let response = build_router(Arc::clone(&ctx))
        .oneshot(request)
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body, json!({"input_tokens": GENERATED_IDS.len()}));
    assert_eq!(tokenizer.captured_requests().len(), 1);
}

/// The IDs are fixed before selection, and a retry can pick a worker whose
/// breaker is open once every breaker is (the pool fails open). So a worker
/// that renders differently stops router IDs even while it is unhealthy.
#[tokio::test]
async fn kimi_k3_an_unhealthy_worker_that_renders_differently_stops_router_ids() {
    let tokenizer = FakeTokenizer::successful();
    let matching = MockWorker::start(vec![]).await;
    let other = MockWorker::start(vec![]).await;
    let ctx = tokenizer.context(&matching, &other);
    for worker in [&matching, &other] {
        ctx.registry.remove(&WorkerId(worker.url.clone()));
    }
    register_capabilities(&ctx.registry, &matching, WorkerMode::Plain, Some(1), None);
    let spec = WorkerSpec {
        id: WorkerId(other.url.clone()),
        url: other.url.clone(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId(MODEL.into())],
        ..Default::default()
    };
    let profile = EngineProfile {
        capabilities: WorkerCapabilities {
            kimi_k3_input_ids_fingerprint: Some("other-checkpoint".into()),
            ..capabilities(Some(1))
        },
        openai: Some(Arc::new(OpenAiSettings::default())),
        ..WireProtocol::default().into()
    };
    ctx.registry
        .add_with_cb(spec, Some(Default::default()), profile)
        .unwrap();
    let unhealthy = ctx.registry.get(&WorkerId(other.url.clone())).unwrap();
    while unhealthy.breaker.would_allow() {
        unhealthy.breaker.record_failure();
    }

    let generate = Request::post("/generate")
        .header("content-type", "application/json")
        .body(Body::from(json!({"text": "hi"}).to_string()))
        .unwrap();
    let response = build_router(Arc::clone(&ctx))
        .oneshot(generate)
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = captured_body(&matching).await;
    assert_eq!(body["text"], "hi");
    assert!(body.get("input_ids").is_none(), "{body}");

    let request = text_request();
    assert_eq!(send(Arc::clone(&ctx), &request).await, StatusCode::OK);
    let body = captured_body(&matching).await;
    assert!(body.get("input_ids").is_none(), "{body}");
    assert!(body.get("kimi_k3_input_ids").is_some(), "{body}");
}

/// Pause a real upstream request so discovery can add a worker between its
/// first dispatch and retry, without relying on sleeps or reconcile timing.
struct PausedRetryWorker {
    url: String,
    requests: tokio::sync::mpsc::Receiver<(String, Value)>,
    release: Arc<tokio::sync::Notify>,
    _shutdown: tokio::sync::oneshot::Sender<()>,
}

impl PausedRetryWorker {
    async fn start() -> Self {
        use axum::response::IntoResponse;

        let (requests, received) = tokio::sync::mpsc::channel(1);
        let release = Arc::new(tokio::sync::Notify::new());
        let handler_release = Arc::clone(&release);
        let app = axum::Router::new().fallback(
            move |uri: axum::http::Uri, axum::Json(body): axum::Json<Value>| {
                let requests = requests.clone();
                let release = Arc::clone(&handler_release);
                async move {
                    if uri.path() == "/abort_request" {
                        return StatusCode::OK.into_response();
                    }
                    requests.send((uri.path().to_owned(), body)).await.unwrap();
                    release.notified().await;
                    (
                        StatusCode::BAD_GATEWAY,
                        axum::Json(json!({"error": "retry after worker discovery"})),
                    )
                        .into_response()
                }
            },
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (shutdown, stopped) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = stopped.await;
                })
                .await
                .unwrap();
        });
        Self {
            url,
            requests: received,
            release,
            _shutdown: shutdown,
        }
    }

    async fn received(&mut self) -> (String, Value) {
        tokio::time::timeout(Duration::from_secs(2), self.requests.recv())
            .await
            .expect("first worker did not receive the request")
            .expect("first worker stopped")
    }
}

fn register_retry_profile(
    registry: &WorkerRegistry,
    url: &str,
    mode: WorkerMode,
    fingerprint: &str,
    defaults: Option<serde_json::Map<String, Value>>,
) {
    registry
        .add_with_cb(
            WorkerSpec {
                id: WorkerId(url.into()),
                url: url.into(),
                mode,
                model_ids: vec![ModelId(MODEL.into())],
                bootstrap_port: (mode == WorkerMode::Prefill).then_some(8997),
                ..Default::default()
            },
            None,
            EngineProfile {
                // An older worker takes native router IDs but not Python encoder IDs.
                capabilities: WorkerCapabilities {
                    kimi_k3_input_ids_fingerprint: Some(fingerprint.into()),
                    ..capabilities(None)
                },
                openai: Some(Arc::new(OpenAiSettings {
                    default_chat_template_kwargs: defaults,
                    ..Default::default()
                })),
                ..WireProtocol::default().into()
            },
        )
        .unwrap();
}

/// A worker discovered while inference is in flight was absent from the
/// preparation guard. Retrying on it must restore the client's text/messages,
/// including a lowered OpenAI route, while retaining caller-provided IDs.
#[tokio::test]
async fn kimi_k3_native_retries_restore_inputs_for_a_new_incompatible_worker() {
    use sgl_router::config::{ParamSpec, SamplingField};

    for (name, path, request, first_path, defaults_only) in [
        (
            "generate",
            "/generate",
            json!({"text": "hello", "sampling_params": {"max_new_tokens": 8, "frequency_penalty": null}}),
            "/generate",
            false,
        ),
        (
            "generate batch",
            "/generate",
            json!({"text": ["hello", "world"], "sampling_params": [{"max_new_tokens": 8}, {"max_new_tokens": 9}]}),
            "/generate",
            false,
        ),
        (
            "completion",
            "/v1/completions",
            json!({"model": MODEL, "prompt": "hello", "max_tokens": 8}),
            "/generate",
            false,
        ),
        (
            "native chat",
            "/v1/chat/completions",
            json!({"model": MODEL, "messages": [{"role": "user", "content": "hello"}], "return_input_ids_in_sglext": true}),
            "/v1/chat/completions",
            false,
        ),
        (
            "lowered chat",
            "/v1/chat/completions",
            text_request(),
            "/generate",
            false,
        ),
        (
            "chat defaults",
            "/v1/chat/completions",
            json!({"model": MODEL, "messages": [{"role": "user", "content": "hello"}], "return_input_ids_in_sglext": true}),
            "/v1/chat/completions",
            true,
        ),
        (
            "lowered chat defaults",
            "/v1/chat/completions",
            text_request(),
            "/generate",
            true,
        ),
        (
            "embeddings",
            "/v1/embeddings",
            json!({"model": MODEL, "input": ["hello", "world"]}),
            "/v1/embeddings",
            false,
        ),
        (
            "classify",
            "/v1/classify",
            json!({"model": MODEL, "input": "hello"}),
            "/v1/classify",
            false,
        ),
        (
            "caller generate IDs",
            "/generate",
            json!({"input_ids": [7, 8]}),
            "/generate",
            false,
        ),
        (
            "caller chat IDs",
            "/v1/chat/completions",
            json!({"model": MODEL, "messages": [{"role": "user", "content": "hello"}], "input_ids": [7, 8]}),
            "/v1/chat/completions",
            false,
        ),
        (
            "caller completion IDs",
            "/v1/completions",
            json!({"model": MODEL, "prompt": [7, 8], "max_tokens": 8}),
            "/generate",
            false,
        ),
        (
            "caller embedding IDs",
            "/v1/embeddings",
            json!({"model": MODEL, "input": [7, 8]}),
            "/v1/embeddings",
            false,
        ),
    ] {
        let tokenizer = FakeTokenizer::successful();
        let placeholder = MockWorker::start(vec![]).await;
        let next = MockWorker::start(vec![]).await;
        let mut first = PausedRetryWorker::start().await;
        let mut cfg = tokenizer.config();
        cfg.proxy.max_attempts = NonZeroU32::new(2).unwrap();
        cfg.proxy.initial_backoff_ms = 0;
        cfg.proxy.max_backoff_ms = 0;
        cfg.model.sampling_overrides.params.insert(
            SamplingField::FrequencyPenalty,
            ParamSpec::Exact(serde_json::Number::from_f64(0.5).unwrap()),
        );
        let ctx = tokenizer.context_with_config(&placeholder, &next, cfg, 5, 300);
        for worker in ctx.registry.all() {
            ctx.registry.remove(&worker.id);
        }
        register_retry_profile(
            &ctx.registry,
            &first.url,
            WorkerMode::Plain,
            FINGERPRINT,
            None,
        );
        let sending = tokio::spawn({
            let ctx = Arc::clone(&ctx);
            let body = request.to_string();
            async move {
                build_router(ctx)
                    .oneshot(
                        Request::post(path)
                            .header("content-type", "application/json")
                            .body(Body::from(body))
                            .unwrap(),
                    )
                    .await
                    .unwrap()
            }
        });
        let (actual_path, encoded) = first.received().await;
        assert_eq!(actual_path, first_path, "{name}");
        let ids = if request.get("input").is_some() {
            &encoded["input"]
        } else {
            &encoded["input_ids"]
        };
        assert!(
            ids.is_array(),
            "first dispatch did not forward IDs for {name}: {encoded}"
        );
        register_retry_profile(
            &ctx.registry,
            &next.url,
            WorkerMode::Plain,
            if defaults_only {
                FINGERPRINT
            } else {
                "other-checkpoint"
            },
            defaults_only.then(|| json!({"thinking": false}).as_object().unwrap().clone()),
        );
        first.release.notify_one();
        let response = tokio::time::timeout(Duration::from_secs(3), sending)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{name}");
        let response: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), 1 << 20)
                .await
                .unwrap(),
        )
        .unwrap();
        let restored = captured_body(&next).await;
        for key in [
            "text",
            "prompt",
            "messages",
            "input",
            "input_ids",
            "max_tokens",
            "return_input_ids_in_sglext",
        ] {
            if let Some(value) = request.get(key) {
                assert_eq!(&restored[key], value, "{name}: {restored}");
            }
        }
        if request.get("input_ids").is_none() {
            assert!(restored.get("input_ids").is_none(), "{name}: {restored}");
        }
        assert!(
            restored.get("kimi_k3_input_ids").is_none(),
            "{name}: {restored}"
        );
        if path == "/generate" {
            let sampling = &restored["sampling_params"];
            let rows = sampling
                .as_array()
                .map_or_else(|| vec![sampling], |rows| rows.iter().collect());
            for params in rows {
                assert_eq!(
                    params["frequency_penalty"],
                    json!(0.5),
                    "{name}: {restored}"
                );
            }
            if let Some(original) = request.get("sampling_params") {
                if let Some(originals) = original.as_array() {
                    for (index, original) in originals.iter().enumerate() {
                        assert_eq!(
                            sampling[index]["max_new_tokens"], original["max_new_tokens"],
                            "{name}: {restored}"
                        );
                    }
                } else {
                    assert_eq!(
                        sampling["max_new_tokens"], original["max_new_tokens"],
                        "{name}: {restored}"
                    );
                }
            }
        }
        if path == "/v1/chat/completions" || path == "/v1/completions" {
            assert_eq!(
                restored["frequency_penalty"],
                json!(0.5),
                "{name}: {restored}"
            );
            // The mock's original OpenAI response must pass through, without
            // the responder for the first attempt's /generate body.
            assert_eq!(response["object"], "chat.completion", "{name}: {response}");
            assert_eq!(
                response["choices"][0]["message"]["content"], "ok",
                "{name}: {response}"
            );
        }
        assert!(tokenizer.captured_requests().is_empty(), "{name}");
    }
}

/// Either newly selected PD leg can invalidate native IDs. Both workers must
/// receive the restored prompt and the same new bootstrap fields on retry.
#[tokio::test]
async fn kimi_k3_native_pd_retry_restores_both_legs_after_discovery() {
    for changed_leg in [WorkerMode::Prefill, WorkerMode::Decode] {
        let tokenizer = FakeTokenizer::successful();
        let next = MockWorker::start(vec![]).await;
        let decode = MockWorker::start(vec![]).await;
        let mut first = PausedRetryWorker::start().await;
        let mut cfg = tokenizer.config();
        cfg.proxy.max_attempts = NonZeroU32::new(2).unwrap();
        cfg.proxy.initial_backoff_ms = 0;
        cfg.proxy.max_backoff_ms = 0;
        cfg.proxy.stream_after_prefill = true;
        let ctx = tokenizer.context_with_config(&next, &decode, cfg, 5, 300);
        ctx.registry.remove(&WorkerId(next.url.clone()));
        register_retry_profile(
            &ctx.registry,
            &first.url,
            WorkerMode::Prefill,
            FINGERPRINT,
            None,
        );
        register_retry_profile(
            &ctx.registry,
            &decode.url,
            WorkerMode::Decode,
            FINGERPRINT,
            None,
        );
        let sending = tokio::spawn({
            let ctx = Arc::clone(&ctx);
            async move {
                build_router(ctx)
                    .oneshot(
                        Request::post("/generate")
                            .header("content-type", "application/json")
                            .body(Body::from(json!({"text": "hello"}).to_string()))
                            .unwrap(),
                    )
                    .await
                    .unwrap()
            }
        });
        let (_, encoded) = first.received().await;
        assert!(encoded["input_ids"].is_array(), "{encoded}");
        let first_decode = captured_body(&decode).await;
        assert_eq!(first_decode["input_ids"], encoded["input_ids"]);
        register_retry_profile(
            &ctx.registry,
            &next.url,
            WorkerMode::Prefill,
            if changed_leg == WorkerMode::Prefill {
                "other-checkpoint"
            } else {
                FINGERPRINT
            },
            None,
        );
        if changed_leg == WorkerMode::Decode {
            register_retry_profile(
                &ctx.registry,
                &decode.url,
                WorkerMode::Decode,
                "other-checkpoint",
                None,
            );
        }
        first.release.notify_one();
        let response = tokio::time::timeout(Duration::from_secs(3), sending)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{changed_leg:?}");
        let prefill_body = captured_body(&next).await;
        let decode_body = captured_body(&decode).await;
        for body in [&prefill_body, &decode_body] {
            assert_eq!(body["text"], "hello", "{changed_leg:?}: {body}");
            assert!(body.get("input_ids").is_none(), "{changed_leg:?}: {body}");
        }
        for key in ["bootstrap_host", "bootstrap_port", "bootstrap_room", "rid"] {
            assert_eq!(
                prefill_body[key], decode_body[key],
                "{changed_leg:?}: {key}"
            );
        }
        assert_ne!(prefill_body["bootstrap_room"], encoded["bootstrap_room"]);
        assert!(tokenizer.captured_requests().is_empty());
    }
}
