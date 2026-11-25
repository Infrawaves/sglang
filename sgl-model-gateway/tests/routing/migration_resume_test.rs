//! Decode hot migration in HTTP PD mode: a retiring decode hands a chat
//! request off and the gateway resumes it on another prefill/decode pair.

use std::time::{Duration, Instant};

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use smg::config::{RetryConfig, RouterConfig};
use tower::ServiceExt;

use crate::common::{
    mock_worker::{
        set_chat_rejection, set_migration_handoff, set_supports_migration_resume,
        take_chat_requests, MockMigration, MockRejection,
    },
    AppTestContext, TestWorkerConfig,
};

fn url(port: u16) -> String {
    format!("http://127.0.0.1:{port}")
}

fn pd_config(prefill: &[u16], decode: &[u16], router_port: u16) -> RouterConfig {
    RouterConfig::builder()
        .prefill_decode_mode(
            prefill.iter().map(|p| (url(*p), None)).collect(),
            decode.iter().map(|p| url(*p)).collect(),
        )
        .round_robin_policy()
        .decode_migration_resume(true)
        .host("127.0.0.1")
        .port(router_port)
        .max_payload_size(256 * 1024 * 1024)
        .request_timeout_secs(600)
        .worker_startup_timeout_secs(5)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .queue_timeout_secs(60)
        .retry_config(RetryConfig {
            max_retries: 3,
            initial_backoff_ms: 1,
            max_backoff_ms: 5,
            ..Default::default()
        })
        .build_unchecked()
}

async fn start(
    prefill: &[u16],
    decode: &[u16],
    capable: &[u16],
    router_port: u16,
) -> AppTestContext {
    for port in capable {
        set_supports_migration_resume(*port, true);
    }
    let workers = prefill
        .iter()
        .map(|p| TestWorkerConfig::prefill(*p))
        .chain(decode.iter().map(|p| TestWorkerConfig::decode(*p)))
        .collect();
    AppTestContext::new_with_config(pd_config(prefill, decode, router_port), workers).await
}

fn handoff_once() -> MockMigration {
    MockMigration {
        chunks_before_handoff: 2,
        origin_len: 5,
        tokens_per_segment: 3,
        handoff_below_attempt: 1,
    }
}

fn chat_body(stream: bool, max_tokens: u64) -> Value {
    json!({
        "model": "mock-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
        "max_tokens": max_tokens,
        "stream_options": if stream { json!({"include_usage": true}) } else { Value::Null },
    })
}

async fn post_chat(app: &axum::Router, body: &Value) -> (StatusCode, bytes::Bytes) {
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    tokio::time::timeout(Duration::from_secs(10), async {
        let response = app.clone().oneshot(request).await.unwrap();
        let status = response.status();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        (status, body)
    })
    .await
    .expect("chat request must complete")
}

/// The `data:` payloads of an SSE body, in order.
fn sse_payloads(body: &[u8]) -> Vec<String> {
    std::str::from_utf8(body)
        .unwrap()
        .split("\n\n")
        .map(str::trim)
        .filter(|event| !event.is_empty())
        .map(|event| {
            event
                .strip_prefix("data:")
                .expect("data event")
                .trim()
                .to_string()
        })
        .collect()
}

fn json_payloads(payloads: &[String]) -> Vec<Value> {
    payloads
        .iter()
        .filter(|p| p.as_str() != "[DONE]")
        .map(|p| serde_json::from_str(p).unwrap())
        .collect()
}

fn content_of(chunks: &[Value]) -> Vec<String> {
    chunks
        .iter()
        .filter_map(|c| c.pointer("/choices/0/delta/content")?.as_str())
        .map(str::to_string)
        .collect()
}

fn breaker_failures(ctx: &AppTestContext, port: u16) -> u64 {
    ctx.app_context
        .worker_registry
        .get_by_url(&url(port))
        .expect("registered worker")
        .circuit_breaker()
        .total_failures()
}

fn requests_on(ports: &[u16]) -> Vec<Value> {
    ports.iter().flat_map(|p| take_chat_requests(*p)).collect()
}

#[tokio::test]
async fn test_stream_resume_is_one_stream_with_stable_id_and_usage() {
    let (prefill, decodes) = (19940, [19941, 19942]);
    for port in decodes {
        set_migration_handoff(port, handoff_once());
    }
    let ctx = start(&[prefill], &decodes, &[prefill, 19941, 19942], 3940).await;
    let app = ctx.create_app().await;
    take_chat_requests(prefill);
    requests_on(&decodes);

    let (status, body) = post_chat(&app, &chat_body(true, 100)).await;
    assert_eq!(status, StatusCode::OK);
    let payloads = sse_payloads(&body);
    assert_eq!(payloads.last().map(String::as_str), Some("[DONE]"));
    assert_eq!(payloads.iter().filter(|p| *p == "[DONE]").count(), 1);
    assert!(!payloads.iter().any(|p| p.contains("sglang_migration")));

    let chunks = json_payloads(&payloads);
    assert_eq!(
        content_of(&chunks),
        ["s0-0 ", "s0-1 ", "s1-0 ", "s1-1 ", "s1-2 "]
    );
    let first_id = chunks[0]["id"].as_str().unwrap().to_string();
    assert!(first_id.starts_with("chatcmpl-"));
    for chunk in &chunks {
        assert_eq!(chunk["id"], first_id.as_str());
        assert_eq!(chunk["created"], chunks[0]["created"]);
    }
    let usage = &chunks.last().unwrap()["usage"];
    assert_eq!(usage["prompt_tokens"], 5);
    assert_eq!(usage["completion_tokens"], 6);
    assert_eq!(usage["total_tokens"], 11);

    let decode_requests = requests_on(&decodes);
    assert_eq!(decode_requests.len(), 2);
    assert!(decode_requests.iter().all(|r| r["allow_migration"] == true));
    let resumed = decode_requests
        .iter()
        .find(|r| r.get("migration_resume").is_some())
        .expect("a resumed request");
    assert_eq!(
        resumed["migration_resume"],
        json!({
            "origin_input_ids": [0, 1, 2, 3, 4],
            "output_ids": [1000, 1001, 1002],
            "attempt": 1,
            "reasoning_tokens": 0,
        })
    );
    assert_eq!(resumed["max_tokens"], 97);
    assert_eq!(resumed["max_completion_tokens"], 97);
    assert!(resumed["bootstrap_room"].is_u64());
    let prefill_requests = take_chat_requests(prefill);
    assert_eq!(prefill_requests.len(), 2);
    assert!(prefill_requests.iter().all(|r| r["allow_migration"] == true));
    assert_eq!(resumed["bootstrap_room"], prefill_requests[1]["bootstrap_room"]);

    for port in decodes {
        assert_eq!(breaker_failures(&ctx, port), 0, "hand-off is not a failure");
    }
    ctx.shutdown().await;
}

#[tokio::test]
async fn test_non_stream_resume_returns_last_segment_with_first_identity() {
    let (prefill, decodes) = (19943, [19944, 19945]);
    for port in decodes {
        set_migration_handoff(port, handoff_once());
    }
    let ctx = start(&[prefill], &decodes, &[prefill, 19944, 19945], 3941).await;
    let app = ctx.create_app().await;

    let (status, body) = post_chat(&app, &chat_body(false, 100)).await;
    assert_eq!(status, StatusCode::OK);
    let response: Value = serde_json::from_slice(&body).unwrap();
    assert!(response.get("sglang_migration").is_none());
    assert_eq!(response["choices"][0]["message"]["content"], "segment-1");
    assert_eq!(response["usage"]["prompt_tokens"], 5);
    assert_eq!(response["usage"]["completion_tokens"], 6);
    assert_eq!(response["usage"]["total_tokens"], 11);
    // The first segment reported cached_tokens = 100, the resumed one 101.
    assert_eq!(response["usage"]["prompt_tokens_details"]["cached_tokens"], 100);
    ctx.shutdown().await;
}

#[tokio::test]
async fn test_resume_limit_ends_stream_with_error() {
    let (prefill, decodes) = (19946, [19947, 19948]);
    let always = MockMigration {
        handoff_below_attempt: u64::MAX,
        ..handoff_once()
    };
    for port in decodes {
        set_migration_handoff(port, always);
    }
    let ctx = start(&[prefill], &decodes, &[prefill, 19947, 19948], 3942).await;
    let app = ctx.create_app().await;
    requests_on(&decodes);

    let (status, body) = post_chat(&app, &chat_body(true, 1000)).await;
    assert_eq!(status, StatusCode::OK);
    let payloads = sse_payloads(&body);
    assert_eq!(payloads.last().map(String::as_str), Some("[DONE]"));
    let chunks = json_payloads(&payloads);
    assert_eq!(content_of(&chunks).len(), 2 * 4);
    assert!(chunks.last().unwrap().get("error").is_some());
    // The original request plus three resumes.
    assert_eq!(requests_on(&decodes).len(), 4);

    let (status, _) = post_chat(&app, &chat_body(false, 1000)).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    ctx.shutdown().await;
}

#[tokio::test]
async fn test_exhausted_budget_finishes_with_length() {
    let (prefill, decode) = (19949, 19950);
    set_migration_handoff(decode, handoff_once());
    let ctx = start(&[prefill], &[decode], &[prefill, decode], 3943).await;
    let app = ctx.create_app().await;
    requests_on(&[decode]);

    // The hand-off reports 3 generated tokens, the whole budget.
    let (status, body) = post_chat(&app, &chat_body(true, 3)).await;
    assert_eq!(status, StatusCode::OK);
    let payloads = sse_payloads(&body);
    assert_eq!(payloads.last().map(String::as_str), Some("[DONE]"));
    let chunks = json_payloads(&payloads);
    let finish = chunks
        .iter()
        .find(|c| c.pointer("/choices/0/finish_reason").is_some_and(|f| f == "length"))
        .expect("length finish");
    assert_eq!(finish["id"], chunks[0]["id"]);
    assert_eq!(chunks.last().unwrap()["usage"]["completion_tokens"], 3);
    assert_eq!(requests_on(&[decode]).len(), 1, "no resubmission");
    ctx.shutdown().await;
}

#[tokio::test]
async fn test_draining_decode_is_retried_elsewhere_and_excluded() {
    let (prefill, draining, healthy) = (19951, 19952, 19953);
    set_chat_rejection(draining, Some(MockRejection::Draining));
    let ctx = start(&[prefill], &[draining, healthy], &[], 3944).await;
    let app = ctx.create_app().await;
    requests_on(&[draining, healthy]);

    for _ in 0..4 {
        let (status, _) = post_chat(&app, &chat_body(false, 10)).await;
        assert_eq!(status, StatusCode::OK);
    }
    assert_eq!(take_chat_requests(draining).len(), 1, "hit at most once");
    assert_eq!(take_chat_requests(healthy).len(), 4);
    assert_eq!(breaker_failures(&ctx, draining), 0);
    let worker = ctx
        .app_context
        .worker_registry
        .get_by_url(&url(draining))
        .unwrap();
    assert!(worker.is_draining());
    assert!(worker.is_healthy());
    assert!(!worker.is_available());
    set_chat_rejection(draining, None);
    ctx.shutdown().await;
}

#[tokio::test]
async fn test_unmarked_503_still_counts_as_breaker_failure() {
    let (prefill, failing, healthy) = (19954, 19955, 19956);
    set_chat_rejection(failing, Some(MockRejection::Unavailable));
    let ctx = start(&[prefill], &[failing, healthy], &[], 3945).await;
    let app = ctx.create_app().await;

    for _ in 0..2 {
        let (status, _) = post_chat(&app, &chat_body(false, 10)).await;
        assert_eq!(status, StatusCode::OK);
    }
    assert!(breaker_failures(&ctx, failing) >= 1);
    let worker = ctx
        .app_context
        .worker_registry
        .get_by_url(&url(failing))
        .unwrap();
    assert!(!worker.is_draining());
    set_chat_rejection(failing, None);
    ctx.shutdown().await;
}

/// A draining decode answers before its slow prefill; the router must not
/// wait for that prefill before retrying.
#[tokio::test]
async fn test_draining_decode_does_not_wait_for_prefill() {
    let (prefill, draining, healthy) = (19957, 19958, 19959);
    set_chat_rejection(draining, Some(MockRejection::Draining));
    let mut slow_prefill = TestWorkerConfig::prefill(prefill);
    slow_prefill.response_delay_ms = 1500;
    let ctx = AppTestContext::new_with_config(
        pd_config(&[prefill], &[draining, healthy], 3946),
        vec![
            slow_prefill,
            TestWorkerConfig::decode(draining),
            TestWorkerConfig::decode(healthy),
        ],
    )
    .await;
    let app = ctx.create_app().await;

    // Round robin sends one of these two to the draining decode.
    for _ in 0..2 {
        let start = Instant::now();
        let (status, _) = post_chat(&app, &chat_body(false, 10)).await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            start.elapsed() < Duration::from_millis(2600),
            "took {:?}",
            start.elapsed()
        );
    }
    set_chat_rejection(draining, None);
    ctx.shutdown().await;
}

#[tokio::test]
async fn test_resume_only_goes_to_capable_workers() {
    let (prefill, old_decode, new_decode) = (19960, 19961, 19962);
    set_migration_handoff(old_decode, handoff_once());
    set_migration_handoff(new_decode, handoff_once());
    let ctx = start(&[prefill], &[old_decode, new_decode], &[prefill, new_decode], 3947).await;
    let app = ctx.create_app().await;
    requests_on(&[old_decode, new_decode]);

    for _ in 0..2 {
        let (status, body) = post_chat(&app, &chat_body(true, 100)).await;
        assert_eq!(status, StatusCode::OK);
        let chunks = json_payloads(&sse_payloads(&body));
        assert_eq!(content_of(&chunks).len(), 5);
    }
    assert!(take_chat_requests(old_decode)
        .iter()
        .all(|r| r.get("migration_resume").is_none()));
    assert_eq!(
        take_chat_requests(new_decode)
            .iter()
            .filter(|r| r.get("migration_resume").is_some())
            .count(),
        2
    );
    ctx.shutdown().await;
}

#[tokio::test]
async fn test_no_capable_pair_ends_stream_with_error() {
    let (prefill, decode) = (19963, 19964);
    set_migration_handoff(decode, handoff_once());
    // The prefill predates resume support.
    let ctx = start(&[prefill], &[decode], &[decode], 3948).await;
    let app = ctx.create_app().await;
    requests_on(&[decode]);

    let (status, body) = post_chat(&app, &chat_body(true, 100)).await;
    assert_eq!(status, StatusCode::OK);
    let payloads = sse_payloads(&body);
    assert_eq!(payloads.last().map(String::as_str), Some("[DONE]"));
    let chunks = json_payloads(&payloads);
    assert_eq!(content_of(&chunks), ["s0-0 ", "s0-1 "]);
    assert!(chunks.last().unwrap().get("error").is_some());
    assert_eq!(requests_on(&[decode]).len(), 1);
    ctx.shutdown().await;
}
