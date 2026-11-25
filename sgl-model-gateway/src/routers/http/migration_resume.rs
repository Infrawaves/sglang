//! Decode hot migration: resume chat requests that a retiring decode handed off.
//!
//! A decode paused with `pause_generation(mode="migrate")` ends every request
//! that carries `allow_migration` with a hand-off: one SSE event (or a
//! non-streaming JSON field) named `sglang_migration` holding the prompt and
//! the tokens generated so far. The gateway resubmits the original chat
//! request plus `migration_resume` to a new prefill/decode pair and stitches
//! the continuation into the client's response. This module holds the pure
//! pieces; `pd_router` drives the dispatch.

use axum::http::{HeaderMap, StatusCode};
use bytes::Bytes;
use memchr::memmem;
use serde::Deserialize;
use serde_json::{json, Map, Value};

/// Marker the engine requires before it freezes a request instead of failing it.
pub const ALLOW_MIGRATION_KEY: &str = "allow_migration";
pub const MIGRATION_EVENT_KEY: &str = "sglang_migration";
pub const MIGRATION_RESUME_KEY: &str = "migration_resume";
/// Response header a retiring engine sets on the 503s it answers new requests with.
pub const DRAINING_HEADER: &str = "x-sglang-draining";
/// A request that keeps landing on retiring decodes is given up on.
pub const MAX_RESUMES: u32 = 3;
pub const DONE_EVENT: &[u8] = b"data: [DONE]\n\n";

/// Fields that point at the previous prefill/decode pair.
const PER_PAIR_FIELDS: [&str; 5] = [
    "bootstrap_host",
    "bootstrap_port",
    "bootstrap_room",
    "routed_dp_rank",
    "disagg_prefill_dp_rank",
];

/// The `sglang_migration` object of a hand-off. `output_ids` is cumulative
/// over every segment so far, as the engine sees it.
#[derive(Debug, Clone, Deserialize)]
pub struct HandOff {
    #[serde(default)]
    pub index: u32,
    pub origin_input_ids: Vec<i64>,
    pub output_ids: Vec<i64>,
    #[serde(default)]
    pub kv_flushed_tokens: u64,
    /// Reasoning tokens among `output_ids`, counted by the engine.
    #[serde(default)]
    pub reasoning_tokens: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumeStop {
    /// The original output budget is used up; finish with `length`.
    BudgetExhausted,
    /// More than `MAX_RESUMES` hand-offs.
    LimitExceeded,
    /// The original request is not a JSON object.
    InvalidRequest,
}

/// Only the explicit marker identifies a planned retirement; a bare 503 is a fault.
pub fn is_draining_rejection(status: StatusCode, headers: &HeaderMap) -> bool {
    status == StatusCode::SERVICE_UNAVAILABLE
        && headers
            .get(DRAINING_HEADER)
            .is_some_and(|v| v.as_bytes().trim_ascii() == b"1")
}

pub fn mark_allow_migration(body: &mut Value) {
    if let Some(obj) = body.as_object_mut() {
        obj.insert(ALLOW_MIGRATION_KEY.to_string(), Value::Bool(true));
    }
}

/// Removes and parses the hand-off from a chunk or response object.
pub fn take_handoff(json: &mut Value) -> Option<HandOff> {
    let raw = json.as_object_mut()?.remove(MIGRATION_EVENT_KEY)?;
    match serde_json::from_value(raw) {
        Ok(handoff) => Some(handoff),
        Err(e) => {
            tracing::warn!("Ignoring malformed {} object: {}", MIGRATION_EVENT_KEY, e);
            None
        }
    }
}

/// Cheap byte check before parsing a payload for a hand-off.
pub fn may_contain_handoff(payload: &[u8]) -> bool {
    memmem::find(payload, MIGRATION_EVENT_KEY.as_bytes()).is_some()
}

/// The envelope and hand-off of a hand-off SSE event, or None for any other event.
pub fn parse_handoff_event(event: &[u8]) -> Option<(Value, HandOff)> {
    if !may_contain_handoff(event) {
        return None;
    }
    let mut envelope: Value = serde_json::from_slice(event_data(event)?).ok()?;
    let handoff = take_handoff(&mut envelope)?;
    Some((envelope, handoff))
}

/// The payload of a single-line `data: ...` SSE event.
pub fn event_data(event: &[u8]) -> Option<&[u8]> {
    event
        .trim_ascii()
        .strip_prefix(b"data:")
        .map(|rest| rest.trim_ascii())
}

pub fn to_sse(payload: &Value) -> Bytes {
    let mut out = b"data: ".to_vec();
    // Serializing a Value cannot fail.
    serde_json::to_writer(&mut out, payload).unwrap_or_default();
    out.extend_from_slice(b"\n\n");
    Bytes::from(out)
}

/// An SSE error event in the shape `handle_decode_error_response` uses.
pub fn error_event(status: StatusCode, body: &[u8]) -> Bytes {
    let message = serde_json::from_slice::<Value>(body)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(body).into_owned()));
    to_sse(&json!({ "error": { "message": message, "status": status.as_u16() } }))
}

/// Splits a byte stream into complete `\n\n`-terminated SSE events. The
/// hand-off event carries the whole token history and can span many chunks.
#[derive(Debug, Default)]
pub struct SseEventSplitter {
    buffer: Vec<u8>,
    // Bytes before this offset hold no delimiter.
    scanned: usize,
}

impl SseEventSplitter {
    pub fn feed(&mut self, chunk: &[u8]) -> Vec<Bytes> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();
        let mut start = 0;
        let mut search_from = self.scanned;
        while let Some(pos) = memmem::find(&self.buffer[search_from..], b"\n\n") {
            let end = search_from + pos + 2;
            events.push(Bytes::copy_from_slice(&self.buffer[start..end]));
            start = end;
            search_from = end;
        }
        self.buffer.drain(..start);
        // The delimiter may straddle this chunk and the next one.
        self.scanned = self.buffer.len().saturating_sub(1);
        events
    }

    /// The unterminated tail, if it holds anything but whitespace.
    pub fn flush(&mut self) -> Option<Bytes> {
        self.scanned = 0;
        let rest = std::mem::take(&mut self.buffer);
        (!rest.iter().all(u8::is_ascii_whitespace)).then(|| Bytes::from(rest))
    }
}

/// Builds the body that continues `original` after `handoff`.
pub fn build_resume_body(
    original: &Value,
    handoff: &HandOff,
    attempt: u32,
) -> Result<Value, ResumeStop> {
    let mut body = original.clone();
    let obj = body.as_object_mut().ok_or(ResumeStop::InvalidRequest)?;
    for key in PER_PAIR_FIELDS {
        obj.remove(key);
    }
    // Same precedence as the engine: max_completion_tokens, else max_tokens.
    let budget = ["max_completion_tokens", "max_tokens"]
        .iter()
        .find_map(|key| obj.get(*key).and_then(Value::as_u64));
    if let Some(budget) = budget {
        let generated = handoff.output_ids.len() as u64;
        if budget <= generated {
            return Err(ResumeStop::BudgetExhausted);
        }
        let remaining = Value::from(budget - generated);
        obj.insert("max_tokens".to_string(), remaining.clone());
        obj.insert("max_completion_tokens".to_string(), remaining);
    }
    // The engine counts min_tokens against this segment's output only.
    if let Some(min_tokens) = obj.get("min_tokens").and_then(Value::as_u64) {
        let generated = handoff.output_ids.len() as u64;
        obj.insert(
            "min_tokens".to_string(),
            min_tokens.saturating_sub(generated).into(),
        );
    }
    obj.insert(
        MIGRATION_RESUME_KEY.to_string(),
        json!({
            "origin_input_ids": handoff.origin_input_ids,
            "output_ids": handoff.output_ids,
            "attempt": attempt,
            "reasoning_tokens": handoff.reasoning_tokens,
        }),
    );
    obj.insert(ALLOW_MIGRATION_KEY.to_string(), Value::Bool(true));
    Ok(body)
}

/// Stitching state of one client request across its segments.
#[derive(Debug, Default)]
pub struct ResumeState {
    attempts: u32,
    id: Option<Value>,
    created: Option<Value>,
    model: Option<Value>,
    /// Fallback prompt count for usage the gateway builds itself; relayed
    /// usage keeps the engine's own count, which applies template adjustments.
    prompt_tokens: u64,
    /// Tokens generated before the current segment.
    generated: u64,
    /// Reasoning tokens among `generated`.
    earlier_reasoning: u64,
    first_cached_tokens: Option<Value>,
}

impl ResumeState {
    pub fn attempts(&self) -> u32 {
        self.attempts
    }

    pub fn is_resumed(&self) -> bool {
        self.attempts > 0
    }

    /// Remembers the first segment's `cached_tokens`, which later segments keep.
    pub fn observe_first_segment(&mut self, json: &Value) {
        if self.is_resumed() || self.first_cached_tokens.is_some() {
            return;
        }
        self.first_cached_tokens = json
            .pointer("/usage/prompt_tokens_details/cached_tokens")
            .cloned();
    }

    /// Records a hand-off and returns the body that resumes it.
    pub fn hand_off(
        &mut self,
        envelope: &Value,
        handoff: &HandOff,
        original: &Value,
    ) -> Result<Value, ResumeStop> {
        if !self.is_resumed() {
            self.id = envelope.get("id").cloned();
            self.created = envelope.get("created").cloned();
            self.model = envelope.get("model").cloned();
            self.prompt_tokens = handoff.origin_input_ids.len() as u64;
        }
        self.attempts += 1;
        self.generated = handoff.output_ids.len() as u64;
        self.earlier_reasoning = handoff.reasoning_tokens;
        let body = build_resume_body(original, handoff, self.attempts)?;
        if self.attempts > MAX_RESUMES {
            return Err(ResumeStop::LimitExceeded);
        }
        Ok(body)
    }

    /// A relayed SSE event, rewritten as part of the first segment once resumed.
    pub fn rewrite_event(&mut self, event: Bytes) -> Bytes {
        if !self.is_resumed() {
            let has_usage = memmem::find(&event, b"\"usage\"").is_some();
            if self.first_cached_tokens.is_none() && has_usage {
                if let Some(json) =
                    event_data(&event).and_then(|d| serde_json::from_slice::<Value>(d).ok())
                {
                    self.observe_first_segment(&json);
                }
            }
            return event;
        }
        let parsed = match event_data(&event) {
            Some(data) if data != b"[DONE]" => serde_json::from_slice::<Value>(data).ok(),
            _ => None,
        };
        let Some(mut json) = parsed else {
            return event;
        };
        self.rewrite_chunk(&mut json);
        to_sse(&json)
    }

    /// Keeps the first segment's `id` / `created` and accumulates usage.
    pub fn rewrite_chunk(&self, json: &mut Value) {
        let Some(obj) = json.as_object_mut() else {
            return;
        };
        if let Some(id) = &self.id {
            obj.insert("id".to_string(), id.clone());
        }
        if let Some(created) = &self.created {
            obj.insert("created".to_string(), created.clone());
        }
        if let Some(usage) = obj.get_mut("usage").and_then(Value::as_object_mut) {
            self.rewrite_usage(usage);
        }
        // Some encodings also report usage per choice in the end frame.
        if let Some(choices) = obj.get_mut("choices").and_then(Value::as_array_mut) {
            for choice in choices.iter_mut().filter_map(Value::as_object_mut) {
                if let Some(usage) = choice.get_mut("usage").and_then(Value::as_object_mut) {
                    self.rewrite_usage(usage);
                }
            }
        }
    }

    fn rewrite_usage(&self, usage: &mut Map<String, Value>) {
        // The resumed engine already reports the original prompt; only the
        // earlier output is missing from completion.
        let prompt = usage
            .get("prompt_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(self.prompt_tokens);
        let completion = usage
            .get("completion_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            + self.generated;
        usage.insert("prompt_tokens".to_string(), prompt.into());
        usage.insert("completion_tokens".to_string(), completion.into());
        usage.insert("total_tokens".to_string(), (prompt + completion).into());
        add_to(usage.get_mut("reasoning_tokens"), self.earlier_reasoning);
        if let Some(details) = usage
            .get_mut("completion_tokens_details")
            .and_then(Value::as_object_mut)
        {
            add_to(details.get_mut("reasoning_tokens"), self.earlier_reasoning);
        }
        if let (Some(cached), Some(details)) = (
            &self.first_cached_tokens,
            usage
                .get_mut("prompt_tokens_details")
                .and_then(Value::as_object_mut),
        ) {
            details.insert("cached_tokens".to_string(), cached.clone());
        }
    }

    fn usage(&self) -> Value {
        json!({
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.generated,
            "total_tokens": self.prompt_tokens + self.generated,
        })
    }

    /// Events that end a stream whose budget ran out exactly at a hand-off.
    pub fn length_finish_events(&self, include_usage: bool) -> Vec<Bytes> {
        let mut finish = json!({
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
        });
        self.stamp(&mut finish);
        let mut events = vec![to_sse(&finish)];
        if include_usage {
            let mut usage = json!({
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": self.usage(),
            });
            self.stamp(&mut usage);
            events.push(to_sse(&usage));
        }
        events.push(Bytes::from_static(DONE_EVENT));
        events
    }

    /// A non-streaming hand-off response turned into a `length` finish.
    pub fn length_finish_response(&self, mut response: Value) -> Value {
        if let Some(choices) = response.get_mut("choices").and_then(Value::as_array_mut) {
            for choice in choices.iter_mut().filter_map(Value::as_object_mut) {
                choice.insert("finish_reason".to_string(), json!("length"));
            }
        }
        self.stamp(&mut response);
        if let Some(obj) = response.as_object_mut() {
            obj.insert("usage".to_string(), self.usage());
        }
        response
    }

    fn stamp(&self, json: &mut Value) {
        let Some(obj) = json.as_object_mut() else {
            return;
        };
        for (key, value) in [
            ("id", &self.id),
            ("created", &self.created),
            ("model", &self.model),
        ] {
            if let Some(value) = value {
                obj.insert(key.to_string(), value.clone());
            }
        }
    }
}

fn add_to(value: Option<&mut Value>, extra: u64) {
    if let Some(value) = value {
        if let Some(current) = value.as_u64() {
            *value = Value::from(current + extra);
        }
    }
}

#[cfg(test)]
mod tests {
    use axum::http::HeaderValue;

    use super::*;

    fn handoff(origin: usize, output: usize) -> HandOff {
        HandOff {
            index: 0,
            origin_input_ids: (0..origin as i64).collect(),
            output_ids: (100..100 + output as i64).collect(),
            kv_flushed_tokens: 0,
            reasoning_tokens: 0,
        }
    }

    fn handoff_event(id: &str, created: u64, origin: usize, output: usize) -> Vec<u8> {
        let payload = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "m",
            "choices": [],
            "sglang_migration": {
                "index": 0,
                "origin_input_ids": (0..origin as i64).collect::<Vec<_>>(),
                "output_ids": (100..100 + output as i64).collect::<Vec<_>>(),
                "kv_flushed_tokens": 7,
            }
        });
        to_sse(&payload).to_vec()
    }

    #[test]
    fn splitter_joins_events_across_chunks() {
        let mut splitter = SseEventSplitter::default();
        assert!(splitter.feed(b"data: {\"a\":").is_empty());
        assert!(splitter.feed(b"1}\n").is_empty());
        let events = splitter.feed(b"\ndata: [DONE]\n\ndata: x");
        assert_eq!(
            events,
            vec![
                Bytes::from_static(b"data: {\"a\":1}\n\n"),
                Bytes::from_static(b"data: [DONE]\n\n"),
            ]
        );
        assert_eq!(splitter.flush(), Some(Bytes::from_static(b"data: x")));
        assert_eq!(splitter.flush(), None);
    }

    #[test]
    fn splitter_handles_one_byte_chunks() {
        let stream = handoff_event("c1", 5, 3, 2);
        let mut splitter = SseEventSplitter::default();
        let mut events = Vec::new();
        for byte in &stream {
            events.extend(splitter.feed(std::slice::from_ref(byte)));
        }
        assert_eq!(events, vec![Bytes::from(stream)]);
    }

    #[test]
    fn parses_handoff_event_and_ignores_others() {
        let (envelope, handoff) = parse_handoff_event(&handoff_event("c1", 5, 3, 2)).unwrap();
        assert_eq!(envelope["id"], "c1");
        assert!(envelope.get(MIGRATION_EVENT_KEY).is_none());
        assert_eq!(handoff.origin_input_ids, vec![0, 1, 2]);
        assert_eq!(handoff.output_ids, vec![100, 101]);
        assert_eq!(handoff.kv_flushed_tokens, 7);

        assert!(parse_handoff_event(b"data: {\"choices\":[]}\n\n").is_none());
        assert!(parse_handoff_event(b"data: [DONE]\n\n").is_none());
    }

    #[test]
    fn draining_needs_the_marker() {
        let mut headers = HeaderMap::new();
        assert!(!is_draining_rejection(
            StatusCode::SERVICE_UNAVAILABLE,
            &headers
        ));
        headers.insert(DRAINING_HEADER, HeaderValue::from_static("1"));
        assert!(is_draining_rejection(
            StatusCode::SERVICE_UNAVAILABLE,
            &headers
        ));
        assert!(!is_draining_rejection(
            StatusCode::INTERNAL_SERVER_ERROR,
            &headers
        ));
    }

    #[test]
    fn resume_body_shrinks_both_budgets_and_drops_pair_fields() {
        let original = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "max_tokens": 50,
            "max_completion_tokens": 10,
            "bootstrap_room": 1,
            "routed_dp_rank": 2,
            "disagg_prefill_dp_rank": 3,
        });
        let body = build_resume_body(&original, &handoff(3, 4), 2).unwrap();
        assert_eq!(body["max_tokens"], 6);
        assert_eq!(body["max_completion_tokens"], 6);
        assert_eq!(body["stream"], true);
        assert_eq!(body["allow_migration"], true);
        assert_eq!(body["messages"], original["messages"]);
        assert_eq!(
            body["migration_resume"],
            json!({
                "origin_input_ids": [0, 1, 2],
                "output_ids": [100, 101, 102, 103],
                "attempt": 2,
                "reasoning_tokens": 0,
            })
        );
        for key in PER_PAIR_FIELDS {
            assert!(body.get(key).is_none(), "{key}");
        }
    }

    #[test]
    fn resume_body_uses_max_tokens_and_leaves_unbounded_requests_unbounded() {
        let body = build_resume_body(&json!({"max_tokens": 10}), &handoff(1, 4), 1).unwrap();
        assert_eq!(body["max_tokens"], 6);
        assert_eq!(body["max_completion_tokens"], 6);

        let body =
            build_resume_body(&json!({"max_tokens": null}), &handoff(1, 4), 1).unwrap();
        assert!(body["max_tokens"].is_null());
        assert!(body.get("max_completion_tokens").is_none());
    }

    #[test]
    fn resume_body_shrinks_min_tokens_to_what_is_left() {
        let body = build_resume_body(&json!({"min_tokens": 6}), &handoff(1, 4), 1).unwrap();
        assert_eq!(body["min_tokens"], 2);
        let body = build_resume_body(&json!({"min_tokens": 3}), &handoff(1, 4), 1).unwrap();
        assert_eq!(body["min_tokens"], 0);
    }

    #[test]
    fn resume_body_stops_when_budget_is_used_up() {
        assert_eq!(
            build_resume_body(&json!({"max_tokens": 4}), &handoff(1, 4), 1),
            Err(ResumeStop::BudgetExhausted)
        );
    }

    #[test]
    fn hand_off_enforces_the_resume_limit() {
        let mut state = ResumeState::default();
        let envelope = json!({"id": "c1", "created": 1});
        for _ in 0..MAX_RESUMES {
            assert!(state.hand_off(&envelope, &handoff(1, 1), &json!({})).is_ok());
        }
        assert_eq!(
            state.hand_off(&envelope, &handoff(1, 1), &json!({})),
            Err(ResumeStop::LimitExceeded)
        );
    }

    #[test]
    fn rewrite_keeps_first_identity_and_accumulates_usage() {
        let mut state = ResumeState::default();
        state.observe_first_segment(&json!({
            "usage": {"prompt_tokens_details": {"cached_tokens": 2}}
        }));
        let envelope = json!({"id": "first", "created": 11, "model": "m"});
        let mut earlier = handoff(3, 4);
        earlier.reasoning_tokens = 4;
        state.hand_off(&envelope, &earlier, &json!({})).unwrap();

        let event = to_sse(&json!({
            "id": "second",
            "created": 22,
            "choices": [],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 5,
                "total_tokens": 12,
                "reasoning_tokens": 5,
                "completion_tokens_details": {"reasoning_tokens": 5},
                "prompt_tokens_details": {"cached_tokens": 6},
            }
        }));
        let out = state.rewrite_event(event);
        let json: Value = serde_json::from_slice(event_data(&out).unwrap()).unwrap();
        assert_eq!(json["id"], "first");
        assert_eq!(json["created"], 11);
        // The engine's prompt count is kept: it applies template adjustments.
        assert_eq!(json["usage"]["prompt_tokens"], 7);
        assert_eq!(json["usage"]["completion_tokens"], 9);
        assert_eq!(json["usage"]["total_tokens"], 16);
        assert_eq!(json["usage"]["reasoning_tokens"], 9);
        assert_eq!(
            json["usage"]["completion_tokens_details"]["reasoning_tokens"],
            9
        );
        assert_eq!(json["usage"]["prompt_tokens_details"]["cached_tokens"], 2);
    }

    #[test]
    fn rewrite_keeps_earlier_reasoning_after_the_reasoning_ended() {
        // A segment that starts after the reasoning closed reports none of
        // its own; the reasoning of earlier segments still counts.
        let mut state = ResumeState::default();
        let mut earlier = handoff(3, 4);
        earlier.reasoning_tokens = 3;
        state
            .hand_off(&json!({"id": "first"}), &earlier, &json!({}))
            .unwrap();
        let out = state.rewrite_event(to_sse(&json!({
            "id": "x",
            "choices": [{"index": 0, "usage": {"prompt_tokens": 3, "completion_tokens": 1}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "reasoning_tokens": 0}
        })));
        let json: Value = serde_json::from_slice(event_data(&out).unwrap()).unwrap();
        assert_eq!(json["usage"]["reasoning_tokens"], 3);
        assert_eq!(json["usage"]["completion_tokens"], 5);
        assert_eq!(json["choices"][0]["usage"]["completion_tokens"], 5);
        assert_eq!(json["choices"][0]["usage"]["total_tokens"], 8);

        let done = Bytes::from_static(DONE_EVENT);
        assert_eq!(state.rewrite_event(done.clone()), done);
    }

    #[test]
    fn first_segment_events_pass_through_unchanged() {
        let mut state = ResumeState::default();
        let event = Bytes::from_static(b"data: {\"id\":\"a\",  \"choices\":[]}\n\n");
        assert_eq!(state.rewrite_event(event.clone()), event);
    }

    #[test]
    fn length_finish_reports_generated_tokens() {
        let mut state = ResumeState::default();
        let envelope = json!({"id": "first", "created": 1, "model": "m"});
        state
            .hand_off(&envelope, &handoff(3, 4), &json!({}))
            .unwrap();
        let events = state.length_finish_events(true);
        assert_eq!(events.len(), 3);
        let finish: Value = serde_json::from_slice(event_data(&events[0]).unwrap()).unwrap();
        assert_eq!(finish["id"], "first");
        assert_eq!(finish["choices"][0]["finish_reason"], "length");
        let usage: Value = serde_json::from_slice(event_data(&events[1]).unwrap()).unwrap();
        assert_eq!(usage["usage"]["completion_tokens"], 4);
        assert_eq!(usage["usage"]["total_tokens"], 7);
        assert_eq!(events[2], Bytes::from_static(DONE_EVENT));
    }
}
