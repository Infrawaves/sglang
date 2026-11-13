// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Bounded router-owned requests and fenced, idempotent pull assignments.
//! A committed assignment is never made available to a second prefill worker.
use crate::config::PullConfig;
use crate::server::error::ApiError;
use axum::body::Body;
use axum::http::{HeaderMap, Response, StatusCode};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Notify};

pub const MAX_OUTCOME_BYTES: usize = 4 * 1024 * 1024;
// The raw HTTP body is a JSON string in the control message. Each byte can
// expand to a six-byte JSON escape, in addition to the bounded identity fields.
pub const MAX_OUTCOME_WIRE_BYTES: usize = 6 * MAX_OUTCOME_BYTES + 16 * 1024;

#[derive(Clone, Debug, Serialize)]
pub struct Offer {
    pub status: &'static str,
    pub assignment_id: String,
    pub request_id: String,
    pub path: &'static str,
    pub body: serde_json::Value,
    pub headers: HashMap<String, String>,
    pub input_tokens: usize,
    pub expires_in_ms: u64,
}

#[derive(Deserialize)]
pub struct Acquire {
    pub worker_url: String,
    pub worker_epoch: String,
    pub dp_rank: u32,
    pub poll_id: String,
    pub max_tokens: usize,
    pub max_requests: usize,
    pub wait_ms: u64,
}

#[derive(Deserialize)]
pub struct Action {
    pub assignment_id: String,
    pub worker_epoch: String,
    pub worker_url: Option<String>,
}

#[derive(Deserialize)]
pub struct Outcome {
    #[serde(flatten)]
    pub action: Action,
    pub status_code: u16,
    pub body: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Offered,
    Committed,
    Completed,
    Cancelled,
    Expired,
}
impl Phase {
    fn name(self) -> &'static str {
        match self {
            Self::Offered => "offered",
            Self::Committed => "committed",
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
            Self::Expired => "expired",
        }
    }
    fn terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Cancelled | Self::Expired)
    }
}
struct AssignmentState {
    phase: Phase,
    offer: Option<Offer>,
    outcome: Option<(u16, String)>,
}
pub struct Assignment {
    pub id: String,
    pub worker_url: String,
    epoch: String,
    poll_id: String,
    tokens: usize,
    expires: Instant,
    created: Instant,
    state: Mutex<AssignmentState>,
    changed: Notify,
    queue_changed: Arc<Notify>,
}
impl Assignment {
    fn notify_changed(&self) {
        self.changed.notify_waiters();
        // Leaving Offered makes another queued request claimable by this P.
        // Wake queue waiters as well as this assignment's client/worker.
        self.queue_changed.notify_waiters();
    }
    fn state(&self) -> std::sync::MutexGuard<'_, AssignmentState> {
        let mut state = self.state.lock().unwrap();
        if state.phase == Phase::Offered && Instant::now() >= self.expires {
            state.phase = Phase::Expired;
            state.offer = None;
            self.notify_changed();
        }
        state
    }
    pub fn status(&self) -> &'static str {
        self.state().phase.name()
    }
    pub fn expired(&self) -> bool {
        self.state().phase == Phase::Expired
    }
    pub fn cancel(&self) {
        let mut state = self.state();
        if matches!(state.phase, Phase::Offered | Phase::Committed) {
            state.phase = Phase::Cancelled;
            state.offer = None;
            self.notify_changed();
        }
    }
    fn abandon(&self) {
        self.cancel();
        // Keep only the terminal status for idempotency once the client is gone.
        // In particular, an outcome can race with client cancellation before
        // response() consumes it; retaining every body would dwarf queue limits.
        self.state().outcome = None;
    }
    pub fn accept(&self) -> &'static str {
        let mut state = self.state();
        if state.phase == Phase::Offered && state.offer.is_some() {
            // The durable-in-process transition precedes both D dispatch and the ACK.
            state.phase = Phase::Committed;
            state.offer = None;
            self.notify_changed();
        }
        state.phase.name()
    }
    pub fn reject(&self) -> &'static str {
        let mut state = self.state();
        state.phase = match state.phase {
            Phase::Offered => Phase::Expired,
            Phase::Committed => Phase::Cancelled,
            other => other,
        };
        state.offer = None;
        self.notify_changed();
        state.phase.name()
    }
    pub fn outcome(&self, status: u16, body: String) -> &'static str {
        let mut state = self.state();
        match state.phase {
            Phase::Committed => {
                state.outcome = Some((status, body));
                state.phase = Phase::Completed;
            }
            // No execution was authorized; let the waiting client reoffer safely.
            Phase::Offered => state.phase = Phase::Expired,
            _ => {}
        }
        state.offer = None;
        self.notify_changed();
        state.phase.name()
    }
    pub async fn offer(&self) -> Option<Offer> {
        loop {
            let changed = self.changed.notified();
            {
                let state = self.state();
                if state.phase != Phase::Offered {
                    return None;
                }
                if let Some(mut offer) = state.offer.clone() {
                    offer.expires_in_ms = self
                        .expires
                        .saturating_duration_since(Instant::now())
                        .as_millis() as u64;
                    return Some(offer);
                }
            }
            let _ = tokio::time::timeout_at(self.expires.into(), changed).await;
        }
    }
    pub async fn publish_and_wait(
        &self,
        rid: &str,
        path: &'static str,
        body: &Bytes,
        headers: &HeaderMap,
    ) -> Result<(), ApiError> {
        if body.len() > 15 * 1024 * 1024 {
            return Err(ApiError::BadRequest(
                "pull offer payload exceeds 15 MiB".into(),
            ));
        }
        let body = serde_json::from_slice(body).map_err(|e| ApiError::BadRequest(e.to_string()))?;
        let headers = [
            "x-request-id",
            "venus-request-id",
            "traceparent",
            "tracestate",
            "x-sgl-decode-url",
        ]
        .into_iter()
        .filter_map(|name| {
            headers
                .get(name)
                .and_then(|v| v.to_str().ok())
                .filter(|v| v.len() <= 4096)
                .map(|v| (name.to_owned(), v.to_owned()))
        })
        .collect();
        {
            let mut state = self.state();
            if state.phase == Phase::Offered {
                state.offer = Some(Offer {
                    status: "offer",
                    assignment_id: self.id.clone(),
                    request_id: rid.to_owned(),
                    path,
                    body,
                    headers,
                    input_tokens: self.tokens,
                    expires_in_ms: 0,
                });
                self.changed.notify_waiters();
            }
        }
        loop {
            let changed = self.changed.notified();
            match self.state().phase {
                Phase::Committed | Phase::Completed => return Ok(()),
                Phase::Offered => {}
                _ => return Err(ApiError::PrefillFailed { status: None }),
            }
            let _ = tokio::time::timeout_at(self.expires.into(), changed).await;
        }
    }
    pub async fn response(&self) -> Result<Response<Body>, ApiError> {
        loop {
            let changed = self.changed.notified();
            {
                let mut state = self.state();
                if let Some((status, body)) = state.outcome.take() {
                    return Ok(Response::builder()
                        .status(status)
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap());
                }
                if state.phase.terminal() {
                    return Err(ApiError::PrefillFailed { status: None });
                }
            }
            changed.await;
        }
    }
}

struct Pending {
    eligible: HashSet<String>,
    tokens: usize,
    bytes: usize,
    claims: mpsc::UnboundedSender<Arc<Assignment>>,
    current: Option<Arc<Assignment>>,
}
#[derive(Default)]
struct QueueState {
    pending: HashMap<String, Pending>,
    fifo: VecDeque<String>,
    assignments: HashMap<String, Arc<Assignment>>,
    // Retired epochs never become current again. Rotation count is bounded;
    // after 128 agent restarts this experimental single-router session must restart.
    epochs: HashMap<String, (String, HashSet<String>)>,
    bytes: usize,
    tokens: usize,
}
pub struct PullQueue {
    limits: PullConfig,
    state: Mutex<QueueState>,
    changed: Arc<Notify>,
}
impl PullQueue {
    pub fn new(limits: PullConfig) -> Arc<Self> {
        Arc::new(Self {
            limits,
            state: Mutex::new(QueueState::default()),
            changed: Arc::new(Notify::new()),
        })
    }
    pub fn enqueue(
        self: &Arc<Self>,
        eligible: HashSet<String>,
        tokens: usize,
        bytes: usize,
    ) -> Result<(RequestLease, mpsc::UnboundedReceiver<Arc<Assignment>>), ApiError> {
        let mut state = self.state.lock().unwrap();
        if state.pending.len() >= self.limits.max_requests
            || bytes > self.limits.max_bytes.saturating_sub(state.bytes)
            || tokens > self.limits.max_tokens.saturating_sub(state.tokens)
        {
            return Err(ApiError::PullQueueFull);
        }
        let id = uuid::Uuid::new_v4().to_string();
        let (claims, receiver) = mpsc::unbounded_channel();
        state.pending.insert(
            id.clone(),
            Pending {
                eligible,
                tokens,
                bytes,
                claims,
                current: None,
            },
        );
        state.fifo.push_back(id.clone());
        state.bytes += bytes;
        state.tokens += tokens;
        self.changed.notify_waiters();
        Ok((
            RequestLease {
                queue: self.clone(),
                id,
            },
            receiver,
        ))
    }
    pub fn requeue(&self, lease: &RequestLease) {
        let mut state = self.state.lock().unwrap();
        if let Some(p) = state.pending.get_mut(&lease.id) {
            if p.current.as_ref().is_some_and(|a| a.expired()) {
                // Its original queue position remains reserved while offered.
                // Only an uncommitted attempt can return to the waiting set.
                p.current = None;
                self.changed.notify_waiters();
            }
        }
    }
    fn try_acquire(&self, input: &Acquire) -> Result<Option<Arc<Assignment>>, ApiError> {
        let mut state = self.state.lock().unwrap();
        let now = Instant::now();
        state.assignments.retain(|_, a| {
            !a.state().phase.terminal() || now.duration_since(a.created) < Duration::from_secs(600)
        });
        if let Some((current, retired)) = state.epochs.get_mut(&input.worker_url) {
            if current != &input.worker_epoch {
                if retired.contains(&input.worker_epoch) || retired.len() >= 128 {
                    return Err(ApiError::BadRequest(
                        "stale worker epoch; restart the agent/router session".into(),
                    ));
                }
                retired.insert(std::mem::replace(current, input.worker_epoch.clone()));
                for a in state
                    .assignments
                    .values()
                    .filter(|a| a.worker_url == input.worker_url)
                {
                    // Losing an agent session revokes uncommitted offers so
                    // their clients can requeue. Committed work stays cancelled:
                    // an epoch change cannot prove that execution has stopped.
                    a.reject();
                }
            }
        } else {
            state.epochs.insert(
                input.worker_url.clone(),
                (input.worker_epoch.clone(), HashSet::new()),
            );
        }
        if let Some(a) = state.assignments.values().find(|a| {
            a.worker_url == input.worker_url
                && a.epoch == input.worker_epoch
                && a.poll_id == input.poll_id
        }) {
            return Ok(Some(a.clone()));
        }
        if state
            .assignments
            .values()
            .any(|a| a.worker_url == input.worker_url && a.state().phase == Phase::Offered)
        {
            return Ok(None);
        }
        // Bound outcome tombstones independently of live requests.
        if state.assignments.len() >= self.limits.max_requests.saturating_mul(5) {
            let oldest = state
                .assignments
                .iter()
                .filter(|(_, a)| a.state().phase.terminal())
                .min_by_key(|(_, a)| a.created)
                .map(|(id, _)| id.clone());
            if let Some(id) = oldest {
                state.assignments.remove(&id);
            } else {
                return Ok(None);
            }
        }
        let position = state.fifo.iter().position(|id| {
            state.pending.get(id).is_some_and(|p| {
                p.current.is_none()
                    && p.tokens <= input.max_tokens
                    && p.eligible.contains(&input.worker_url)
            })
        });
        let Some(position) = position else {
            return Ok(None);
        };
        let id = state.fifo[position].clone();
        let p = state.pending.get_mut(&id).unwrap();
        let assignment = Arc::new(Assignment {
            id: uuid::Uuid::new_v4().to_string(),
            worker_url: input.worker_url.clone(),
            epoch: input.worker_epoch.clone(),
            poll_id: input.poll_id.clone(),
            tokens: p.tokens,
            created: now,
            expires: now + Duration::from_millis(self.limits.offer_timeout_ms),
            state: Mutex::new(AssignmentState {
                phase: Phase::Offered,
                offer: None,
                outcome: None,
            }),
            changed: Notify::new(),
            queue_changed: self.changed.clone(),
        });
        p.current = Some(assignment.clone());
        let _ = p.claims.send(assignment.clone());
        state
            .assignments
            .insert(assignment.id.clone(), assignment.clone());
        Ok(Some(assignment))
    }
    pub async fn acquire(&self, input: &Acquire) -> Result<Option<Offer>, ApiError> {
        let deadline = Instant::now() + Duration::from_millis(input.wait_ms.min(30_000));
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            // Register before checking the queue so a concurrent accept cannot
            // fall between observing Offered and beginning the long poll.
            changed.as_mut().enable();
            if let Some(a) = self.try_acquire(input)? {
                return Ok(a.offer().await);
            }
            if tokio::time::timeout_at(deadline.into(), changed)
                .await
                .is_err()
            {
                // Check once at the boundary: work becoming claimable at the
                // deadline should not force an additional empty poll/backoff.
                return match self.try_acquire(input)? {
                    Some(a) => Ok(a.offer().await),
                    None => Ok(None),
                };
            }
        }
    }
    pub fn assignment(&self, input: &Action) -> Result<Option<Arc<Assignment>>, ApiError> {
        let state = self.state.lock().unwrap();
        let Some(a) = state.assignments.get(&input.assignment_id) else {
            return Ok(None);
        };
        if a.epoch != input.worker_epoch
            || input
                .worker_url
                .as_ref()
                .is_some_and(|url| url.trim_end_matches('/') != a.worker_url)
        {
            return Err(ApiError::BadRequest(
                "assignment belongs to another worker epoch".into(),
            ));
        }
        Ok(Some(a.clone()))
    }
}

pub struct RequestLease {
    queue: Arc<PullQueue>,
    id: String,
}
impl Drop for RequestLease {
    fn drop(&mut self) {
        let mut state = self.queue.state.lock().unwrap();
        if let Some(p) = state.pending.remove(&self.id) {
            state.bytes -= p.bytes;
            state.tokens -= p.tokens;
            if let Some(a) = p.current {
                a.abandon();
            }
        }
        state.fifo.retain(|id| id != &self.id);
        self.queue.changed.notify_waiters();
    }
}

pub fn validate_acquire(input: &Acquire) -> Result<(), ApiError> {
    if input.dp_rank != 0
        || input.max_requests != 1
        || input.max_tokens == 0
        || input.worker_epoch.is_empty()
        || input.worker_epoch.len() > 128
        || input.poll_id.is_empty()
        || input.poll_id.len() > 128
    {
        return Err(ApiError::BadRequest("pull requires dp_rank=0, max_requests=1, positive max_tokens and bounded epoch/poll IDs".into()));
    }
    Ok(())
}

pub fn validate_outcome(input: &Outcome) -> Result<(), ApiError> {
    if StatusCode::from_u16(input.status_code).is_err()
        || input.status_code < 200
        || input.body.len() > MAX_OUTCOME_BYTES
    {
        return Err(ApiError::BadRequest(
            "invalid pull outcome status or body size".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::poll;
    fn acquire(worker: &str, epoch: &str, poll: &str, max_tokens: usize) -> Acquire {
        Acquire {
            worker_url: worker.into(),
            worker_epoch: epoch.into(),
            dp_rank: 0,
            poll_id: poll.into(),
            max_tokens,
            max_requests: 1,
            wait_ms: 0,
        }
    }
    #[test]
    fn queued_requests_respect_bytes_tokens_and_reclaim_on_cancellation() {
        let queue = PullQueue::new(PullConfig {
            max_requests: 10,
            max_bytes: 100,
            max_tokens: 20,
            ..Default::default()
        });
        let (first, _) = queue.enqueue(HashSet::from(["p".into()]), 10, 60).unwrap();
        assert!(queue.enqueue(HashSet::new(), 1, 41).is_err());
        assert!(queue.enqueue(HashSet::new(), 11, 1).is_err());
        drop(first);
        assert!(queue.enqueue(HashSet::new(), 20, 100).is_ok());
    }
    #[test]
    fn claimant_is_pinned_and_another_poll_cannot_steal_an_offer() {
        let queue = PullQueue::new(PullConfig::default());
        let (_lease, _) = queue.enqueue(HashSet::from(["p".into()]), 10, 10).unwrap();
        assert!(queue
            .try_acquire(&acquire("q", "e", "1", 10))
            .unwrap()
            .is_none());
        assert!(queue
            .try_acquire(&acquire("p", "e", "1", 9))
            .unwrap()
            .is_none());
        let a = queue
            .try_acquire(&acquire("p", "e", "1", 10))
            .unwrap()
            .unwrap();
        assert!(Arc::ptr_eq(
            &a,
            &queue
                .try_acquire(&acquire("p", "e", "1", 10))
                .unwrap()
                .unwrap()
        ));
        assert!(queue
            .try_acquire(&acquire("p", "e", "2", 10))
            .unwrap()
            .is_none());
    }
    #[test]
    fn epoch_rotation_requeues_unpublished_work_after_decode_rejection() {
        let queue = PullQueue::new(PullConfig::default());
        let (lease, mut claims) = queue.enqueue(HashSet::from(["p".into()]), 10, 10).unwrap();
        let a = queue
            .try_acquire(&acquire("p", "old", "1", 10))
            .unwrap()
            .unwrap();
        queue.try_acquire(&acquire("p", "new", "2", 10)).unwrap();
        assert_eq!(a.accept(), "expired");
        assert!(queue.try_acquire(&acquire("p", "old", "3", 10)).is_err());
        // Rotation can happen during D selection, before an offer is published.
        // A subsequent D rejection must still permit clearing Pending.current.
        assert_eq!(a.reject(), "expired");
        queue.requeue(&lease);
        let retried = queue
            .try_acquire(&acquire("p", "new", "4", 10))
            .unwrap()
            .expect("uncommitted work must remain claimable after rotation");
        assert_ne!(retried.id, a.id);
        assert_eq!(claims.try_recv().unwrap().id, a.id);
        assert_eq!(claims.try_recv().unwrap().id, retried.id);
        assert!(claims.try_recv().is_err());
    }

    #[tokio::test]
    async fn epoch_rotation_cancels_committed_work_without_requeue() {
        let queue = PullQueue::new(PullConfig::default());
        let (lease, _) = queue.enqueue(HashSet::from(["p".into()]), 10, 10).unwrap();
        let a = queue
            .try_acquire(&acquire("p", "old", "1", 10))
            .unwrap()
            .unwrap();
        let body = Bytes::from_static(b"{}");
        let headers = HeaderMap::new();
        let publish = a.publish_and_wait("rid", "/generate", &body, &headers);
        tokio::pin!(publish);
        assert!(poll!(publish.as_mut()).is_pending());
        assert_eq!(a.accept(), "committed");

        assert!(queue
            .try_acquire(&acquire("p", "new", "2", 10))
            .unwrap()
            .is_none());
        assert_eq!(a.status(), "cancelled");
        assert_eq!(a.accept(), "cancelled");
        assert_eq!(a.reject(), "cancelled");
        queue.requeue(&lease);
        assert!(queue
            .try_acquire(&acquire("p", "new", "3", 10))
            .unwrap()
            .is_none());
    }
    #[test]
    fn stale_actions_cannot_commit_another_workers_assignment() {
        let queue = PullQueue::new(PullConfig::default());
        let (_lease, _) = queue.enqueue(HashSet::from(["p".into()]), 10, 10).unwrap();
        let a = queue
            .try_acquire(&acquire("p", "e", "1", 10))
            .unwrap()
            .unwrap();
        assert!(queue
            .assignment(&Action {
                assignment_id: a.id.clone(),
                worker_epoch: "other".into(),
                worker_url: None
            })
            .is_err());
        assert!(queue
            .assignment(&Action {
                assignment_id: a.id.clone(),
                worker_epoch: "e".into(),
                worker_url: Some("q".into())
            })
            .is_err());
        assert_eq!(a.status(), "offered");
    }

    #[test]
    fn client_cancellation_drops_a_completed_but_unread_outcome() {
        let queue = PullQueue::new(PullConfig::default());
        let (lease, _) = queue.enqueue(HashSet::from(["p".into()]), 10, 10).unwrap();
        let assignment = queue
            .try_acquire(&acquire("p", "e", "1", 10))
            .unwrap()
            .unwrap();
        assignment.state().phase = Phase::Committed;
        assert_eq!(
            assignment.outcome(200, "x".repeat(1024 * 1024)),
            "completed"
        );
        drop(lease);
        // Duplicate outcomes still see the final state, without retaining the
        // potentially large response after its sole consumer disconnected.
        assert_eq!(assignment.status(), "completed");
        assert!(assignment.state().outcome.is_none());
        assert_eq!(assignment.outcome(200, "retry".into()), "completed");
        assert!(assignment.state().outcome.is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn accepting_one_offer_wakes_the_next_acquire_without_new_arrivals() {
        let queue = PullQueue::new(PullConfig::default());
        let (_first, mut first_claims) =
            queue.enqueue(HashSet::from(["p".into()]), 10, 10).unwrap();
        let (_second, mut second_claims) =
            queue.enqueue(HashSet::from(["p".into()]), 10, 10).unwrap();
        let first = queue
            .try_acquire(&acquire("p", "e", "first", 10))
            .unwrap()
            .unwrap();
        assert_eq!(first_claims.recv().await.unwrap().id, first.id);
        let body = Bytes::from_static(b"{}");
        let headers = HeaderMap::new();
        let publish_first = first.publish_and_wait("rid-first", "/generate", &body, &headers);
        tokio::pin!(publish_first);
        assert!(poll!(publish_first.as_mut()).is_pending());

        let mut input = acquire("p", "e", "second", 10);
        input.wait_ms = 10_000;
        let waiting = queue.acquire(&input);
        tokio::pin!(waiting);
        assert!(poll!(waiting.as_mut()).is_pending());
        assert!(second_claims.try_recv().is_err());
        assert_eq!(first.accept(), "committed");

        // The next poll must claim B at the same instant, without advancing the
        // long-poll timer, adding requests, or changing D's state.
        assert!(poll!(waiting.as_mut()).is_pending());
        let second = second_claims
            .try_recv()
            .expect("accept must wake the queue");
        let publish_second = second.publish_and_wait("rid-second", "/generate", &body, &headers);
        tokio::pin!(publish_second);
        assert!(poll!(publish_second.as_mut()).is_pending());
        let offer = match poll!(waiting.as_mut()) {
            std::task::Poll::Ready(Ok(Some(offer))) => offer,
            _ => panic!("second offer should be immediately available"),
        };
        assert_eq!(offer.request_id, "rid-second");
    }

    #[test]
    fn rejected_request_keeps_its_original_fifo_position_without_duplicates() {
        let queue = PullQueue::new(PullConfig::default());
        let (first, _) = queue.enqueue(HashSet::from(["p".into()]), 10, 10).unwrap();
        let (_second, mut second_claims) =
            queue.enqueue(HashSet::from(["p".into()]), 10, 10).unwrap();
        let a = queue
            .try_acquire(&acquire("p", "e", "first", 10))
            .unwrap()
            .unwrap();
        a.reject();
        queue.requeue(&first);
        queue.requeue(&first);
        let retried = queue
            .try_acquire(&acquire("p", "e", "retry", 10))
            .unwrap()
            .unwrap();
        assert!(
            second_claims.try_recv().is_err(),
            "A must precede B after rejection"
        );
        assert_ne!(retried.id, a.id);
        retried.state().phase = Phase::Committed;
        queue.requeue(&first);
        let b = queue
            .try_acquire(&acquire("p", "e", "next", 10))
            .unwrap()
            .unwrap();
        assert_eq!(second_claims.try_recv().unwrap().id, b.id);
        assert_eq!(queue.state.lock().unwrap().fifo.len(), 2);
    }
}
