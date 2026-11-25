// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Bounded router-owned requests and fenced, idempotent pull assignments.
//! A committed assignment is never made available to a second prefill worker.
use crate::config::PullConfig;
use crate::health::circuit_breaker::{CircuitBreaker, OwnedCircuitPermit};
use crate::server::error::ApiError;
use axum::body::Body;
use axum::http::{HeaderMap, Response, StatusCode};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Notify};

pub const MAX_OUTCOME_BYTES: usize = 4 * 1024 * 1024;
/// Largest prefill body an offer carries; the agent caps offers at 16 MiB.
pub const MAX_OFFER_BODY_BYTES: usize = 15 * 1024 * 1024;
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
    /// Uncommitted offers this worker may hold at once: its concurrent claims.
    #[serde(default = "one")]
    pub max_offered: usize,
    pub wait_ms: u64,
}

fn one() -> usize {
    1
}

/// Upper bound on `Acquire::max_offered`, so one agent cannot pin the queue.
pub const MAX_OFFERED_PER_WORKER: usize = 16;
/// Fitting requests one claim compares for prefix affinity.
const AFFINITY_SCAN: usize = 16;

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
/// Cumulative pull counters, rendered on `/metrics` scrape.
#[derive(Default)]
struct PullCounters {
    enqueued: AtomicU64,
    queue_full: AtomicU64,
    offered: AtomicU64,
    committed: AtomicU64,
    completed: AtomicU64,
    cancelled: AtomicU64,
    expired: AtomicU64,
    /// Enqueue to commit, across every attempt of the request.
    queue_wait_us: AtomicU64,
    /// Assignment creation to commit: one claim's control-plane round trips.
    claim_us: AtomicU64,
}

fn micros(elapsed: Duration) -> u64 {
    u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX)
}

struct AssignmentState {
    phase: Phase,
    offer: Option<Offer>,
    prefill_permit: Option<OwnedCircuitPermit>,
    outcome: Option<(u16, String)>,
    /// Set once the client's response ended while still committed. The
    /// background prefill task still needs its outcome before this deadline.
    orphaned_until: Option<Instant>,
}
impl AssignmentState {
    /// Preserve the background task's result after the client no longer needs
    /// the body, including a failure carried by a successful SSE response.
    fn compact_outcome(&mut self) {
        if let Some((status, body)) = &mut self.outcome {
            *body = if (200..300).contains(status)
                && crate::proxy::sse::has_error_event(body.as_bytes())
            {
                "data: {\"error\":{}}\n\n".into()
            } else {
                String::new()
            };
        }
    }
}
pub struct Assignment {
    pub id: String,
    pub worker_url: String,
    epoch: String,
    poll_id: String,
    tokens: usize,
    expires: Instant,
    created: Instant,
    enqueued: Instant,
    /// The queued request (lease id) this assignment serves.
    request: String,
    hold: Arc<QuotaHold>,
    state: Mutex<AssignmentState>,
    changed: Notify,
    queue_changed: Arc<Notify>,
    counters: Arc<PullCounters>,
    /// The queue's terminal log: each assignment appends itself once, on its
    /// first terminal transition, so tombstones are evicted in that order
    /// instead of by scanning every assignment under the queue lock.
    retired: Arc<Mutex<VecDeque<(Instant, String)>>>,
}
impl Assignment {
    fn notify_changed(&self) {
        self.changed.notify_waiters();
        // Leaving Offered makes another queued request claimable by this P.
        // Wake queue waiters as well as this assignment's client/worker.
        self.queue_changed.notify_waiters();
    }
    /// The only writer of `phase` outside tests, so each change counts once.
    fn transition(&self, state: &mut AssignmentState, to: Phase) {
        if state.phase == to {
            return;
        }
        if to.terminal() && !state.phase.terminal() {
            drop(state.prefill_permit.take());
            // A leaf lock: nothing holding it takes an assignment or queue lock.
            self.retired
                .lock()
                .unwrap()
                .push_back((Instant::now(), self.id.clone()));
        }
        state.phase = to;
        let c = &self.counters;
        match to {
            Phase::Committed => {
                self.hold.release();
                c.committed.fetch_add(1, Ordering::Relaxed);
                c.queue_wait_us
                    .fetch_add(micros(self.enqueued.elapsed()), Ordering::Relaxed);
                c.claim_us
                    .fetch_add(micros(self.created.elapsed()), Ordering::Relaxed);
            }
            Phase::Completed => {
                c.completed.fetch_add(1, Ordering::Relaxed);
            }
            Phase::Cancelled => {
                c.cancelled.fetch_add(1, Ordering::Relaxed);
            }
            Phase::Expired => {
                c.expired.fetch_add(1, Ordering::Relaxed);
            }
            Phase::Offered => {}
        }
    }
    fn state(&self) -> std::sync::MutexGuard<'_, AssignmentState> {
        let mut state = self.state.lock().unwrap();
        let now = Instant::now();
        if state.phase == Phase::Offered && now >= self.expires {
            self.transition(&mut state, Phase::Expired);
            state.offer = None;
            self.notify_changed();
        }
        self.expire_orphan_locked(&mut state, now);
        state
    }
    fn expire_orphan(&self, now: Instant) {
        let mut state = self.state.lock().unwrap();
        self.expire_orphan_locked(&mut state, now);
    }
    fn expire_orphan_locked(&self, state: &mut AssignmentState, now: Instant) {
        if state.phase == Phase::Committed && state.orphaned_until.is_some_and(|at| now >= at) {
            self.transition(state, Phase::Cancelled);
            self.notify_changed();
        }
    }
    pub fn status(&self) -> &'static str {
        self.state().phase.name()
    }
    pub fn expired(&self) -> bool {
        self.state().phase == Phase::Expired
    }
    /// Attach admission before publishing the offer. P may finish after D's
    /// response ends, so the assignment owns the permit until its terminal state.
    pub fn admit_prefill(&self, breaker: &Arc<CircuitBreaker>) -> bool {
        let mut state = self.state();
        if state.phase != Phase::Offered || state.prefill_permit.is_some() {
            return false;
        }
        state.prefill_permit = breaker.acquire_owned();
        state.prefill_permit.is_some()
    }
    pub fn cancel(&self) {
        let mut state = self.state();
        if matches!(state.phase, Phase::Offered | Phase::Committed) {
            self.transition(&mut state, Phase::Cancelled);
            state.offer = None;
            self.notify_changed();
        }
    }
    fn abandon(&self) {
        self.cancel();
        // The client can leave after P reports but before response() consumes
        // it. Preserve its classification without retaining the full body.
        self.state().compact_outcome();
    }
    /// Production commits through [`PullQueue::accept`], which also leaves the FIFO.
    fn accept(&self) -> &'static str {
        let mut state = self.state();
        if state.phase == Phase::Offered && state.offer.is_some() {
            // The durable-in-process transition precedes both D dispatch and the ACK.
            self.transition(&mut state, Phase::Committed);
            state.offer = None;
            self.notify_changed();
        }
        state.phase.name()
    }
    pub fn reject(&self) -> &'static str {
        let mut state = self.state();
        let to = match state.phase {
            Phase::Offered => Phase::Expired,
            Phase::Committed => Phase::Cancelled,
            other => other,
        };
        self.transition(&mut state, to);
        state.offer = None;
        self.notify_changed();
        state.phase.name()
    }
    pub fn outcome(&self, status: u16, body: String) -> &'static str {
        let mut state = self.state();
        match state.phase {
            Phase::Committed => {
                // Count the first committed outcome even when D's response has
                // already ended. Resolve the breaker before releasing the probe.
                if let Some(permit) = &state.prefill_permit {
                    crate::proxy::record_upstream_status(permit.breaker(), status);
                }
                state.outcome = Some((status, body));
                if state.orphaned_until.is_some() {
                    state.compact_outcome();
                }
                self.transition(&mut state, Phase::Completed);
            }
            // No execution was authorized; let the waiting client reoffer safely.
            Phase::Offered => self.transition(&mut state, Phase::Expired),
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
        if body.len() > MAX_OFFER_BODY_BYTES {
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

/// Why a poll claimed nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Unclaimed {
    Nothing,
    /// The breaker is open or already has a recovery probe.
    BreakerUnavailable,
    /// A starved request this worker cannot fit yet holds it back.
    Held,
}

/// What one acquire long poll ended with.
pub enum Polled {
    Offer(Offer),
    Empty,
    /// Held back for a starved request: the worker should ask its scheduler
    /// for fresher room and poll again soon, not wait out a long poll.
    Held,
}

/// What `--pull-max-requests/bytes/tokens` bound: requests not yet committed.
/// Committed work is bounded by its decode worker's admission instead, so a
/// long decode no longer keeps new requests out with 503s.
#[derive(Default)]
struct Quota {
    requests: AtomicUsize,
    bytes: AtomicUsize,
    tokens: AtomicUsize,
}

/// One request's share of [`Quota`], returned exactly once: at commit, or when
/// the request ends uncommitted.
struct QuotaHold {
    quota: Arc<Quota>,
    tokens: usize,
    bytes: usize,
    held: AtomicBool,
}

impl QuotaHold {
    fn release(&self) {
        if self.held.swap(false, Ordering::Relaxed) {
            self.quota.requests.fetch_sub(1, Ordering::Relaxed);
            self.quota.bytes.fetch_sub(self.bytes, Ordering::Relaxed);
            self.quota.tokens.fetch_sub(self.tokens, Ordering::Relaxed);
        }
    }
}

struct Pending {
    eligible: HashSet<String>,
    tokens: usize,
    hold: Arc<QuotaHold>,
    claims: mpsc::UnboundedSender<Arc<Assignment>>,
    current: Option<Arc<Assignment>>,
    enqueued: Instant,
    /// Prompt tokens each eligible worker held cached at enqueue, by URL.
    matched: HashMap<String, usize>,
}
#[derive(Default)]
struct QueueState {
    /// Set once shutdown stops new intake; held requests still finish.
    closing: bool,
    /// Each worker's assignments that were live when last looked at; terminal
    /// ones are dropped as they are seen. Bounded by in-flight work per worker,
    /// unlike `assignments`, which also keeps tombstones for idempotent replies.
    by_worker: HashMap<String, Vec<Arc<Assignment>>>,
    pending: HashMap<String, Pending>,
    fifo: VecDeque<String>,
    assignments: HashMap<String, Arc<Assignment>>,
    // Retired epochs never become current again. Rotation count is bounded;
    // after 128 agent restarts this experimental single-router session must restart.
    epochs: HashMap<String, (String, HashSet<String>)>,
    /// Largest grant each worker has reported: a request above it may never fit.
    largest_grant: HashMap<String, usize>,
    /// Committed assignments whose client finished, by outcome deadline; the
    /// grace is constant, so pushing to the back keeps it sorted.
    orphans: VecDeque<(Instant, Arc<Assignment>)>,
    /// Last acquire per worker. Discovered workers come and go with their URLs,
    /// so per-worker state is dropped once a worker stops polling.
    last_poll: HashMap<String, Instant>,
}
/// An agent long-polls at least every 30 s, so ten idle minutes means it is gone.
const WORKER_IDLE_FORGET: Duration = Duration::from_secs(600);

/// `url`'s live assignments, pruning the terminal ones first.
fn live<'a>(
    by_worker: &'a mut HashMap<String, Vec<Arc<Assignment>>>,
    url: &str,
) -> &'a [Arc<Assignment>] {
    match by_worker.get_mut(url) {
        Some(list) => {
            list.retain(|a| !a.state().phase.terminal());
            list
        }
        None => &[],
    }
}

/// How long a terminal assignment answers status/outcome before it is dropped.
const TOMBSTONE_TTL: Duration = Duration::from_secs(600);

impl QueueState {
    /// Drops the epoch fence and grant history of workers that stopped polling
    /// and hold no live assignment, so worker churn under discovery stays bounded.
    fn forget_idle_workers(&mut self, now: Instant) {
        let idle: Vec<String> = self
            .last_poll
            .iter()
            .filter(|(_, at)| now.duration_since(**at) >= WORKER_IDLE_FORGET)
            .map(|(url, _)| url.clone())
            .collect();
        for url in idle {
            if live(&mut self.by_worker, &url).is_empty() {
                self.by_worker.remove(&url);
                self.last_poll.remove(&url);
                self.epochs.remove(&url);
                self.largest_grant.remove(&url);
            }
        }
    }
}

pub struct PullQueue {
    limits: PullConfig,
    state: Mutex<QueueState>,
    changed: Arc<Notify>,
    counters: Arc<PullCounters>,
    quota: Arc<Quota>,
    retired: Arc<Mutex<VecDeque<(Instant, String)>>>,
    /// Per model: when the worker-fleet fingerprint was computed, and its value.
    fleet_cache: Mutex<HashMap<String, (Instant, u64)>>,
}
impl PullQueue {
    pub fn new(limits: PullConfig) -> Arc<Self> {
        Arc::new(Self {
            limits,
            state: Mutex::new(QueueState::default()),
            changed: Arc::new(Notify::new()),
            counters: Arc::new(PullCounters::default()),
            quota: Arc::new(Quota::default()),
            retired: Arc::new(Mutex::new(VecDeque::new())),
            fleet_cache: Mutex::new(HashMap::new()),
        })
    }

    /// Stops new requests at the door. Already queued or claimed requests keep
    /// being claimed and served, which needs the control plane: the server must
    /// keep accepting connections until [`Self::drained`] resolves.
    pub fn close(&self) {
        // Under the queue lock, as enqueue's check and drained's emptiness test
        // are: otherwise an enqueue past its check could land after drained()
        // saw an empty queue and the listener closed.
        self.state.lock().unwrap().closing = true;
    }

    /// Resolves once no request is held: every queued, claimed or decoding
    /// request has answered its client or given up.
    pub async fn drained(&self) {
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if self.state.lock().unwrap().pending.is_empty() {
                return;
            }
            changed.await;
        }
    }

    /// The model's worker-fleet fingerprint, recomputed at most once per `ttl`
    /// and shared by every queued request of that model.
    pub fn fleet_generation(
        &self,
        model: &str,
        ttl: Duration,
        compute: impl FnOnce() -> u64,
    ) -> u64 {
        let now = Instant::now();
        let mut cache = self.fleet_cache.lock().unwrap();
        if let Some(&(at, generation)) = cache.get(model) {
            if now.duration_since(at) < ttl {
                return generation;
            }
        }
        let generation = compute();
        cache.insert(model.to_owned(), (now, generation));
        generation
    }

    /// Drops tombstones past their TTL, then the oldest while there are `cap`.
    /// Live assignments are not counted: committed ones are bounded by the
    /// prefills' own admission, and counting them stopped every offer once
    /// enough requests were in flight.
    fn evict_tombstones(&self, state: &mut QueueState, now: Instant, cap: usize) {
        // `retired` holds exactly the tombstones still in `assignments`.
        let mut retired = self.retired.lock().unwrap();
        while let Some((at, id)) = retired.front() {
            if now.duration_since(*at) < TOMBSTONE_TTL && retired.len() < cap {
                break;
            }
            state.assignments.remove(id);
            retired.pop_front();
        }
    }

    /// Cancels committed assignments whose client finished and whose outcome
    /// is overdue. Run on every poll from any worker, so it does not wait for
    /// the claimant (possibly gone) to read its status.
    fn expire_orphans(state: &mut QueueState, now: Instant) {
        while let Some((at, _)) = state.orphans.front() {
            if *at > now {
                break;
            }
            let (_, a) = state.orphans.pop_front().unwrap();
            a.expire_orphan(now);
        }
    }

    /// Prometheus text for the pull queue: live occupancy plus cumulative counters.
    pub fn render_metrics(&self) -> String {
        let (waiting, claimed, bytes, tokens) = {
            let state = self.state.lock().unwrap();
            let waiting = state
                .pending
                .values()
                .filter(|p| p.current.is_none())
                .count();
            (
                waiting,
                state.pending.len() - waiting,
                self.quota.bytes.load(Ordering::Relaxed),
                self.quota.tokens.load(Ordering::Relaxed),
            )
        };
        let c = &self.counters;
        let read = |v: &AtomicU64| v.load(Ordering::Relaxed);
        let mut out = String::new();
        let mut metric = |name: &str, kind: &str, help: &str, samples: &[(&str, String)]| {
            out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} {kind}\n"));
            for (labels, value) in samples {
                out.push_str(&format!("{name}{labels} {value}\n"));
            }
        };
        metric(
            "sgl_router_pull_requests",
            "gauge",
            "Requests the pull queue holds. state=waiting: unclaimed; state=claimed: offered, executing, or in decode until the response ends.",
            &[
                ("{state=\"waiting\"}", waiting.to_string()),
                ("{state=\"claimed\"}", claimed.to_string()),
            ],
        );
        metric(
            "sgl_router_pull_held_bytes",
            "gauge",
            "Request body bytes counted against --pull-max-bytes: queued and offered requests, not committed ones.",
            &[("", bytes.to_string())],
        );
        metric(
            "sgl_router_pull_held_tokens",
            "gauge",
            "Prompt tokens counted against --pull-max-tokens: queued and offered requests, not committed ones.",
            &[("", tokens.to_string())],
        );
        metric(
            "sgl_router_pull_enqueued_total",
            "counter",
            "Requests admitted to the pull queue.",
            &[("", read(&c.enqueued).to_string())],
        );
        metric(
            "sgl_router_pull_queue_full_total",
            "counter",
            "Requests refused with 503 pull_queue_full.",
            &[("", read(&c.queue_full).to_string())],
        );
        metric(
            "sgl_router_pull_assignments_total",
            "counter",
            "Assignment transitions. phase=offered: claims; expired: uncommitted offers that timed out or were rejected, and may be reoffered; cancelled: committed or offered work revoked by the client or an epoch change.",
            &[
                ("{phase=\"offered\"}", read(&c.offered).to_string()),
                ("{phase=\"committed\"}", read(&c.committed).to_string()),
                ("{phase=\"completed\"}", read(&c.completed).to_string()),
                ("{phase=\"cancelled\"}", read(&c.cancelled).to_string()),
                ("{phase=\"expired\"}", read(&c.expired).to_string()),
            ],
        );
        let seconds = |v: &AtomicU64| format!("{:.6}", read(v) as f64 / 1e6);
        metric(
            "sgl_router_pull_queue_wait_seconds_sum",
            "counter",
            "Enqueue-to-commit time summed over committed requests; divide by assignments_total{phase=\"committed\"}.",
            &[("", seconds(&c.queue_wait_us))],
        );
        metric(
            "sgl_router_pull_claim_seconds_sum",
            "counter",
            "Offer-to-commit time (bind and accept round trips) summed over committed requests.",
            &[("", seconds(&c.claim_us))],
        );
        out
    }
    pub fn enqueue(
        self: &Arc<Self>,
        eligible: HashSet<String>,
        matched: HashMap<String, usize>,
        tokens: usize,
        bytes: usize,
    ) -> Result<(RequestLease, mpsc::UnboundedReceiver<Arc<Assignment>>), ApiError> {
        let mut state = self.state.lock().unwrap();
        if state.closing {
            return Err(ApiError::PullDraining);
        }
        // Checked and added under the queue lock; releases may run without it,
        // which only makes more room.
        let q = &self.quota;
        if q.requests.load(Ordering::Relaxed) >= self.limits.max_requests
            || bytes
                > self
                    .limits
                    .max_bytes
                    .saturating_sub(q.bytes.load(Ordering::Relaxed))
            || tokens
                > self
                    .limits
                    .max_tokens
                    .saturating_sub(q.tokens.load(Ordering::Relaxed))
        {
            self.counters.queue_full.fetch_add(1, Ordering::Relaxed);
            return Err(ApiError::PullQueueFull);
        }
        let id = uuid::Uuid::new_v4().to_string();
        let (claims, receiver) = mpsc::unbounded_channel();
        state.pending.insert(
            id.clone(),
            Pending {
                eligible,
                tokens,
                hold: Arc::new(QuotaHold {
                    quota: self.quota.clone(),
                    tokens,
                    bytes,
                    held: AtomicBool::new(true),
                }),
                claims,
                current: None,
                enqueued: Instant::now(),
                matched,
            },
        );
        state.fifo.push_back(id.clone());
        q.requests.fetch_add(1, Ordering::Relaxed);
        q.bytes.fetch_add(bytes, Ordering::Relaxed);
        q.tokens.fetch_add(tokens, Ordering::Relaxed);
        self.counters.enqueued.fetch_add(1, Ordering::Relaxed);
        self.changed.notify_waiters();
        Ok((
            RequestLease {
                queue: self.clone(),
                id,
                finished: AtomicBool::new(false),
            },
            receiver,
        ))
    }
    /// Replaces who may claim a queued request and their prefix hits, e.g.
    /// after prefill workers were replaced. A live claim keeps its worker.
    pub fn update_candidates(
        &self,
        lease: &RequestLease,
        eligible: HashSet<String>,
        matched: HashMap<String, usize>,
    ) {
        let mut state = self.state.lock().unwrap();
        if let Some(p) = state.pending.get_mut(&lease.id) {
            p.eligible = eligible;
            p.matched = matched;
            self.changed.notify_waiters();
        }
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
    /// FIFO position of the request this worker should claim, if any.
    ///
    /// The queue is in enqueue order, so requests past the affinity wait form
    /// its head: the first that fits is taken as is. Among younger requests,
    /// the first `AFFINITY_SCAN` that fit are compared and the longest prefix
    /// hit on this worker wins, FIFO on ties; one whose prefix another worker
    /// holds longer is left for that worker until it ages. A request starved
    /// past `starvation_ms` that this worker has ever had room for stops it
    /// from claiming anything younger, so its headroom can grow to fit.
    fn select(
        &self,
        state: &QueueState,
        input: &Acquire,
        largest_grant: usize,
        now: Instant,
    ) -> Result<usize, Unclaimed> {
        let affinity_wait = Duration::from_millis(self.limits.affinity_max_wait_ms);
        let starvation = Duration::from_millis(self.limits.starvation_ms);
        let worker = input.worker_url.as_str();
        let mut pick: Option<(usize, usize)> = None;
        let mut compared = 0;
        for (position, id) in state.fifo.iter().enumerate() {
            let Some(p) = state.pending.get(id) else {
                continue;
            };
            if p.current.is_some() || !p.eligible.contains(worker) {
                continue;
            }
            let waited = now.saturating_duration_since(p.enqueued);
            if p.tokens > input.max_tokens {
                // Hold this worker back for one more starvation period at
                // most. Unbounded, it could stall the whole queue: other
                // routers and push traffic keep using the room this router
                // waits for, so it may never appear.
                let holding = waited >= starvation && waited < starvation.saturating_mul(2);
                if pick.is_none() && holding && p.tokens <= largest_grant {
                    return Err(Unclaimed::Held);
                }
                continue;
            }
            if waited >= affinity_wait {
                return Ok(pick.map_or(position, |(chosen, _)| chosen));
            }
            let mine = p.matched.get(worker).copied().unwrap_or(0);
            let elsewhere = p
                .matched
                .iter()
                .filter(|(url, _)| url.as_str() != worker)
                .map(|(_, &tokens)| tokens)
                .max()
                .unwrap_or(0);
            if elsewhere <= mine && pick.is_none_or(|(_, best)| mine > best) {
                pick = Some((position, mine));
            }
            compared += 1;
            if compared >= AFFINITY_SCAN {
                break;
            }
        }
        pick.map(|(position, _)| position).ok_or(Unclaimed::Nothing)
    }

    #[cfg(test)]
    fn try_acquire(&self, input: &Acquire) -> Result<Option<Arc<Assignment>>, ApiError> {
        self.try_acquire_at(input, Instant::now())
    }

    #[cfg(test)]
    fn try_acquire_at(
        &self,
        input: &Acquire,
        now: Instant,
    ) -> Result<Option<Arc<Assignment>>, ApiError> {
        Ok(self
            .try_claim(input, &CircuitBreaker::disabled(), now)?
            .ok())
    }

    /// `now` is shared with [`Self::next_wake`] so both agree on which
    /// requests are still within their affinity wait or starvation hold.
    fn try_claim(
        &self,
        input: &Acquire,
        breaker: &CircuitBreaker,
        now: Instant,
    ) -> Result<Result<Arc<Assignment>, Unclaimed>, ApiError> {
        let mut state = self.state.lock().unwrap();
        // Bound outcome tombstones independently of live requests.
        self.evict_tombstones(&mut state, now, self.limits.max_requests.saturating_mul(5));
        Self::expire_orphans(&mut state, now);
        state.forget_idle_workers(now);
        state.last_poll.insert(input.worker_url.clone(), now);
        if let Some((current, retired)) = state.epochs.get_mut(&input.worker_url) {
            if current != &input.worker_epoch {
                if retired.contains(&input.worker_epoch) || retired.len() >= 128 {
                    return Err(ApiError::BadRequest(
                        "stale worker epoch; restart the agent/router session".into(),
                    ));
                }
                retired.insert(std::mem::replace(current, input.worker_epoch.clone()));
                // A restarted agent may have less room (smaller pool, new
                // flags). The old epoch's largest grant would let a request
                // it can no longer fit hold back the shorter ones behind it.
                state.largest_grant.remove(&input.worker_url);
                for a in live(&mut state.by_worker, &input.worker_url) {
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
        // A retried poll of a live assignment gets it back. Once terminal, a
        // retry may claim afresh: the agent reserves only after an offer, and
        // a lost offer that expired has been returned to the queue.
        let mine = live(&mut state.by_worker, &input.worker_url);
        if let Some(a) = mine
            .iter()
            .find(|a| a.epoch == input.worker_epoch && a.poll_id == input.poll_id)
        {
            return Ok(Ok(a.clone()));
        }
        // An offered probe already occupies the half-open slot. Its poll can
        // still be retried, but a new poll must wait for the breaker to admit.
        if !breaker.would_allow() {
            return Ok(Err(Unclaimed::BreakerUnavailable));
        }
        let offered = mine
            .iter()
            .filter(|a| a.state().phase == Phase::Offered)
            .count();
        if offered >= input.max_offered {
            return Ok(Err(Unclaimed::Nothing));
        }
        let largest_grant = {
            let largest = state
                .largest_grant
                .entry(input.worker_url.clone())
                .or_default();
            *largest = (*largest).max(input.max_tokens);
            *largest
        };
        let position = match self.select(&state, input, largest_grant, now) {
            Ok(position) => position,
            Err(unclaimed) => return Ok(Err(unclaimed)),
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
                prefill_permit: None,
                outcome: None,
                orphaned_until: None,
            }),
            changed: Notify::new(),
            queue_changed: self.changed.clone(),
            enqueued: p.enqueued,
            request: id.clone(),
            hold: p.hold.clone(),
            counters: self.counters.clone(),
            retired: self.retired.clone(),
        });
        self.counters.offered.fetch_add(1, Ordering::Relaxed);
        p.current = Some(assignment.clone());
        let _ = p.claims.send(assignment.clone());
        state
            .assignments
            .insert(assignment.id.clone(), assignment.clone());
        state
            .by_worker
            .entry(input.worker_url.clone())
            .or_default()
            .push(assignment.clone());
        Ok(Ok(assignment))
    }
    pub async fn acquire(
        &self,
        input: &Acquire,
        breaker: &CircuitBreaker,
    ) -> Result<Polled, ApiError> {
        let deadline = Instant::now() + Duration::from_millis(input.wait_ms.min(30_000));
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            // Register before checking the queue so a concurrent accept cannot
            // fall between observing Offered and beginning the long poll.
            changed.as_mut().enable();
            // One instant for both checks: with two, a wait ending between them
            // was "not yet" to the claim and "already over" to the wake-up, and
            // the poll slept to its deadline.
            let now = Instant::now();
            match self.try_claim(input, breaker, now)? {
                Ok(a) => return Ok(a.offer().await.map_or(Polled::Empty, Polled::Offer)),
                Err(Unclaimed::BreakerUnavailable) => return Ok(Polled::Empty),
                // Only fresher room can serve this worker now, and it learns
                // of that by asking its scheduler, not by waiting here.
                Err(Unclaimed::Held) => return Ok(Polled::Held),
                Err(Unclaimed::Nothing) => {}
            }
            // A request left for another worker's cache, or a starved one's
            // hold, ends with time; no queue notification signals either.
            let wake = self
                .next_wake(input, now)
                .map_or(deadline, |at| at.min(deadline));
            if tokio::time::timeout_at(wake.into(), changed).await.is_err()
                && Instant::now() >= deadline
            {
                // Check once at the boundary: work becoming claimable at the
                // deadline should not force an additional empty poll/backoff.
                return Ok(match self.try_claim(input, breaker, Instant::now())? {
                    Ok(a) => a.offer().await.map_or(Polled::Empty, Polled::Offer),
                    Err(Unclaimed::Held) => Polled::Held,
                    Err(Unclaimed::Nothing | Unclaimed::BreakerUnavailable) => Polled::Empty,
                });
            }
        }
    }

    /// When `select` may next answer differently for this worker without a
    /// queue notification: a request it skips for another worker's longer
    /// prefix hit stops waiting for that worker, or a starved request's hold
    /// on it ends. Only future instants count; a past one made the long poll
    /// time out immediately and rescan the queue in a busy loop.
    fn next_wake(&self, input: &Acquire, now: Instant) -> Option<Instant> {
        let affinity_wait = Duration::from_millis(self.limits.affinity_max_wait_ms);
        let hold_end = Duration::from_millis(self.limits.starvation_ms).saturating_mul(2);
        let worker = input.worker_url.as_str();
        let state = self.state.lock().unwrap();
        let largest_grant = state.largest_grant.get(worker).copied().unwrap_or(0);
        state
            .fifo
            .iter()
            .filter_map(|id| state.pending.get(id))
            .filter(|p| p.current.is_none() && p.eligible.contains(worker))
            .filter_map(|p| {
                if p.tokens > input.max_tokens {
                    return (p.tokens <= largest_grant).then(|| p.enqueued + hold_end);
                }
                let mine = p.matched.get(worker).copied().unwrap_or(0);
                p.matched
                    .iter()
                    .any(|(url, &tokens)| url.as_str() != worker && tokens > mine)
                    .then(|| p.enqueued + affinity_wait)
            })
            .filter(|&at| at > now)
            .min()
    }
    /// Commits `a` and takes its request out of the FIFO: a committed request
    /// never returns to waiting, and leaving it there made every poll and
    /// scan grow with the requests in decode.
    pub fn accept(&self, a: &Assignment) -> &'static str {
        let status = a.accept();
        if status == "committed" {
            let mut state = self.state.lock().unwrap();
            if let Some(position) = state.fifo.iter().position(|id| id == &a.request) {
                state.fifo.remove(position);
            }
        }
        status
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
    finished: AtomicBool,
}
impl RequestLease {
    /// The client's response ended normally. P's outcome usually arrives
    /// after D's stream ends, so its assignment stays committed for that
    /// report instead of being cancelled (and P aborting work it finished).
    pub fn finish(&self) {
        self.finished.store(true, Ordering::Relaxed);
    }
}
impl Drop for RequestLease {
    fn drop(&mut self) {
        let mut state = self.queue.state.lock().unwrap();
        let Some(p) = state.pending.remove(&self.id) else {
            return;
        };
        p.hold.release();
        let mut committed = false;
        if let Some(a) = p.current {
            let mut s = a.state();
            if self.finished.load(Ordering::Relaxed) && s.phase == Phase::Committed {
                // D succeeded, so P's outcome is due shortly; a 200 stream
                // that carried an error, or a lost outcome, ends at the grace.
                let until = Instant::now() + ORPHAN_OUTCOME_GRACE;
                s.orphaned_until = Some(until);
                s.outcome = None;
                drop(s);
                state.orphans.push_back((until, a.clone()));
                committed = true;
            } else {
                committed = s.phase == Phase::Committed;
                drop(s);
                a.abandon();
            }
        }
        if !committed {
            state.fifo.retain(|id| id != &self.id);
        }
        // Pollers only care about waiting requests; drained() about the last.
        if !committed || state.pending.is_empty() {
            self.queue.changed.notify_waiters();
        }
    }
}

/// How long P may take to report a committed prefill once its client's
/// response ended. D finishing implies the KV arrived, so P is done or close;
/// arbitrary, well under the bootstrap timeout P would otherwise wait out.
const ORPHAN_OUTCOME_GRACE: Duration = Duration::from_secs(30);

pub fn validate_acquire(input: &Acquire) -> Result<(), ApiError> {
    if input.dp_rank != 0
        || input.max_requests != 1
        || !(1..=MAX_OFFERED_PER_WORKER).contains(&input.max_offered)
        || input.max_tokens == 0
        || input.worker_epoch.is_empty()
        || input.worker_epoch.len() > 128
        || input.poll_id.is_empty()
        || input.poll_id.len() > 128
    {
        return Err(ApiError::BadRequest("pull requires dp_rank=0, max_requests=1, max_offered in 1..=16, positive max_tokens and bounded epoch/poll IDs".into()));
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
            max_offered: 1,
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
        let (first, _) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 60)
            .unwrap();
        assert!(queue
            .enqueue(HashSet::new(), HashMap::new(), 1, 41)
            .is_err());
        assert!(queue
            .enqueue(HashSet::new(), HashMap::new(), 11, 1)
            .is_err());
        drop(first);
        assert!(queue
            .enqueue(HashSet::new(), HashMap::new(), 20, 100)
            .is_ok());
    }
    #[test]
    fn claimant_is_pinned_and_another_poll_cannot_steal_an_offer() {
        let queue = PullQueue::new(PullConfig::default());
        let (_lease, _) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 10)
            .unwrap();
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
        let (lease, mut claims) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 10)
            .unwrap();
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
        let (lease, _) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 10)
            .unwrap();
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
        let (_lease, _) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 10)
            .unwrap();
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

    #[tokio::test]
    async fn client_cancellation_compacts_a_completed_but_unread_outcome() {
        let queue = PullQueue::new(PullConfig::default());
        let (lease, _) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 10)
            .unwrap();
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
        // Keep the result for the background task, without retaining the large
        // body. Duplicate reports cannot replace the first result.
        assert_eq!(assignment.status(), "completed");
        assert_eq!(assignment.state().outcome, Some((200, String::new())));
        assert_eq!(assignment.outcome(200, "retry".into()), "completed");
        let response = assignment.response().await.unwrap();
        assert_eq!(response.status(), 200);
        assert!(axum::body::to_bytes(response.into_body(), 0)
            .await
            .unwrap()
            .is_empty());
        assert!(assignment.state().outcome.is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn accepting_one_offer_wakes_the_next_acquire_without_new_arrivals() {
        let queue = PullQueue::new(PullConfig::default());
        let (_first, mut first_claims) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 10)
            .unwrap();
        let (_second, mut second_claims) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 10)
            .unwrap();
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
        let breaker = CircuitBreaker::disabled();
        let waiting = queue.acquire(&input, &breaker);
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
            std::task::Poll::Ready(Ok(Polled::Offer(offer))) => offer,
            _ => panic!("second offer should be immediately available"),
        };
        assert_eq!(offer.request_id, "rid-second");
    }

    #[test]
    fn claims_prefer_own_prefix_and_leave_other_owners_prefix_until_it_ages() {
        let both = || HashSet::from(["p".to_string(), "q".to_string()]);
        let queue = PullQueue::new(PullConfig {
            affinity_max_wait_ms: 60_000,
            ..Default::default()
        });
        let (_a, mut a_claims) = queue
            .enqueue(both(), HashMap::from([("q".to_string(), 8)]), 10, 10)
            .unwrap();
        let (_b, mut b_claims) = queue.enqueue(both(), HashMap::new(), 10, 10).unwrap();
        let (_c, mut c_claims) = queue
            .enqueue(both(), HashMap::from([("p".to_string(), 6)]), 10, 10)
            .unwrap();
        let mut input = acquire("p", "e", "1", 10);
        input.max_offered = 3;
        // C hits p's cache, so it goes before the older B; A is q's prefix.
        let first = queue.try_acquire(&input).unwrap().unwrap();
        assert_eq!(c_claims.try_recv().unwrap().id, first.id);
        input.poll_id = "2".into();
        let second = queue.try_acquire(&input).unwrap().unwrap();
        assert_eq!(b_claims.try_recv().unwrap().id, second.id);
        input.poll_id = "3".into();
        assert!(queue.try_acquire(&input).unwrap().is_none());
        let for_q = queue
            .try_acquire(&acquire("q", "e", "1", 10))
            .unwrap()
            .unwrap();
        assert_eq!(a_claims.try_recv().unwrap().id, for_q.id);

        // Once aged, the owner's claim no longer wins: plain FIFO.
        let queue = PullQueue::new(PullConfig {
            affinity_max_wait_ms: 0,
            ..Default::default()
        });
        let (_a, mut a_claims) = queue
            .enqueue(both(), HashMap::from([("q".to_string(), 8)]), 10, 10)
            .unwrap();
        let (_c, _) = queue
            .enqueue(both(), HashMap::from([("p".to_string(), 6)]), 10, 10)
            .unwrap();
        let aged = queue
            .try_acquire(&acquire("p", "e", "1", 10))
            .unwrap()
            .unwrap();
        assert_eq!(a_claims.try_recv().unwrap().id, aged.id);
    }

    #[tokio::test]
    async fn long_poll_claims_when_another_workers_affinity_wait_ends() {
        // The request is q's prefix; p must get it once the 50 ms wait ends,
        // not when its own 5 s long poll does.
        let queue = PullQueue::new(PullConfig {
            affinity_max_wait_ms: 50,
            ..Default::default()
        });
        let both = HashSet::from(["p".to_string(), "q".to_string()]);
        let (_lease, _claims) = queue
            .enqueue(both, HashMap::from([("q".to_string(), 8)]), 10, 10)
            .unwrap();
        let mut input = acquire("p", "e", "1", 10);
        input.wait_ms = 5_000;
        let started = Instant::now();
        let claimed = tokio::time::timeout(Duration::from_secs(2), async {
            // acquire() also waits for D to publish; observe the claim itself.
            let breaker = CircuitBreaker::disabled();
            let waiting = queue.acquire(&input, &breaker);
            tokio::pin!(waiting);
            loop {
                tokio::select! {
                    _ = &mut waiting => unreachable!("no offer is ever published"),
                    _ = tokio::time::sleep(Duration::from_millis(10)) => {
                        let state = queue.state.lock().unwrap();
                        if state.pending.values().any(|p| p.current.is_some()) {
                            return;
                        }
                    }
                }
            }
        })
        .await;
        assert!(
            claimed.is_ok(),
            "the claim waited for the long-poll deadline"
        );
        assert!(started.elapsed() >= Duration::from_millis(50));
    }

    #[test]
    fn affinity_wakeups_ignore_waits_that_already_ended() {
        // A request past its wait yet unclaimable (held by starvation or
        // max_offered) once produced a past instant, so every long poll woke
        // at once and rescanned the queue until its deadline.
        let both = || HashSet::from(["p".to_string(), "q".to_string()]);
        let q_prefix = || HashMap::from([("q".to_string(), 8)]);
        let ended = PullQueue::new(PullConfig {
            affinity_max_wait_ms: 0,
            ..Default::default()
        });
        let (_ended, _) = ended.enqueue(both(), q_prefix(), 10, 10).unwrap();
        assert!(ended
            .next_wake(&acquire("p", "e", "1", 10), Instant::now())
            .is_none());
        let pending = PullQueue::new(PullConfig {
            affinity_max_wait_ms: 60_000,
            ..Default::default()
        });
        let (_pending, _) = pending.enqueue(both(), q_prefix(), 10, 10).unwrap();
        let now = Instant::now();
        assert!(pending
            .next_wake(&acquire("p", "e", "1", 10), now)
            .is_some_and(|at| at > now));
    }

    #[test]
    fn a_claim_that_skips_for_affinity_always_gets_a_wakeup() {
        // Judged at two instants, a wait ending between them was skipped by the
        // claim and dropped by the wake-up, so the poll slept to its deadline.
        let queue = PullQueue::new(PullConfig {
            affinity_max_wait_ms: 50,
            ..Default::default()
        });
        let both = HashSet::from(["p".to_string(), "q".to_string()]);
        let (_lease, _) = queue
            .enqueue(both, HashMap::from([("q".to_string(), 8)]), 10, 10)
            .unwrap();
        let input = acquire("p", "e", "1", 10);
        let start = queue
            .state
            .lock()
            .unwrap()
            .pending
            .values()
            .next()
            .unwrap()
            .enqueued;
        for offset_ms in [0, 25, 49] {
            let now = start + Duration::from_millis(offset_ms);
            assert!(queue.try_acquire_at(&input, now).unwrap().is_none());
            assert!(
                queue.next_wake(&input, now).is_some_and(|at| at > now),
                "skipped at +{offset_ms}ms without a wake-up"
            );
        }
        let aged = start + Duration::from_millis(60);
        assert!(queue.try_acquire_at(&input, aged).unwrap().is_some());
    }

    #[test]
    fn starved_long_request_holds_back_younger_ones_for_a_bounded_time() {
        // Holding back without a bound could stall the whole queue: other
        // routers and push traffic keep taking the room this one waits for.
        let queue = PullQueue::new(PullConfig {
            starvation_ms: 1000,
            ..Default::default()
        });
        // p has reported room for 100 tokens before.
        assert!(queue
            .try_acquire(&acquire("p", "e", "0", 100))
            .unwrap()
            .is_none());
        let (_long, _) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 80, 10)
            .unwrap();
        let (_short, mut short_claims) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 10)
            .unwrap();
        let start = queue
            .state
            .lock()
            .unwrap()
            .pending
            .values()
            .map(|p| p.enqueued)
            .min()
            .unwrap();
        let at = |ms| start + Duration::from_millis(ms);
        // Within one starvation period of starving, the short request waits.
        assert!(queue
            .try_acquire_at(&acquire("p", "e", "1", 20), at(1500))
            .unwrap()
            .is_none());
        assert!(short_claims.try_recv().is_err());
        // After it, the hold lapses and the short request is served.
        let short = queue
            .try_acquire_at(&acquire("p", "e", "2", 20), at(2500))
            .unwrap()
            .expect("the hold must end after one starvation period");
        assert_eq!(short_claims.try_recv().unwrap().id, short.id);

        // Longer than any grant this worker ever reported: it never holds p.
        let queue = PullQueue::new(PullConfig {
            starvation_ms: 1000,
            ..Default::default()
        });
        let (_huge, _) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 500, 10)
            .unwrap();
        let (_short, mut short_claims) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 10)
            .unwrap();
        let short = queue
            .try_acquire_at(
                &acquire("p", "e", "1", 20),
                Instant::now() + Duration::from_millis(1500),
            )
            .unwrap()
            .unwrap();
        assert_eq!(short_claims.try_recv().unwrap().id, short.id);
    }

    #[test]
    fn committed_requests_leave_the_queue_quota() {
        // --pull-max-* counted decode too, so long decodes 503'd new requests.
        let queue = PullQueue::new(PullConfig {
            max_requests: 1,
            ..Default::default()
        });
        let (_held, _) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 10)
            .unwrap();
        assert!(matches!(
            queue.enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 10),
            Err(ApiError::PullQueueFull)
        ));
        let a = queue
            .try_acquire(&acquire("p", "e", "1", 10))
            .unwrap()
            .unwrap();
        // Offered is still queue work; committed is decode work.
        assert!(queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 10)
            .is_err());
        a.state().offer = Some(Offer {
            status: "offer",
            assignment_id: a.id.clone(),
            request_id: "rid".into(),
            path: "/generate",
            body: serde_json::Value::Null,
            headers: HashMap::new(),
            input_tokens: 10,
            expires_in_ms: 0,
        });
        assert_eq!(a.accept(), "committed");
        assert!(queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 10)
            .is_ok());
    }

    #[test]
    fn an_orphaned_committed_assignment_keeps_only_a_compact_outcome() {
        // With the client gone, a lost outcome left the assignment committed
        // forever (never a tombstone), and a late one stored an unread body.
        let queue = PullQueue::new(PullConfig::default());
        let (lease, _) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 10)
            .unwrap();
        let a = queue
            .try_acquire(&acquire("p", "e", "1", 10))
            .unwrap()
            .unwrap();
        a.state().phase = Phase::Committed;
        lease.finish();
        drop(lease);
        assert_eq!(a.outcome(500, "x".repeat(1 << 20)), "completed");
        assert_eq!(a.state().outcome, Some((500, String::new())));

        let (lease, _) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 10)
            .unwrap();
        let b = queue
            .try_acquire(&acquire("p", "e", "2", 10))
            .unwrap()
            .unwrap();
        b.state().phase = Phase::Committed;
        lease.finish();
        drop(lease);
        assert_eq!(b.status(), "committed");
        b.state().orphaned_until = Some(Instant::now());
        assert_eq!(b.status(), "cancelled");
        assert!(queue
            .retired
            .lock()
            .unwrap()
            .iter()
            .any(|(_, id)| id == &b.id));
    }

    #[test]
    fn committed_work_in_flight_does_not_stop_offers() {
        // The tombstone cap (5 x max_requests) counted live assignments, so
        // enough committed work in flight stopped every offer router-wide.
        let queue = PullQueue::new(PullConfig {
            max_requests: 1,
            ..Default::default()
        });
        let mut leases = Vec::new();
        for i in 0..7 {
            let (lease, _) = queue
                .enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 10)
                .unwrap();
            let a = queue
                .try_acquire(&acquire("p", "e", &format!("{i}"), 10))
                .unwrap()
                .expect("committed work must not use up the tombstone cap");
            a.state().offer = Some(Offer {
                status: "offer",
                assignment_id: a.id.clone(),
                request_id: "rid".into(),
                path: "/generate",
                body: serde_json::Value::Null,
                headers: HashMap::new(),
                input_tokens: 10,
                expires_in_ms: 0,
            });
            assert_eq!(queue.accept(&a), "committed");
            leases.push(lease);
        }
    }

    #[test]
    fn an_overdue_orphan_is_cancelled_by_any_workers_poll() {
        // Expiry ran only when something read the assignment's status, which
        // a departed claimant never does again.
        let queue = PullQueue::new(PullConfig::default());
        let (lease, _) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 10)
            .unwrap();
        let a = queue
            .try_acquire(&acquire("p", "e", "1", 10))
            .unwrap()
            .unwrap();
        a.state().phase = Phase::Committed;
        lease.finish();
        drop(lease);
        let later = Instant::now() + ORPHAN_OUTCOME_GRACE + Duration::from_secs(1);
        assert!(queue
            .try_acquire_at(&acquire("q", "e", "1", 10), later)
            .unwrap()
            .is_none());
        assert!(queue
            .retired
            .lock()
            .unwrap()
            .iter()
            .any(|(_, id)| id == &a.id));
        assert!(queue.state.lock().unwrap().orphans.is_empty());
    }

    #[tokio::test]
    async fn a_held_poll_returns_at_once_for_fresher_room() {
        // A held worker's long poll waited out wait_ms with the grant it began
        // with, so room freed meanwhile was not seen for up to 10 s.
        let queue = PullQueue::new(PullConfig {
            starvation_ms: 300,
            ..Default::default()
        });
        assert!(queue
            .try_acquire(&acquire("p", "e", "0", 100))
            .unwrap()
            .is_none());
        let (_long, _) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 80, 10)
            .unwrap();
        tokio::time::sleep(Duration::from_millis(350)).await;
        let mut input = acquire("p", "e", "1", 20);
        input.wait_ms = 10_000;
        let polled = tokio::time::timeout(
            Duration::from_millis(200),
            queue.acquire(&input, &CircuitBreaker::disabled()),
        )
        .await
        .expect("a held poll must not wait out wait_ms")
        .unwrap();
        assert!(matches!(polled, Polled::Held));
    }

    #[test]
    fn a_starvation_hold_wakes_its_poll_when_it_ends() {
        // The hold ended with time, which no queue notification signals, so
        // the poll slept to its deadline at low traffic.
        let queue = PullQueue::new(PullConfig {
            starvation_ms: 1000,
            ..Default::default()
        });
        assert!(queue
            .try_acquire(&acquire("p", "e", "0", 100))
            .unwrap()
            .is_none());
        let (_long, _) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 80, 10)
            .unwrap();
        let enqueued = queue
            .state
            .lock()
            .unwrap()
            .pending
            .values()
            .next()
            .unwrap()
            .enqueued;
        assert_eq!(
            queue.next_wake(
                &acquire("p", "e", "1", 20),
                enqueued + Duration::from_millis(1500)
            ),
            Some(enqueued + Duration::from_millis(2000))
        );
    }

    #[test]
    fn a_committed_request_leaves_the_fifo() {
        // Requests in decode stayed in the FIFO, so every poll's scan grew
        // with decode concurrency.
        let queue = PullQueue::new(PullConfig::default());
        let (_lease, _) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 10)
            .unwrap();
        let a = queue
            .try_acquire(&acquire("p", "e", "1", 10))
            .unwrap()
            .unwrap();
        a.state().offer = Some(Offer {
            status: "offer",
            assignment_id: a.id.clone(),
            request_id: "rid".into(),
            path: "/generate",
            body: serde_json::Value::Null,
            headers: HashMap::new(),
            input_tokens: 10,
            expires_in_ms: 0,
        });
        assert_eq!(queue.accept(&a), "committed");
        assert!(queue.state.lock().unwrap().fifo.is_empty());
    }

    #[test]
    fn a_finished_response_does_not_cancel_its_committed_prefill() {
        // D's stream ends before P reports; dropping the lease then cancelled
        // the assignment, and P aborted work it had finished and reported 499.
        let queue = PullQueue::new(PullConfig::default());
        let (lease, _) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 10)
            .unwrap();
        let a = queue
            .try_acquire(&acquire("p", "e", "1", 10))
            .unwrap()
            .unwrap();
        a.state().phase = Phase::Committed;
        lease.finish();
        drop(lease);
        assert_eq!(a.status(), "committed");
        assert_eq!(a.outcome(200, "{}".into()), "completed");

        // A client that went away still cancels.
        let (lease, _) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 10)
            .unwrap();
        let b = queue
            .try_acquire(&acquire("p", "e", "2", 10))
            .unwrap()
            .unwrap();
        b.state().phase = Phase::Committed;
        drop(lease);
        assert_eq!(b.status(), "cancelled");
    }

    #[test]
    fn tombstones_are_evicted_oldest_first_without_touching_live_work() {
        // Eviction scanned every assignment under the queue lock on each poll.
        // It now follows terminal order, and a live assignment is never evicted.
        let queue = PullQueue::new(PullConfig {
            max_requests: 1,
            ..Default::default()
        });
        let (lease, _) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 10)
            .unwrap();
        let mut ids = Vec::new();
        for i in 0..7 {
            let a = queue
                .try_acquire(&acquire("p", "e", &format!("{i}"), 10))
                .unwrap()
                .unwrap();
            ids.push(a.id.clone());
            a.reject();
            queue.requeue(&lease);
        }
        let live = queue
            .try_acquire(&acquire("p", "e", "live", 10))
            .unwrap()
            .unwrap();
        let state = queue.state.lock().unwrap();
        // Cap is 5: the two oldest tombstones went, the newest stayed.
        assert!(state.assignments.len() <= 5);
        assert!(!state.assignments.contains_key(&ids[0]));
        assert!(!state.assignments.contains_key(&ids[1]));
        assert!(state.assignments.contains_key(&ids[6]));
        assert!(state.assignments.contains_key(&live.id));
        assert_eq!(state.by_worker["p"].len(), 1);
    }

    #[test]
    fn fleet_fingerprint_is_shared_within_its_ttl() {
        // Each queued request rebuilt and sorted the fleet every second; they
        // now share one computation per model per interval.
        let queue = PullQueue::new(PullConfig::default());
        let calls = std::cell::Cell::new(0);
        let compute = || {
            calls.set(calls.get() + 1);
            7
        };
        for _ in 0..100 {
            assert_eq!(
                queue.fleet_generation("m", Duration::from_secs(60), compute),
                7
            );
        }
        assert_eq!(calls.get(), 1);
        assert_eq!(queue.fleet_generation("m", Duration::ZERO, || 8), 8);
    }

    #[test]
    fn a_restarted_worker_is_not_held_by_its_old_capacity() {
        // Largest grant 100 under the old epoch, 20 after the restart: an
        // 80-token request starved past its wait kept blocking the 10-token one.
        let queue = PullQueue::new(PullConfig {
            starvation_ms: 0,
            ..Default::default()
        });
        assert!(queue
            .try_acquire(&acquire("p", "old", "0", 100))
            .unwrap()
            .is_none());
        let (_long, _) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 80, 10)
            .unwrap();
        let (_short, mut short_claims) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 10)
            .unwrap();
        let short = queue
            .try_acquire(&acquire("p", "new", "1", 20))
            .unwrap()
            .expect("the old epoch's capacity must not hold back a fitting request");
        assert_eq!(short_claims.try_recv().unwrap().id, short.id);
    }

    #[tokio::test]
    async fn closing_refuses_new_requests_and_drains_once_held_ones_end() {
        // Shutdown stopped axum accepting while held requests still needed the
        // control plane; the queue now refuses intake and reports when empty.
        let queue = PullQueue::new(PullConfig::default());
        let (held, _) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 10)
            .unwrap();
        queue.close();
        assert!(matches!(
            queue.enqueue(HashSet::from(["p".into()]), HashMap::new(), 1, 1),
            Err(ApiError::PullDraining)
        ));
        // Held work is still claimable while draining.
        assert!(queue
            .try_acquire(&acquire("p", "e", "1", 10))
            .unwrap()
            .is_some());
        let drained = queue.drained();
        tokio::pin!(drained);
        assert!(futures::poll!(drained.as_mut()).is_pending());
        drop(held);
        assert!(futures::poll!(drained.as_mut()).is_ready());
    }

    #[test]
    fn metrics_count_each_assignment_transition_once() {
        let queue = PullQueue::new(PullConfig {
            max_requests: 1,
            ..Default::default()
        });
        let (lease, _) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 10)
            .unwrap();
        assert!(queue.enqueue(HashSet::new(), HashMap::new(), 1, 1).is_err());
        let first = queue
            .try_acquire(&acquire("p", "e", "1", 10))
            .unwrap()
            .unwrap();
        // A repeated reject is idempotent and must not count a second expiry.
        first.reject();
        first.reject();
        queue.requeue(&lease);
        let _second = queue
            .try_acquire(&acquire("p", "e", "2", 10))
            .unwrap()
            .unwrap();
        // The client going away cancels the live offer.
        drop(lease);
        let text = queue.render_metrics();
        for line in [
            "sgl_router_pull_enqueued_total 1",
            "sgl_router_pull_queue_full_total 1",
            "sgl_router_pull_assignments_total{phase=\"offered\"} 2",
            "sgl_router_pull_assignments_total{phase=\"expired\"} 1",
            "sgl_router_pull_assignments_total{phase=\"cancelled\"} 1",
            "sgl_router_pull_assignments_total{phase=\"committed\"} 0",
            "sgl_router_pull_requests{state=\"waiting\"} 0",
            "sgl_router_pull_requests{state=\"claimed\"} 0",
        ] {
            assert!(text.contains(line), "missing `{line}` in:\n{text}");
        }
    }

    #[test]
    fn rejected_request_keeps_its_original_fifo_position_without_duplicates() {
        let queue = PullQueue::new(PullConfig::default());
        let (first, _) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 10)
            .unwrap();
        let (_second, mut second_claims) = queue
            .enqueue(HashSet::from(["p".into()]), HashMap::new(), 10, 10)
            .unwrap();
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
