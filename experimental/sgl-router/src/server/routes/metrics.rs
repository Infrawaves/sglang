// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `/metrics` endpoint — Prometheus 0.0.4 exposition.
//!
//! Returns the live snapshot of [`crate::server::metrics::MetricsRegistry`].
//! Plain-text body; charset is utf-8. We deliberately don't gate this on
//! readiness — scrapers should be able to read the metrics surface even
//! while the router is warming up so the "router started but no workers
//! discovered" failure mode is observable.

use crate::discovery::{ModelId, WorkerMode};
use crate::policies::admission::engine_pending_prefill_tokens;
use crate::server::app_context::AppContext;
use crate::server::metrics::{escape_label, PrefillLoadView, WorkerSnapshot};
use crate::state::kv_events::bootstrap::PeerRegistry;
use crate::state::kv_events::{KvIndexMetrics, Tiers, ACCOUNTING_REASONS};
use crate::state::load_monitor::engine_reported_load::EngineReportedLoadSnapshot;
use crate::workers::Worker;
use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

/// Content-Type per Prometheus exposition format spec.
const PROMETHEUS_CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

pub async fn metrics(State(ctx): State<Arc<AppContext>>) -> impl IntoResponse {
    // Sample the live registry into a snapshot for the worker gauges. These
    // are pull-on-scrape (not pushed) so removed workers stop emitting series
    // immediately; see `MetricsRegistry::render_with_workers`.
    let all_workers = ctx.registry.all();
    let engine_load = all_workers
        .iter()
        .any(|w| w.mode() == WorkerMode::Prefill)
        .then(|| ctx.engine_reported_load.capture_snapshot(Instant::now()));
    let engine_models = engine_comparable_models(&all_workers, engine_load.as_ref());
    let workers: Vec<WorkerSnapshot> = all_workers
        .into_iter()
        .map(|w| {
            let prefill_load = (w.mode() == WorkerMode::Prefill)
                .then(|| prefill_load_view(&w, engine_load.as_ref(), &engine_models));
            // One lock acquisition for both health + state so the two gauges
            // can't report a torn (self-contradictory) pair for one scrape.
            let cb = w.breaker.snapshot();
            WorkerSnapshot {
                worker_url: w.url.clone(),
                mode: match w.mode() {
                    WorkerMode::Plain => "plain",
                    WorkerMode::Prefill => "prefill",
                    WorkerMode::Decode => "decode",
                },
                healthy: cb.admit,
                cb_state: cb.state_code,
                // Saturating rather than `as i64`: a guard-accounting
                // underflow would wrap usize and render as a nonsensical
                // negative gauge; clamp to a large positive ceiling instead.
                inflight: i64::try_from(w.router_inflight_load()).unwrap_or(i64::MAX),
                prefill_load,
            }
        })
        .collect();
    let mut body = ctx.metrics.render_with_workers(&workers);
    // Pull-on-scrape, like the worker gauges above: the tree and the tally
    // own the numbers, so a worker that goes away stops emitting series
    // without anything having to reset a pushed counter.
    // Emitted unconditionally: without it, "no kv series" is indistinguishable
    // between the intended metadata-only mode, a broken `kv_metrics` wiring,
    // and a regressed endpoint.
    body.push_str(
        "# HELP sgl_router_kv_tree_maintained 1 when this router maintains its own cache-aware KV tree and therefore emits the sgl_router_kv_* series; 0 when placement comes from an external Indexer, where those series would be a structural zero and are omitted rather than reported as an empty tier stream.\n",
    );
    body.push_str("# TYPE sgl_router_kv_tree_maintained gauge\n");
    body.push_str(&format!(
        "sgl_router_kv_tree_maintained {}\n",
        u8::from(ctx.kv_metrics.is_some()),
    ));
    if let Some(kv) = ctx.kv_metrics.as_ref() {
        body.push_str(&render_kv_tiers(
            kv,
            ctx.block_size_oracle.get().unwrap_or(0),
        ));
    }
    // A local tree alone does not mean peer bootstrap is on; the selector does.
    if let (Some(index), Some(_)) = (ctx.kv_index.as_ref(), ctx.config.discovery.peer_selector()) {
        body.push_str(&render_kv_peers(&index.peers()));
    }
    (
        StatusCode::OK,
        [(CONTENT_TYPE, PROMETHEUS_CONTENT_TYPE)],
        body,
    )
}

/// Models whose prefill workers all have a fresh engine sample: the only case
/// in which routing compares engine values, since one stale candidate sends the
/// whole selection to router-local tokens.
fn engine_comparable_models(
    workers: &[Arc<Worker>],
    engine_load: Option<&EngineReportedLoadSnapshot>,
) -> HashSet<ModelId> {
    let mut stale = HashSet::new();
    let mut seen = HashSet::new();
    for w in workers.iter().filter(|w| w.mode() == WorkerMode::Prefill) {
        let fresh =
            engine_load.is_some_and(|s| s.fresh_native_cache_load_for_url(&w.url).is_some());
        for model in &w.model_ids {
            seen.insert(model.clone());
            if !fresh {
                stale.insert(model.clone());
            }
        }
    }
    seen.retain(|m| !stale.contains(m));
    seen
}

/// Reads the cold ratio without feeding the estimator, so scrapes do not move
/// the window routing relies on.
fn prefill_load_view(
    worker: &Worker,
    engine_load: Option<&EngineReportedLoadSnapshot>,
    engine_models: &HashSet<ModelId>,
) -> PrefillLoadView {
    let cold_ratio = worker.measured_prefill_cold_ratio();
    let load = engine_load.and_then(|s| s.fresh_native_cache_load_for_url(&worker.url));
    let engine_path = load.is_some()
        && !worker.model_ids.is_empty()
        && worker.model_ids.iter().all(|m| engine_models.contains(m));
    let (pending_prefill_tokens, pending_source) = match load.filter(|_| engine_path) {
        Some(load) => (
            engine_pending_prefill_tokens(
                load,
                worker.prefill_tokens_acquired_since(load.captured_at),
                cold_ratio.unwrap_or(1.0),
            ),
            "engine",
        ),
        None => (worker.router_inflight_prefill_tokens(), "local"),
    };
    PrefillLoadView {
        sample_age_seconds: load.map(|l| l.captured_at.elapsed().as_secs_f64()),
        pending_prefill_tokens,
        pending_source,
        cold_ratio,
    }
}

/// Render the peer-discovery series, emitted only with a peer selector and a
/// local tree. `synced` separates "not told yet" from "told and alone".
fn render_kv_peers(peers: &PeerRegistry) -> String {
    let (len, synced) = peers.len_and_synced();
    let mut out = String::new();
    out.push_str(
        "# HELP sgl_router_kv_bootstrap_peers Ready sibling router replicas this replica could pull a cache-aware tree snapshot from. 0 with sgl_router_kv_bootstrap_peers_synced=1 means the fleet genuinely has no other ready replica; 0 with synced=0 means peer discovery has not reported yet (check RBAC for endpointslices on the router's own Service).\n",
    );
    out.push_str("# TYPE sgl_router_kv_bootstrap_peers gauge\n");
    out.push_str(&format!("sgl_router_kv_bootstrap_peers {len}\n"));
    out.push_str(
        "# HELP sgl_router_kv_bootstrap_peers_synced 1 once peer discovery has reported at least once, even with an empty result. Until then an empty peer set is not evidence of anything.\n",
    );
    out.push_str("# TYPE sgl_router_kv_bootstrap_peers_synced gauge\n");
    out.push_str(&format!(
        "sgl_router_kv_bootstrap_peers_synced {}\n",
        u8::from(synced),
    ));
    out
}

/// Render the storage-tier series: what the tree holds per worker and tier,
/// the block size to convert it to tokens, and the tagged event stream it
/// consumed.
///
/// These exist to make a router-vs-engine tier mismatch a number instead of
/// an inference. `sgl_router_kv_tree_blocks * sgl_router_kv_block_size`
/// for a worker and tier, divided by that pod's own occupancy of the tier
/// (device: `sglang_kv_used_tokens + sglang_kv_evictable_tokens`; host:
/// `sglang_hicache_host_used_tokens`; `tp_rank="0"`), is the tree's coverage
/// of the tier. About 1 means the tree mirrors the engine; about 0 means the
/// engine holds a tier that routing cannot see; a missing series means the
/// worker publishes nothing. The event counters show whether the tagged
/// stream that should feed the tree is arriving at all.
/// Emitted only when the router maintains a local tree; in metadata-only mode
/// (external Indexer) the tier and event series here would be a structural
/// zero, which the HELP text below would have the operator read as a missing
/// tier stream. See `KvEventIndex::metrics_source`.
///
/// `block_size` is 0 until the first worker reports, and a coverage panel
/// multiplies by it, so such a panel reads 0 rather than NaN on a fleet that
/// has not registered yet.
fn render_kv_tiers(kv: &KvIndexMetrics, block_size: u32) -> String {
    let mut out = String::new();

    out.push_str(
        "# HELP sgl_router_kv_block_size Tokens per KV block hash, as established from the fleet (0 until a worker reports). Multiply sgl_router_kv_tree_blocks by this to compare with the engine's token gauges.\n",
    );
    out.push_str("# TYPE sgl_router_kv_block_size gauge\n");
    out.push_str(&format!("sgl_router_kv_block_size {block_size}\n"));

    // Every tier is emitted per carrier, zeros included: a host row at 0 next
    // to a device row in the millions is the mismatch signature, and an
    // absent series cannot be told from a tier the tree never tracked.
    out.push_str(
        "# HELP sgl_router_kv_tree_blocks Blocks the cache-aware tree attributes to a worker rank, by the storage tier the worker holds them on (a block held on device and host counts under both). Times sgl_router_kv_block_size, and divided by the engine's own occupancy of that tier for the same pod (device: sglang_kv_used_tokens + sglang_kv_evictable_tokens; host: sglang_hicache_host_used_tokens; tp_rank=\"0\"), this is the tree's coverage of the tier: ~1 mirrors the engine, ~0 means the engine holds a tier routing cannot see.\n",
    );
    out.push_str("# TYPE sgl_router_kv_tree_blocks gauge\n");
    for (id, counts) in kv.tree.tier_occupancy() {
        for (slot, (_, tier)) in Tiers::SLOTS.iter().enumerate() {
            out.push_str(&format!(
                "sgl_router_kv_tree_blocks{{worker_url=\"{}\",dp_rank=\"{}\",tier=\"{}\"}} {}\n",
                escape_label(&id.url),
                id.dp_rank,
                tier,
                counts[slot],
            ));
        }
    }

    let rows = kv.tally.snapshot();
    out.push_str(
        "# HELP sgl_router_kv_events_total KV-cache events the pump consumed, by kind and the storage medium tag they carried (untagged = no medium field; unknown = a medium this build does not recognise). On a hierarchical-cache fleet block_stored/CPU_PINNED runs at about the block_removed/GPU rate; a CPU_PINNED row pinned at 0 with hicache enabled means the tier stream is not reaching the router. A nonzero block_stored/unknown row means the engine publishes a tier this build cannot rank and the tree is dropping those stores: upgrade the router.\n",
    );
    out.push_str("# TYPE sgl_router_kv_events_total counter\n");
    for r in &rows {
        out.push_str(&format!(
            "sgl_router_kv_events_total{{event=\"{}\",medium=\"{}\"}} {}\n",
            r.event, r.medium, r.events,
        ));
    }
    out.push_str(
        "# HELP sgl_router_kv_event_blocks_total Block hashes carried by the KV-cache events the pump consumed, by kind and storage medium tag. Times sgl_router_kv_block_size this is comparable to the engine's device eviction volume (block_removed/GPU) and, summed without its pool label, to sglang_hicache_backup_tokens_total (block_stored/CPU_PINNED). The two are not equal: the engine also evicts device blocks it never backed up.\n",
    );
    out.push_str("# TYPE sgl_router_kv_event_blocks_total counter\n");
    for r in &rows {
        out.push_str(&format!(
            "sgl_router_kv_event_blocks_total{{event=\"{}\",medium=\"{}\"}} {}\n",
            r.event, r.medium, r.blocks,
        ));
    }

    // A tagged removal clears only its own tier, so a batch lost in transit
    // can strand a tier bit the tree will never clear on its own. Nonzero
    // here is the explanation for tree coverage drifting above 1.
    out.push_str(
        "# HELP sgl_router_kv_event_batches_lost_total KV-event batches dropped in transit, inferred from gaps in each publisher's dense sequence number (ZMQ drops at the publisher's high-water mark). Nonzero means the tree may hold tiers a worker has already released, which shows up as sgl_router_kv_tree_blocks exceeding the engine's own occupancy of that tier.\n",
    );
    out.push_str("# TYPE sgl_router_kv_event_batches_lost_total counter\n");
    out.push_str(&format!(
        "sgl_router_kv_event_batches_lost_total {}\n",
        kv.tally.batches_lost(),
    ));

    out.push_str(
        "# HELP sgl_router_kv_tree_accounting_errors_total Times the tree's per-tier occupancy bookkeeping contradicted itself. Always 0 on a correct tree. Nonzero means sgl_router_kv_tree_blocks understates what the tree holds, and can drop a worker's series entirely — which the gauge's own HELP would have you read as a worker that publishes nothing.\n",
    );
    out.push_str("# TYPE sgl_router_kv_tree_accounting_errors_total counter\n");
    for (reason, count) in ACCOUNTING_REASONS.iter().zip(kv.tree.accounting_errors()) {
        out.push_str(&format!(
            "sgl_router_kv_tree_accounting_errors_total{{reason=\"{reason}\"}} {count}\n",
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::metrics::{RequestOutcome, WorkerModeLabel};
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    /// The two readings of a zero peer count an operator has to tell apart:
    /// "discovery has not reported yet" and "this replica is genuinely alone".
    /// Only `synced` separates them, so both series are contract.
    #[tokio::test]
    async fn kv_peer_series_distinguish_unsynced_from_alone() {
        let peers = PeerRegistry::new();
        let before = render_kv_peers(&peers);
        assert!(before.contains("sgl_router_kv_bootstrap_peers 0\n"));
        assert!(before.contains("sgl_router_kv_bootstrap_peers_synced 0\n"));

        peers.replace(vec![]);
        let alone = render_kv_peers(&peers);
        assert!(alone.contains("sgl_router_kv_bootstrap_peers 0\n"));
        assert!(
            alone.contains("sgl_router_kv_bootstrap_peers_synced 1\n"),
            "a synced empty set is a different fact from an unsynced one",
        );

        peers.replace(vec!["http://a:30000".into(), "http://b:30000".into()]);
        let warm = render_kv_peers(&peers);
        assert!(warm.contains("sgl_router_kv_bootstrap_peers 2\n"));
        assert!(warm.contains("sgl_router_kv_bootstrap_peers_synced 1\n"));
    }

    /// A router that cannot peer-bootstrap — no local tree, or no selector —
    /// emits no series rather than a confidently wrong 0. The live wiring is
    /// part of the claim, hence the full scrape.
    #[tokio::test]
    async fn kv_peer_series_require_a_peer_selector() {
        use crate::config::{DiscoveryBackend, K8sDiscoveryConfig, K8sDiscoveryMode};
        use crate::state::kv_events::KvEventIndex;

        let scrape = |ctx: Arc<AppContext>| async move {
            let res = crate::server::app::build_router(ctx)
                .oneshot(
                    Request::builder()
                        .uri("/metrics")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            let body = res.into_body().collect().await.unwrap().to_bytes();
            String::from_utf8(body.to_vec()).unwrap()
        };
        let with_selector = || {
            DiscoveryBackend::K8s(K8sDiscoveryConfig {
                namespace: "ns".into(),
                mode: K8sDiscoveryMode::Plain {
                    label_selector: "app=sglang".into(),
                },
                peer_selector: Some("kubernetes.io/service-name=sgl-router".into()),
            })
        };

        let mut ctx = AppContext::stub();
        ctx.config.discovery = with_selector();
        assert!(
            !scrape(Arc::new(ctx))
                .await
                .contains("sgl_router_kv_bootstrap_peers"),
            "a router with no local tree emits no peer series at all",
        );

        let index = KvEventIndex::new();
        index.peers().replace(vec!["http://sibling:30000".into()]);

        let mut ctx = AppContext::stub();
        ctx.kv_index = index.snapshot_source();
        assert!(
            !scrape(Arc::new(ctx))
                .await
                .contains("sgl_router_kv_bootstrap_peers"),
            "a router without --kv-peer-selector emits no peer series",
        );

        let mut ctx = AppContext::stub();
        ctx.kv_index = index.snapshot_source();
        ctx.config.discovery = with_selector();
        let body = scrape(Arc::new(ctx)).await;
        assert!(body.contains("sgl_router_kv_bootstrap_peers 1\n"));
        assert!(body.contains("sgl_router_kv_bootstrap_peers_synced 1\n"));
    }

    /// The tier series are what a coverage dashboard joins on, so their names
    /// and label keys are contract: per-worker blocks by tier with zeros
    /// emitted, the block size to convert them, and every (event, medium)
    /// cell of the tally.
    #[tokio::test]
    async fn kv_tier_series_render_per_worker_and_per_medium() {
        use crate::state::kv_events::{EventKind, EventTally, HashTree, KvWorkerId};

        let kv = KvIndexMetrics::new(Arc::new(HashTree::new()), Arc::new(EventTally::new()));
        let w = KvWorkerId::new("http://w0:30000".into(), 0);
        kv.tree.insert_tiered(&w, None, &[1, 2, 3], Tiers::DEVICE);
        kv.tree.insert_tiered(&w, None, &[1, 2], Tiers::HOST);
        kv.tree.insert_tiered(&w, None, &[1], Tiers::EXTERNAL);
        kv.tally
            .record(EventKind::BlockStored, Some("CPU_PINNED"), 2);

        let out = render_kv_tiers(&kv, 64);
        let blocks = |tier: &str, n: u32| {
            format!(
                r#"sgl_router_kv_tree_blocks{{worker_url="http://w0:30000",dp_rank="0",tier="{tier}"}} {n}"#
            )
        };
        // Every tier is asserted, the zero rows included: those are the ones
        // a later edit to `Tiers::SLOTS` would silently drop.
        let mut want: Vec<String> = [("device", 3), ("host", 2), ("disk", 0), ("external", 1)]
            .iter()
            .map(|(t, n)| blocks(t, *n))
            .collect();
        want.extend(
            [
                "sgl_router_kv_block_size 64\n",
                r#"sgl_router_kv_events_total{event="block_stored",medium="CPU_PINNED"} 1"#,
                r#"sgl_router_kv_event_blocks_total{event="block_stored",medium="CPU_PINNED"} 2"#,
                r#"sgl_router_kv_events_total{event="block_removed",medium="GPU"} 0"#,
            ]
            .iter()
            .map(|s| (*s).to_owned()),
        );
        for w in want {
            assert!(out.contains(&w), "missing {w:?}; got:\n{out}");
        }
    }

    /// The series are pulled from the tree on every scrape, so dropping a
    /// worker must make its rows disappear rather than freeze at their last
    /// value. This is what `KvEventIndex::remove_worker` relies on when it
    /// calls `clear_worker`.
    #[tokio::test]
    async fn kv_tree_blocks_drop_with_the_worker() {
        use crate::state::kv_events::{EventTally, HashTree, KvWorkerId};

        let kv = KvIndexMetrics::new(Arc::new(HashTree::new()), Arc::new(EventTally::new()));
        let w = KvWorkerId::new("http://w0:30000".into(), 0);
        kv.tree.insert_tiered(&w, None, &[1, 2], Tiers::HOST);
        assert!(render_kv_tiers(&kv, 64).contains("http://w0:30000"));

        kv.tree.clear_worker(&w);
        let out = render_kv_tiers(&kv, 64);
        assert!(
            !out.contains("http://w0:30000"),
            "a cleared worker must stop emitting series; got:\n{out}"
        );
        // The family itself stays declared so the scrape shape is stable.
        assert!(out.contains("# TYPE sgl_router_kv_tree_blocks gauge"));
    }

    /// The route wiring itself: `render_kv_tiers` had two direct unit tests
    /// but nothing exercised `ctx.kv_metrics`, so deleting the `if let` in the
    /// handler left the suite green while `/metrics` silently stopped emitting
    /// all four families.
    #[tokio::test]
    async fn metrics_endpoint_emits_kv_series_when_a_tree_is_maintained() {
        use crate::state::kv_events::{EventTally, HashTree, KvWorkerId};

        let mut ctx = AppContext::stub();
        let tree = Arc::new(HashTree::new());
        tree.insert_tiered(
            &KvWorkerId::new("http://w0:30000".into(), 0),
            None,
            &[1, 2],
            Tiers::HOST,
        );
        ctx.kv_metrics = Some(KvIndexMetrics::new(tree, Arc::new(EventTally::new())));
        let ctx = Arc::new(ctx);

        let app = crate::server::app::build_router(ctx.clone());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let body = std::str::from_utf8(&body).unwrap();
        assert!(body.contains("sgl_router_kv_tree_maintained 1"));
        assert!(body.contains(
            r#"sgl_router_kv_tree_blocks{worker_url="http://w0:30000",dp_rank="0",tier="host"} 2"#
        ));
        assert!(body.contains("sgl_router_kv_event_batches_lost_total 0"));
        assert!(
            body.contains(r#"sgl_router_kv_tree_accounting_errors_total{reason="underflow"} 0"#)
        );
    }

    /// The other half of the gate: with no local tree the families are absent,
    /// but the mode gauge still says so rather than leaving the operator to
    /// guess whether the endpoint regressed.
    #[tokio::test]
    async fn metrics_endpoint_reports_the_mode_when_no_tree_is_maintained() {
        let ctx = Arc::new(AppContext::stub());
        assert!(ctx.kv_metrics.is_none());
        let app = crate::server::app::build_router(ctx.clone());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let body = std::str::from_utf8(&body).unwrap();
        assert!(body.contains("sgl_router_kv_tree_maintained 0"));
        assert!(
            !body.contains("sgl_router_kv_tree_blocks{"),
            "a structural zero must not be emitted as if it were a reading",
        );
    }

    #[tokio::test]
    async fn metrics_endpoint_returns_prometheus_text() {
        let ctx = Arc::new(AppContext::stub());
        let app = crate::server::app::build_router(ctx.clone());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let content_type = res
            .headers()
            .get(CONTENT_TYPE)
            .expect("content-type header")
            .to_str()
            .unwrap()
            .to_owned();
        assert!(
            content_type.starts_with("text/plain"),
            "expected text/plain, got {content_type}",
        );
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let body = std::str::from_utf8(&body).unwrap();
        // Every metric family should at least carry its HELP/TYPE lines.
        assert!(body.contains("# TYPE sgl_router_requests_total counter"));
        assert!(body.contains("# TYPE sgl_router_active_load gauge"));
    }

    #[tokio::test]
    async fn metrics_endpoint_reflects_recorded_counters() {
        let ctx = Arc::new(AppContext::stub());
        ctx.metrics.record_worker_request(
            "http://w-test:30000",
            "tiny",
            WorkerModeLabel::Prefill,
            RequestOutcome::Success,
        );
        let app = crate::server::app::build_router(ctx.clone());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let body = std::str::from_utf8(&body).unwrap();
        assert!(
            body.contains(r#"worker_url="http://w-test:30000""#),
            "metrics did not include the recorded worker_url; got:\n{body}",
        );
    }

    #[tokio::test]
    async fn metrics_endpoint_samples_worker_gauges_from_registry() {
        use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};

        let ctx = Arc::new(AppContext::stub());
        ctx.registry
            .add(WorkerSpec {
                id: WorkerId("p0".into()),
                url: "http://p0:30000".into(),
                mode: WorkerMode::Prefill,
                model_ids: vec![ModelId("m".into())],
                bootstrap_port: None,
            })
            .unwrap();
        let app = crate::server::app::build_router(ctx.clone());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let body = std::str::from_utf8(&body).unwrap();
        // Pool size reflects the registered prefill worker, and the per-worker
        // gauges are sampled (fresh breaker => healthy, closed, 0 inflight).
        assert!(
            body.contains(r#"sgl_router_workers{mode="prefill"} 1"#),
            "got:\n{body}"
        );
        assert!(body.contains(r#"sgl_router_worker_health{worker_url="http://p0:30000"} 1"#));
        assert!(body.contains(r#"sgl_router_worker_cb_state{worker_url="http://p0:30000"} 0"#));
        assert!(
            body.contains(r#"sgl_router_worker_inflight_requests{worker_url="http://p0:30000"} 0"#)
        );
    }

    /// Pending tokens must follow routing's source choice: one stale prefill
    /// worker sends the whole model to router-local tokens, so a fresh peer
    /// must not show its engine value then.
    #[tokio::test]
    async fn metrics_endpoint_pending_tokens_follow_routing_source() {
        use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
        use crate::state::load_monitor::engine_reported_load::{LoadStat, NativeCacheRankLoad};

        let ctx = Arc::new(AppContext::stub());
        for id in ["p0", "p1"] {
            ctx.registry
                .add(WorkerSpec {
                    id: WorkerId(id.into()),
                    url: format!("http://{id}:30000"),
                    mode: WorkerMode::Prefill,
                    model_ids: vec![ModelId("m".into())],
                    bootstrap_port: None,
                })
                .unwrap();
        }
        let sample = |waiting_uncached| LoadStat {
            num_running_reqs: 0,
            num_waiting_reqs: 3,
            num_tokens: 0,
            max_total_num_tokens: 100_000,
            native_cache: Some(NativeCacheRankLoad {
                num_waiting_uncached_tokens: waiting_uncached,
                num_total_tokens: 0,
                max_running_requests: 16,
                total_prefill_uncached_tokens: 0,
                total_prefill_busy_us: 0,
                total_prefill_input_tokens: Some(0),
            }),
        };
        // Dispatched before p0's sample, so the engine path does not add it.
        let p0 = ctx.registry.get(&WorkerId("p0".into())).unwrap();
        let _inflight = p0.timestamped_load_guard(700);
        std::thread::sleep(std::time::Duration::from_millis(2));
        ctx.engine_reported_load
            .set("http://p0:30000", 0, sample(4_096), Instant::now());

        let scrape = || async {
            let res = crate::server::app::build_router(Arc::clone(&ctx))
                .oneshot(
                    Request::builder()
                        .uri("/metrics")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            let body = res.into_body().collect().await.unwrap().to_bytes();
            String::from_utf8(body.to_vec()).unwrap()
        };
        let has = |body: &str, line: &str| {
            assert!(
                body.lines().any(|l| l == line),
                "missing `{line}`; got:\n{body}"
            )
        };

        let body = scrape().await;
        has(
            &body,
            r#"sgl_router_worker_engine_load_fresh{worker_url="http://p0:30000"} 1"#,
        );
        has(
            &body,
            r#"sgl_router_worker_engine_load_fresh{worker_url="http://p1:30000"} 0"#,
        );
        has(
            &body,
            r#"sgl_router_worker_pending_prefill_tokens{worker_url="http://p0:30000",source="local"} 700"#,
        );
        assert!(body.contains(
            r#"sgl_router_worker_engine_load_age_seconds{worker_url="http://p0:30000"} "#
        ));

        ctx.engine_reported_load
            .set("http://p1:30000", 0, sample(0), Instant::now());
        let body = scrape().await;
        has(
            &body,
            r#"sgl_router_worker_pending_prefill_tokens{worker_url="http://p0:30000",source="engine"} 4096"#,
        );
    }
}
