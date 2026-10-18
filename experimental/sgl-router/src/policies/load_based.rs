// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::config::LoadBasedMetric;
use crate::policies::admission::FreshLoadLookup;
use crate::policies::scoring::ScoringPolicy;
use crate::policies::SelectionContext;
use crate::workers::Worker;
use std::sync::Arc;

/// Prefers the least-loaded candidate; `select()` is the blanket impl's.
#[derive(Debug, Default)]
pub struct LoadBasedPolicy {
    metric: LoadBasedMetric,
}

impl LoadBasedPolicy {
    /// Ranks by request count (`--load-based-metric requests`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Ranks by the configured load unit.
    pub fn with_metric(metric: LoadBasedMetric) -> Self {
        Self { metric }
    }
}

impl ScoringPolicy for LoadBasedPolicy {
    fn needs_load_snapshot(&self) -> bool {
        true
    }

    fn needs_dispatch_timestamps(&self) -> bool {
        true
    }

    /// `tokens` weighs each dispatch by its prompt length, so the request must
    /// be tokenized rather than sized from its JSON body.
    fn needs_tokens(&self) -> bool {
        self.metric == LoadBasedMetric::Tokens
    }

    /// `1.0` for the least loaded down to `0.0` for the most, min-max scaled to
    /// the CURRENT fleet -- relative, not absolute, so it cannot saturate:
    /// `1 - load/256` reads a busy fleet as all-`0.0`, tied inside
    /// `TIE_EPSILON`, so the term dies exactly when load matters most.
    ///
    /// With `--load-based-metric tokens`, load is the prompt tokens each
    /// worker still has to prefill (see
    /// [`FreshLoadLookup::pending_prefill_tokens`]): two 20K-token prompts
    /// are more work than five 3K-token ones. When every candidate carries
    /// the same token load (an idle fleet) the request count ranks instead.
    ///
    /// Purely a preference: "everybody is busy" is not a reason to refuse to
    /// route, so this term never constrains. Capacity is `--filter`'s job.
    fn scores(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Vec<f32> {
        let lookup = FreshLoadLookup::new(ctx.load_snapshot(), workers.iter());
        if self.metric == LoadBasedMetric::Tokens {
            let tokens: Vec<u64> = (workers.iter())
                .map(|w| lookup.pending_prefill_tokens(w))
                .collect();
            if tokens.iter().min() != tokens.iter().max() {
                return min_max_scores(tokens);
            }
        }
        let requests = workers.iter().map(|w| lookup.score_load(w) as u64);
        min_max_scores(requests.collect())
    }
}

/// `1.0` for the lowest load down to `0.0` for the highest.
fn min_max_scores(loads: Vec<u64>) -> Vec<f32> {
    let lo = loads.iter().min().copied().unwrap_or(0);
    let span = (loads.iter().max().copied().unwrap_or(0) - lo) as f64;
    // `max(1.0)` is exact: a zero span means every `l - lo` is zero too.
    let score = |l: u64| (1.0 - (l - lo) as f64 / span.max(1.0)) as f32;
    loads.into_iter().map(score).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::policies::scoring::argmax::TIE_EPSILON;
    use crate::policies::Policy;
    use crate::state::load_monitor::engine_reported_load::{
        EngineReportedLoadSnapshot, EngineReportedSchedulingLoad, EngineReportedWorkerLoad,
    };
    use std::collections::HashMap;
    use std::time::Instant;

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        }))
    }

    #[test] // upstream's, retargeted from `pick_min_load` onto blanket `select`
    fn empty_returns_none() {
        let m = ModelId("tiny".into());
        let ctx = SelectionContext::new(&m, None);
        assert!(LoadBasedPolicy::new().select(&[], &ctx).is_none());
    }

    /// `select()` alone CANNOT detect a broken score: ARGMAX breaks a tie on
    /// load, so a constant `scores()` still lands on the minimum and that arm
    /// passes for the wrong reason. Ranking is therefore asserted on the vector
    /// itself, strictly outside `TIE_EPSILON` so a saturating curve cannot hide
    /// in the tie band -- what `300,900` is for. NaN needs its own arm because
    /// no ORDERING sees it: it makes every comparison false, which on `0,0` is
    /// the expected answer. Upstream's `picks_lowest_active_load` goes under
    /// rule 4 -- the unique-minimum 2-worker case, which `0,1` subsumes.
    #[test]
    fn scores_rank_strictly_by_load_and_the_choice_lands_on_the_minimum() {
        let model = ModelId("tiny".into());
        let (ctx, p) = (SelectionContext::new(&model, None), LoadBasedPolicy::new());
        for spec in ["0,1", "2,1,0", "1,0,1", "0,0", "5,2,9,2", "300,900"] {
            let loads: Vec<usize> = spec.split(',').map(|s| s.parse().unwrap()).collect();
            let ws: Vec<Arc<Worker>> = (0..loads.len()).map(|i| worker(&format!("w{i}"))).collect();
            let _held: Vec<_> = (ws.iter().zip(&loads))
                .flat_map(|(w, n)| (0..*n).map(move |_| w.load_guard()))
                .collect();
            let scores = p.scores(&ws, &ctx);
            for (i, j) in (0..loads.len()).flat_map(|i| (0..loads.len()).map(move |j| (i, j))) {
                let ok = (scores[i] > scores[j] + TIE_EPSILON, scores[i].is_nan());
                assert_eq!(ok, (loads[i] < loads[j], false), "{spec} scored {scores:?}");
            }
            let got = p
                .select(&ws, &ctx)
                .expect("non-empty")
                .router_inflight_load();
            assert_eq!(got, *loads.iter().min().expect("non-empty"), "{spec}");
        }
    }

    #[test]
    fn request_snapshot_overrides_later_router_active_load() {
        let model = ModelId("tiny".into());
        let w0 = worker("w0");
        let w1 = worker("w1");
        // After the request snapshot, local counters say w0 is lighter.
        // The policy must still preserve the frozen Engine Load ordering.
        let _after_snapshot: Vec<_> = (0..10).map(|_| w1.load_guard()).collect();
        let snapshot = EngineReportedLoadSnapshot::from_workers(
            23,
            HashMap::from([
                (
                    w0.url.clone(),
                    EngineReportedWorkerLoad {
                        num_running_reqs: 50,
                        num_waiting_reqs: 0,
                        num_tokens: 0,
                        max_total_num_tokens: 0,
                        captured_at: Instant::now(),
                    },
                ),
                (
                    w1.url.clone(),
                    EngineReportedWorkerLoad {
                        num_running_reqs: 1,
                        num_waiting_reqs: 0,
                        num_tokens: 0,
                        max_total_num_tokens: 0,
                        captured_at: Instant::now(),
                    },
                ),
            ]),
        );
        let ctx = SelectionContext::new(&model, None).with_load_snapshot(&snapshot);
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];

        assert_eq!(
            LoadBasedPolicy::new().select(&workers, &ctx).unwrap().id,
            w1.id,
            "load-based scoring must use the request snapshot before local active-load"
        );
    }

    #[test]
    fn recent_dispatches_after_snapshot_change_load_based_choice() {
        let model = ModelId("tiny".into());
        let w0 = worker("w0");
        let w1 = worker("w1");
        let captured_at = Instant::now();
        let snapshot = EngineReportedLoadSnapshot::from_workers(
            37,
            HashMap::from([
                (
                    w0.url.clone(),
                    EngineReportedWorkerLoad {
                        num_running_reqs: 0,
                        num_waiting_reqs: 0,
                        num_tokens: 0,
                        max_total_num_tokens: 0,
                        captured_at,
                    },
                ),
                (
                    w1.url.clone(),
                    EngineReportedWorkerLoad {
                        num_running_reqs: 1,
                        num_waiting_reqs: 0,
                        num_tokens: 0,
                        max_total_num_tokens: 0,
                        captured_at,
                    },
                ),
            ]),
        );
        let _after_snapshot = [w0.timestamped_load_guard(1), w0.timestamped_load_guard(1)];
        let ctx = SelectionContext::new(&model, None).with_load_snapshot(&snapshot);
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];

        assert_eq!(
            LoadBasedPolicy::new().select(&workers, &ctx).unwrap().id,
            w1.id,
            "dispatches newer than Engine Load must correct its queue depth"
        );
    }

    #[test]
    fn dispatches_before_snapshot_are_not_double_counted() {
        let model = ModelId("tiny".into());
        let w0 = worker("w0");
        let w1 = worker("w1");
        let _before_snapshot = [w0.timestamped_load_guard(1), w0.timestamped_load_guard(1)];
        std::thread::sleep(std::time::Duration::from_millis(5));
        let captured_at = Instant::now();
        let snapshot = EngineReportedLoadSnapshot::from_workers(
            41,
            HashMap::from([
                (
                    w0.url.clone(),
                    EngineReportedWorkerLoad {
                        num_running_reqs: 0,
                        num_waiting_reqs: 0,
                        num_tokens: 0,
                        max_total_num_tokens: 0,
                        captured_at,
                    },
                ),
                (
                    w1.url.clone(),
                    EngineReportedWorkerLoad {
                        num_running_reqs: 1,
                        num_waiting_reqs: 0,
                        num_tokens: 0,
                        max_total_num_tokens: 0,
                        captured_at,
                    },
                ),
            ]),
        );
        let ctx = SelectionContext::new(&model, None).with_load_snapshot(&snapshot);
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];

        assert_eq!(
            LoadBasedPolicy::new().select(&workers, &ctx).unwrap().id,
            w0.id,
            "slots already covered by the snapshot must not be added again"
        );
    }

    #[test]
    fn incomplete_snapshot_uses_frozen_local_active_fallback() {
        let model = ModelId("tiny".into());
        let w0 = worker("w0");
        let w1 = worker("w1");
        let _local_load = [w0.load_guard(), w0.load_guard()];
        let snapshot = EngineReportedLoadSnapshot::from_workers(
            43,
            HashMap::from([(
                w0.url.clone(),
                EngineReportedWorkerLoad {
                    num_running_reqs: 0,
                    num_waiting_reqs: 0,
                    num_tokens: 0,
                    max_total_num_tokens: 0,
                    captured_at: Instant::now(),
                },
            )]),
        );
        let ctx = SelectionContext::new(&model, None).with_load_snapshot(&snapshot);
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];

        assert_eq!(
            LoadBasedPolicy::new().select(&workers, &ctx).unwrap().id,
            w1.id,
            "a partial Engine Load set must not mix engine and local gauges"
        );
    }

    fn native(waiting_reqs: u64, waiting_tokens: u64, at: Instant) -> EngineReportedSchedulingLoad {
        EngineReportedSchedulingLoad {
            num_running_reqs: 0,
            num_waiting_reqs: waiting_reqs,
            num_waiting_uncached_tokens: waiting_tokens,
            num_used_tokens: 0,
            num_total_tokens: 0,
            max_total_num_tokens: 1_000_000,
            max_running_requests: 64,
            prefill_throughput_tokens_per_s: None,
            estimated_prefill_queue_ms: None,
            captured_at: at,
        }
    }

    fn pick(metric: LoadBasedMetric, ws: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> WorkerId {
        LoadBasedPolicy::with_metric(metric)
            .select(ws, ctx)
            .expect("non-empty")
            .id
            .clone()
    }

    /// The design doc's example on the router-local path: two 20K prompts
    /// outweigh five 3K prompts under `tokens`, but not under `requests`.
    #[test]
    fn tokens_metric_ranks_local_inflight_by_prompt_tokens() {
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        let ws = vec![worker("w0"), worker("w1")];
        let _heavy: Vec<_> = (0..2)
            .map(|_| ws[0].timestamped_load_guard(20_000))
            .collect();
        let _light: Vec<_> = (0..5)
            .map(|_| ws[1].timestamped_load_guard(3_000))
            .collect();

        assert_eq!(pick(LoadBasedMetric::Tokens, &ws, &ctx), ws[1].id);
        assert_eq!(pick(LoadBasedMetric::Requests, &ws, &ctx), ws[0].id);
    }

    /// The same example on the engine path: `num_waiting_uncached_tokens`
    /// decides under `tokens`, the request count under `requests`.
    #[test]
    fn tokens_metric_ranks_native_samples_by_waiting_uncached_tokens() {
        let model = ModelId("tiny".into());
        let ws = vec![worker("w0"), worker("w1")];
        let at = Instant::now();
        let snapshot = EngineReportedLoadSnapshot::from_native_cache_workers(
            51,
            HashMap::from([
                (ws[0].url.clone(), native(2, 40_000, at)),
                (ws[1].url.clone(), native(5, 15_000, at)),
            ]),
        );
        let ctx = SelectionContext::new(&model, None).with_load_snapshot(&snapshot);

        assert_eq!(pick(LoadBasedMetric::Tokens, &ws, &ctx), ws[1].id);
        assert_eq!(pick(LoadBasedMetric::Requests, &ws, &ctx), ws[0].id);
    }

    #[test]
    fn tokens_dispatched_after_the_sample_are_added_to_it() {
        let model = ModelId("tiny".into());
        let ws = vec![worker("w0"), worker("w1")];
        let snapshot = EngineReportedLoadSnapshot::from_native_cache_workers(
            53,
            HashMap::from([
                (ws[0].url.clone(), native(0, 1_000, Instant::now())),
                (ws[1].url.clone(), native(0, 2_000, Instant::now())),
            ]),
        );
        let _after = ws[0].timestamped_load_guard(5_000);
        let ctx = SelectionContext::new(&model, None).with_load_snapshot(&snapshot);

        assert_eq!(
            pick(LoadBasedMetric::Tokens, &ws, &ctx),
            ws[1].id,
            "1_000 reported + 5_000 unreported outweighs 2_000"
        );
    }

    #[test]
    fn tokens_dispatched_before_the_sample_are_not_double_counted() {
        let model = ModelId("tiny".into());
        let ws = vec![worker("w0"), worker("w1")];
        let _before = ws[0].timestamped_load_guard(5_000);
        std::thread::sleep(std::time::Duration::from_millis(5));
        let at = Instant::now();
        let snapshot = EngineReportedLoadSnapshot::from_native_cache_workers(
            55,
            HashMap::from([
                (ws[0].url.clone(), native(0, 1_000, at)),
                (ws[1].url.clone(), native(0, 2_000, at)),
            ]),
        );
        let ctx = SelectionContext::new(&model, None).with_load_snapshot(&snapshot);

        assert_eq!(
            pick(LoadBasedMetric::Tokens, &ws, &ctx),
            ws[0].id,
            "the sample already covers the earlier dispatch"
        );
    }

    #[test]
    fn partial_native_coverage_ranks_every_candidate_by_local_tokens() {
        let model = ModelId("tiny".into());
        let ws = vec![worker("w0"), worker("w1")];
        // Only w0 has a sample, and it says w0 is idle; mixing it with w1's
        // local count would prefer w0. Both must use local tokens instead.
        let snapshot = EngineReportedLoadSnapshot::from_native_cache_workers(
            57,
            HashMap::from([(ws[0].url.clone(), native(0, 0, Instant::now()))]),
        );
        let _w0 = ws[0].timestamped_load_guard(9_000);
        let _w1 = ws[1].timestamped_load_guard(1_000);
        let ctx = SelectionContext::new(&model, None).with_load_snapshot(&snapshot);

        assert_eq!(pick(LoadBasedMetric::Tokens, &ws, &ctx), ws[1].id);
    }

    /// `tokens` must tokenize the request: sizing it from the JSON body
    /// misranks ID-heavy prompts. `requests` keeps the cheap path.
    #[test]
    fn only_the_tokens_metric_asks_for_request_tokens() {
        assert!(LoadBasedPolicy::with_metric(LoadBasedMetric::Tokens).needs_request_tokens());
        assert!(!LoadBasedPolicy::with_metric(LoadBasedMetric::Requests).needs_request_tokens());
        assert!(!LoadBasedPolicy::new().needs_request_tokens());
    }

    #[test]
    fn equal_token_load_falls_back_to_request_count() {
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        let ws = vec![worker("w0"), worker("w1")];
        // Plain guards carry no tokens: both workers read 0 tokens.
        let _held = [ws[0].load_guard(), ws[0].load_guard()];

        assert_eq!(pick(LoadBasedMetric::Tokens, &ws, &ctx), ws[1].id);
    }
}
