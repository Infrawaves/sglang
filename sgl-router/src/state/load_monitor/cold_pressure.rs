// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Cluster cold-token pressure for an upstream gateway's rate limiter: the
//! engines' waiting uncached prefill tokens over the healthy prefill pool's
//! budget, sampled on a fixed interval and served by `GET /prefill_pressure`.

use crate::state::load_monitor::engine_reported_load::{ColdTokenReading, EngineReportedLoadTable};
use crate::workers::Worker;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::task::JoinHandle;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ColdPressureSample {
    /// `capped_cold_tokens / capacity_tokens`, in [0, 1]. Meaningless unless
    /// [`Self::has_data`].
    pub pressure: f64,
    /// Uncapped sum, so a backlog past full stays visible.
    pub cold_tokens: u64,
    /// Each worker's cold tokens capped at its own capacity: one worker's backlog
    /// does not occupy the idle workers the router can still pick.
    pub capped_cold_tokens: u64,
    pub capacity_tokens: u64,
    pub per_worker_capacity_tokens: u64,
    pub healthy_prefill_workers: usize,
    /// Healthy workers with a DP rank stale, or never reported past
    /// [`UNREPORTED_GRACE`]; each counts as at least full, since an idle engine
    /// re-publishes every second.
    pub stale_workers: usize,
    /// Never-reported workers within [`UNREPORTED_GRACE`] of joining; left out of
    /// both sides so a scale-out does not read as load.
    pub unreported_workers: usize,
    /// Never-reported workers past [`UNREPORTED_GRACE`], typically without load
    /// publishing; each counts as full so a silent pool cannot read as idle.
    pub overdue_workers: usize,
    /// For display only; staleness uses `sampled_at`, which clock steps cannot move.
    pub sampled_at_unix_ms: u64,
    #[serde(skip)]
    pub sampled_at: Instant,
}

/// A sample older than this many sampling intervals means the sampler stopped.
/// Arbitrary; tolerates two missed ticks.
pub const STALE_AFTER_INTERVALS: u64 = 3;

/// How long a newly pooled worker may go without its first load report before
/// it counts as full. Arbitrary; a healthy engine reports within seconds.
pub const UNREPORTED_GRACE: Duration = Duration::from_secs(30);

impl ColdPressureSample {
    /// False with no healthy prefill worker, or while none has reported.
    pub fn has_data(&self) -> bool {
        self.healthy_prefill_workers > self.unreported_workers + self.overdue_workers
    }

    /// True once the sampler has missed [`STALE_AFTER_INTERVALS`] ticks.
    pub fn is_stale(&self, sample_interval_secs: u64, now: Instant) -> bool {
        let max_age =
            Duration::from_secs(sample_interval_secs.saturating_mul(STALE_AFTER_INTERVALS));
        now.saturating_duration_since(self.sampled_at) > max_age
    }
}

/// Latest sample, written by [`spawn_sampler`] and read by the HTTP routes.
#[derive(Debug, Default)]
pub struct ColdPressure {
    latest: RwLock<Option<ColdPressureSample>>,
}

impl ColdPressure {
    /// `None` until the first sample.
    pub fn latest(&self) -> Option<ColdPressureSample> {
        self.latest
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub fn store(&self, sample: ColdPressureSample) {
        *self.latest.write().unwrap_or_else(|e| e.into_inner()) = Some(sample);
    }
}

/// `workers` pairs each pool member's URL with how long it has been pooled.
/// `num_waiting_uncached_tokens` is summed over DP ranks and includes a
/// chunked prefill's remainder.
pub fn sample<'u>(
    workers: impl IntoIterator<Item = (&'u str, Duration)>,
    readings: &HashMap<String, ColdTokenReading>,
    per_worker_capacity_tokens: u64,
    sampled_at: Instant,
    sampled_at_unix_ms: u64,
) -> ColdPressureSample {
    let mut healthy_prefill_workers = 0usize;
    let mut stale_workers = 0usize;
    let mut unreported_workers = 0usize;
    let mut overdue_workers = 0usize;
    let mut cold_tokens = 0u64;
    let mut capped_cold_tokens = 0u64;
    for (url, pooled_for) in workers {
        healthy_prefill_workers += 1;
        let cold = match readings.get(url) {
            Some(ColdTokenReading::Fresh(tokens)) => *tokens,
            // Ranks of a new DP worker come up a moment apart; no floor yet.
            Some(ColdTokenReading::Partial(tokens)) if pooled_for < UNREPORTED_GRACE => *tokens,
            Some(ColdTokenReading::Stale(tokens) | ColdTokenReading::Partial(tokens)) => {
                stale_workers += 1;
                (*tokens).max(per_worker_capacity_tokens)
            }
            None if pooled_for < UNREPORTED_GRACE => {
                unreported_workers += 1;
                continue;
            }
            None => {
                overdue_workers += 1;
                per_worker_capacity_tokens
            }
        };
        cold_tokens = cold_tokens.saturating_add(cold);
        capped_cold_tokens =
            capped_cold_tokens.saturating_add(cold.min(per_worker_capacity_tokens));
    }
    let counted_workers = (healthy_prefill_workers - unreported_workers) as u64;
    let capacity_tokens = per_worker_capacity_tokens.saturating_mul(counted_workers);
    let pressure = if capacity_tokens == 0 {
        1.0
    } else {
        capped_cold_tokens as f64 / capacity_tokens as f64
    };
    ColdPressureSample {
        pressure,
        cold_tokens,
        capped_cold_tokens,
        capacity_tokens,
        per_worker_capacity_tokens,
        healthy_prefill_workers,
        stale_workers,
        unreported_workers,
        overdue_workers,
        sampled_at_unix_ms,
        sampled_at,
    }
}

pub(crate) fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// Samples `prefill_pool()` immediately, then every `interval`.
pub fn spawn_sampler(
    pressure: Arc<ColdPressure>,
    prefill_pool: impl Fn() -> Vec<Arc<Worker>> + Send + 'static,
    engine_load: Arc<EngineReportedLoadTable>,
    per_worker_capacity_tokens: u64,
    interval: Duration,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            let workers = prefill_pool();
            let now = Instant::now();
            let readings = engine_load.cold_token_readings(now);
            let sample = sample(
                workers
                    .iter()
                    .map(|w| (w.url.as_str(), w.pooled_at.elapsed())),
                &readings,
                per_worker_capacity_tokens,
                now,
                unix_ms(),
            );
            tracing::debug!(
                pressure = sample.pressure,
                cold_tokens = sample.cold_tokens,
                capped_cold_tokens = sample.capped_cold_tokens,
                capacity_tokens = sample.capacity_tokens,
                healthy_prefill_workers = sample.healthy_prefill_workers,
                stale_workers = sample.stale_workers,
                unreported_workers = sample.unreported_workers,
                overdue_workers = sample.overdue_workers,
                "cold pressure sampled"
            );
            pressure.store(sample);
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ColdTokenReading::{Fresh, Partial, Stale};

    const JUST_JOINED: Duration = Duration::ZERO;
    const SETTLED: Duration = Duration::from_secs(3600);

    fn readings(entries: &[(&str, ColdTokenReading)]) -> HashMap<String, ColdTokenReading> {
        entries
            .iter()
            .map(|(url, reading)| (url.to_string(), *reading))
            .collect()
    }

    fn pool<'u>(urls: &[&'u str], pooled_for: Duration) -> Vec<(&'u str, Duration)> {
        urls.iter().map(|url| (*url, pooled_for)).collect()
    }

    #[test]
    fn pressure_is_cold_tokens_over_healthy_pool_budget() {
        let urls: Vec<String> = (0..10).map(|i| format!("http://p{i}")).collect();
        let entries: Vec<_> = urls.iter().map(|u| (u.as_str(), Fresh(10_000))).collect();
        let workers = urls.iter().map(|u| (u.as_str(), SETTLED));
        let s = sample(workers, &readings(&entries), 20_000, Instant::now(), 7);
        assert_eq!(s.cold_tokens, 100_000);
        assert_eq!(s.capacity_tokens, 200_000);
        assert_eq!(s.healthy_prefill_workers, 10);
        assert_eq!(s.stale_workers, 0);
        assert_eq!(s.unreported_workers, 0);
        assert_eq!(s.overdue_workers, 0);
        assert!((s.pressure - 0.5).abs() < 1e-9, "{}", s.pressure);
        assert_eq!(s.sampled_at_unix_ms, 7);
        assert!(s.has_data());
    }

    /// Adding workers that have not reported yet must not raise pressure.
    #[test]
    fn scale_out_before_first_report_keeps_pressure() {
        let urls: Vec<String> = (0..20).map(|i| format!("http://p{i}")).collect();
        let entries: Vec<_> = urls[..10]
            .iter()
            .map(|u| (u.as_str(), Fresh(10_000)))
            .collect();
        let workers = urls.iter().enumerate().map(|(i, u)| {
            let pooled_for = if i < 10 { SETTLED } else { JUST_JOINED };
            (u.as_str(), pooled_for)
        });
        let s = sample(workers, &readings(&entries), 20_000, Instant::now(), 0);
        assert_eq!(s.unreported_workers, 10);
        assert_eq!(s.overdue_workers, 0);
        assert_eq!(s.capacity_tokens, 200_000);
        assert!((s.pressure - 0.5).abs() < 1e-9, "{}", s.pressure);
    }

    /// One idle reporter must not make a pool of long-silent workers read as idle.
    #[test]
    fn workers_silent_past_the_grace_count_as_full() {
        let urls: Vec<String> = (0..10).map(|i| format!("http://p{i}")).collect();
        let workers = urls.iter().map(|u| (u.as_str(), SETTLED));
        let s = sample(
            workers,
            &readings(&[("http://p0", Fresh(0))]),
            20_000,
            Instant::now(),
            0,
        );
        assert_eq!(s.overdue_workers, 9);
        assert_eq!(s.unreported_workers, 0);
        assert_eq!(s.cold_tokens, 180_000);
        assert_eq!(s.capacity_tokens, 200_000);
        assert!((s.pressure - 0.9).abs() < 1e-9, "{}", s.pressure);
        assert!(s.has_data());
    }

    /// A busy engine's report goes stale; it must not drop out as idle.
    #[test]
    fn stale_workers_count_as_at_least_full() {
        let s = sample(
            pool(
                &["http://idle", "http://busy", "http://overloaded"],
                SETTLED,
            ),
            &readings(&[
                ("http://idle", Fresh(0)),
                ("http://busy", Stale(5_000)),
                ("http://overloaded", Stale(50_000)),
            ]),
            20_000,
            Instant::now(),
            0,
        );
        assert_eq!(s.stale_workers, 2);
        assert_eq!(s.cold_tokens, 70_000);
        assert_eq!(s.capped_cold_tokens, 40_000);
        assert_eq!(s.capacity_tokens, 60_000);
        assert!(s.has_data());
    }

    #[test]
    fn a_pool_that_never_reported_has_no_data() {
        for pooled_for in [JUST_JOINED, SETTLED] {
            let workers = pool(&["http://a", "http://b"], pooled_for);
            let s = sample(workers, &readings(&[]), 20_000, Instant::now(), 0);
            assert!(!s.has_data(), "pooled for {pooled_for:?}");
        }
    }

    #[test]
    fn no_healthy_prefill_has_no_data() {
        let s = sample(
            std::iter::empty(),
            &readings(&[]),
            20_000,
            Instant::now(),
            0,
        );
        assert_eq!(s.healthy_prefill_workers, 0);
        assert!(!s.has_data());
    }

    #[test]
    fn reports_from_workers_outside_the_pool_are_ignored() {
        let s = sample(
            pool(&["http://a"], SETTLED),
            &readings(&[("http://a", Fresh(1_000)), ("http://decode", Fresh(90_000))]),
            20_000,
            Instant::now(),
            0,
        );
        assert_eq!(s.cold_tokens, 1_000);
    }

    #[test]
    fn pressure_saturates_at_one_while_cold_tokens_keep_the_overload() {
        let s = sample(
            pool(&["http://a"], SETTLED),
            &readings(&[("http://a", Fresh(30_000))]),
            20_000,
            Instant::now(),
            0,
        );
        assert_eq!(s.cold_tokens, 30_000);
        assert_eq!(s.capped_cold_tokens, 20_000);
        assert!((s.pressure - 1.0).abs() < 1e-9, "{}", s.pressure);
    }

    /// One huge request on one worker must not read as a saturated pool while
    /// the router can still send new work to the idle workers.
    #[test]
    fn one_overloaded_worker_counts_as_one_full_worker() {
        let urls: Vec<String> = (0..18).map(|i| format!("http://p{i}")).collect();
        let entries: Vec<_> = urls
            .iter()
            .enumerate()
            .map(|(i, u)| (u.as_str(), Fresh(if i == 0 { 400_000 } else { 0 })))
            .collect();
        let workers = urls.iter().map(|u| (u.as_str(), SETTLED));
        let s = sample(workers, &readings(&entries), 20_000, Instant::now(), 0);
        assert_eq!(s.cold_tokens, 400_000);
        assert_eq!(s.capped_cold_tokens, 20_000);
        assert_eq!(s.capacity_tokens, 360_000);
        assert!((s.pressure - 1.0 / 18.0).abs() < 1e-9, "{}", s.pressure);
    }

    #[test]
    fn latest_is_none_until_stored() {
        let pressure = ColdPressure::default();
        assert!(pressure.latest().is_none());
        let s = sample(
            pool(&["http://a"], SETTLED),
            &readings(&[("http://a", Fresh(5))]),
            10,
            Instant::now(),
            0,
        );
        pressure.store(s);
        assert_eq!(pressure.latest().unwrap().pressure, 0.5);
    }

    /// A worker with one rank still silent keeps its other ranks' backlog
    /// instead of reading as idle or as unreported.
    #[test]
    fn a_partially_reported_worker_keeps_its_known_backlog() {
        use crate::state::load_monitor::engine_reported_load::{LoadStat, NativeCacheRankLoad};
        let stat = |cold| LoadStat {
            num_running_reqs: 0,
            num_waiting_reqs: 0,
            num_tokens: 0,
            max_total_num_tokens: 1,
            native_cache: Some(NativeCacheRankLoad {
                num_waiting_uncached_tokens: cold,
                num_total_tokens: 0,
                max_running_requests: 1,
                total_prefill_uncached_tokens: 0,
                total_prefill_busy_us: 0,
                total_prefill_input_tokens: None,
            }),
        };
        let table = EngineReportedLoadTable::new();
        let now = Instant::now();
        table.mark_expected_rank("http://a", 0);
        table.set("http://a", 0, stat(0), now);
        table.mark_expected_rank("http://b", 0);
        table.mark_expected_rank("http://b", 1);
        table.set("http://b", 0, stat(200_000), now);

        for pooled_for in [JUST_JOINED, SETTLED] {
            let s = sample(
                pool(&["http://a", "http://b"], pooled_for),
                &table.cold_token_readings(now),
                20_000,
                Instant::now(),
                0,
            );
            let past_grace = usize::from(pooled_for >= UNREPORTED_GRACE);
            assert_eq!(s.stale_workers, past_grace, "pooled for {pooled_for:?}");
            assert_eq!(s.unreported_workers + s.overdue_workers, 0);
            assert_eq!(s.cold_tokens, 200_000);
            assert!((s.pressure - 0.5).abs() < 1e-9, "{}", s.pressure);
        }
    }

    /// A new DP worker whose ranks report a moment apart must not read as full
    /// during the grace; past it, a silent rank means a broken publisher.
    #[test]
    fn a_new_worker_with_ranks_still_coming_up_is_not_floored() {
        let workers = [("http://old", SETTLED), ("http://new", JUST_JOINED)];
        let entries = [("http://old", Fresh(10_000)), ("http://new", Partial(0))];
        let s = sample(workers, &readings(&entries), 20_000, Instant::now(), 0);
        assert_eq!(s.stale_workers, 0);
        assert_eq!(s.capacity_tokens, 40_000);
        assert!((s.pressure - 0.25).abs() < 1e-9, "{}", s.pressure);

        let workers = [("http://old", SETTLED), ("http://new", SETTLED)];
        let s = sample(workers, &readings(&entries), 20_000, Instant::now(), 0);
        assert_eq!(s.stale_workers, 1);
        assert!((s.pressure - 0.75).abs() < 1e-9, "{}", s.pressure);
    }
}
