// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Per-worker estimate of the fraction of dispatched prompt tokens the engine
//! actually prefills (the rest are prefix-cache hits on any tier, L3 included).
//!
//! The router counts every dispatched prompt token; the engine reports a
//! cumulative `total_prefill_uncached_tokens`. Their delta ratio over a large
//! window converts the router's raw in-flight tokens into the engine's
//! uncached unit, so the two halves of `pending_prefill_tokens` agree.

/// Dispatched tokens a window must cover before its ratio is trusted; the two
/// counters advance at different moments (dispatch vs prefill), so a short
/// window is dominated by that skew. Arbitrary; tune with load tests.
pub const COLD_RATIO_WINDOW_TOKENS: u64 = 1_000_000;
/// Weight of the newest window in the moving average. Arbitrary.
pub const COLD_RATIO_EMA_ALPHA: f64 = 0.3;
pub const COLD_RATIO_MIN: f64 = 0.01;
pub const COLD_RATIO_MAX: f64 = 1.0;

#[derive(Debug, Clone)]
pub struct ColdRatioEstimator {
    /// `(engine_uncached, router_dispatched)` at the start of the open window.
    base: Option<(u64, u64)>,
    /// `None` until the first full window closes.
    ratio: Option<f64>,
}

impl Default for ColdRatioEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl ColdRatioEstimator {
    pub fn new() -> Self {
        Self {
            base: None,
            ratio: None,
        }
    }

    /// Feeds the latest cumulative counters and returns the current ratio.
    /// Without a closed window the ratio is `1.0`, i.e. raw tokens unchanged.
    pub fn observe(&mut self, engine_uncached: u64, router_dispatched: u64) -> f64 {
        match self.base {
            // A counter going backwards means an engine restart: rebaseline.
            Some((e0, d0)) if engine_uncached >= e0 && router_dispatched >= d0 => {
                let dispatched = router_dispatched - d0;
                if dispatched >= COLD_RATIO_WINDOW_TOKENS {
                    let sample = ((engine_uncached - e0) as f64 / dispatched as f64)
                        .clamp(COLD_RATIO_MIN, COLD_RATIO_MAX);
                    self.ratio = Some(match self.ratio {
                        Some(prev) => prev + COLD_RATIO_EMA_ALPHA * (sample - prev),
                        None => sample,
                    });
                    self.base = Some((engine_uncached, router_dispatched));
                }
            }
            _ => self.base = Some((engine_uncached, router_dispatched)),
        }
        self.current()
    }

    pub fn current(&self) -> f64 {
        self.ratio.unwrap_or(COLD_RATIO_MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: u64 = COLD_RATIO_WINDOW_TOKENS;

    #[test]
    fn unchanged_until_a_full_window_closes() {
        let mut est = ColdRatioEstimator::new();
        assert_eq!(est.observe(0, 0), 1.0);
        assert_eq!(est.observe(W / 20, W - 1), 1.0);
        let r = est.observe(W / 20, W);
        assert!((r - 0.05).abs() < 1e-9, "{r}");
    }

    #[test]
    fn later_windows_move_the_ratio_by_the_ema_weight() {
        let mut est = ColdRatioEstimator::new();
        est.observe(0, 0);
        est.observe(W / 10, W); // 0.1
        let r = est.observe(W / 10 + W / 2, 2 * W); // sample 0.5
        let want = 0.1 + COLD_RATIO_EMA_ALPHA * (0.5 - 0.1);
        assert!((r - want).abs() < 1e-9, "{r} vs {want}");
    }

    #[test]
    fn samples_are_clamped() {
        let mut est = ColdRatioEstimator::new();
        est.observe(0, 0);
        assert_eq!(est.observe(0, W), COLD_RATIO_MIN);
        let mut est = ColdRatioEstimator::new();
        est.observe(0, 0);
        assert_eq!(est.observe(3 * W, W), COLD_RATIO_MAX);
    }

    #[test]
    fn engine_restart_rebaselines_and_keeps_the_last_ratio() {
        let mut est = ColdRatioEstimator::new();
        est.observe(10 * W, 0);
        est.observe(10 * W + W / 10, W); // 0.1

        // Engine counter reset to near zero.
        assert!((est.observe(5, 2 * W) - 0.1).abs() < 1e-9);
        // Next window measures from the new base.
        let r = est.observe(5 + W / 10, 3 * W);
        assert!((r - 0.1).abs() < 1e-9, "{r}");
    }
}
