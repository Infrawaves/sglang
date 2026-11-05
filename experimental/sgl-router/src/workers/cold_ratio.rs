// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Per-worker fraction of prompt tokens the engine actually prefills (the rest
//! are prefix-cache hits on any tier, L3 included). Converts raw in-flight
//! tokens into the unit of the engine's `num_waiting_uncached_tokens`.
//!
//! Both counters come from the engine, so the ratio covers every router
//! replica feeding the worker.

/// Input tokens a window must cover before its ratio is trusted. Both counters
/// advance at prefill time, so only chunk boundaries skew a short window.
/// Arbitrary; tune with load tests.
pub const COLD_RATIO_WINDOW_TOKENS: u64 = 200_000;
/// Weight of the newest window in the moving average. Arbitrary.
pub const COLD_RATIO_EMA_ALPHA: f64 = 0.3;
pub const COLD_RATIO_MIN: f64 = 0.01;
pub const COLD_RATIO_MAX: f64 = 1.0;

#[derive(Debug, Clone)]
pub struct ColdRatioEstimator {
    /// `(engine_uncached, engine_input)` at the start of the open window.
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

    /// Feeds the engine's latest cumulative counters and returns the current
    /// ratio. Without a closed window the ratio is `1.0`, i.e. raw tokens
    /// unchanged.
    pub fn observe(&mut self, engine_uncached: u64, engine_input: u64) -> f64 {
        match self.base {
            // A counter going backwards means an engine restart: rebaseline.
            Some((u0, i0)) if engine_uncached >= u0 && engine_input >= i0 => {
                let input = engine_input - i0;
                if input >= COLD_RATIO_WINDOW_TOKENS {
                    let sample = ((engine_uncached - u0) as f64 / input as f64)
                        .clamp(COLD_RATIO_MIN, COLD_RATIO_MAX);
                    self.ratio = Some(match self.ratio {
                        Some(prev) => prev + COLD_RATIO_EMA_ALPHA * (sample - prev),
                        None => sample,
                    });
                    self.base = Some((engine_uncached, engine_input));
                }
            }
            _ => self.base = Some((engine_uncached, engine_input)),
        }
        self.current()
    }

    pub fn current(&self) -> f64 {
        self.ratio.unwrap_or(COLD_RATIO_MAX)
    }

    /// The measured ratio, or `None` until the first window closes.
    pub fn measured(&self) -> Option<f64> {
        self.ratio
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
        est.observe(10 * W, 100 * W);
        est.observe(10 * W + W / 10, 101 * W); // 0.1

        // Engine counters reset to near zero.
        assert!((est.observe(5, 50) - 0.1).abs() < 1e-9);
        // Next window measures from the new base.
        let r = est.observe(5 + W / 10, 50 + W);
        assert!((r - 0.1).abs() < 1e-9, "{r}");
    }
}
