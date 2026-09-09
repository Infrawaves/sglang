"""CPU-only tests for the prefill CUDA-graph warmup manifest.

The manifest is deliberately metadata-only. These tests exercise the state
machine that arms a one-warmup probe after a successful two-warmup capture,
validates the faster policy, and permanently disables it after a failure.
"""

import json
import tempfile
import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.runner.prefill_cuda_graph_cache import (
    CACHED_PREFILL_CUDA_GRAPH_WARMUP_ITERATIONS,
    DEFAULT_PREFILL_CUDA_GRAPH_WARMUP_ITERATIONS,
    PREFILL_CUDA_GRAPH_PROBE_DISABLED,
    PREFILL_CUDA_GRAPH_PROBE_PENDING,
    PREFILL_CUDA_GRAPH_PROBE_VALIDATED,
    PrefillCudaGraphCache,
    _without_rank_identity,
    fingerprint_digest,
    shape_key_id,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    _uses_persistent_prefill_graph_cache,
)
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.tc_piecewise_cuda_graph_backend import (
    TcPiecewiseCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


_SHAPES = ({"size": 4}, {"size": 8})
_FINGERPRINT = {"backend": "breakable", "model": "unit-test"}


def _make_cache(cache_dir: str, *, rank: int = 0) -> PrefillCudaGraphCache:
    return PrefillCudaGraphCache(
        fingerprint=_FINGERPRINT,
        expected_shapes=_SHAPES,
        cache_dir=cache_dir,
        rank=rank,
    )


def _capture(
    cache: PrefillCudaGraphCache, warmup_iterations: int
) -> PrefillCudaGraphCache:
    cache.begin_capture()
    for shape in _SHAPES:
        cache.record_shape(shape, warmup_iterations=warmup_iterations)
    assert cache.commit()
    return cache


def _read_payload(cache: PrefillCudaGraphCache) -> dict:
    return json.loads(cache.path.read_text(encoding="utf-8"))


def _rewrite_payload(cache: PrefillCudaGraphCache, payload: dict) -> None:
    cache.path.write_text(json.dumps(payload), encoding="utf-8")


class TestPrefillCudaGraphCache(CustomTestCase):
    def test_two_warmups_arm_a_one_warmup_probe(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            first = _capture(_make_cache(cache_dir), 2)
            payload = _read_payload(first)
            self.assertEqual(
                payload["recommended_warmup_iterations"],
                DEFAULT_PREFILL_CUDA_GRAPH_WARMUP_ITERATIONS,
            )
            self.assertEqual(payload["probe_status"], PREFILL_CUDA_GRAPH_PROBE_PENDING)
            self.assertEqual(
                payload["next_warmup_iterations"],
                CACHED_PREFILL_CUDA_GRAPH_WARMUP_ITERATIONS,
            )

            next_launch = _make_cache(cache_dir)
            self.assertTrue(next_launch.hit)
            self.assertEqual(
                next_launch.warmup_iterations,
                CACHED_PREFILL_CUDA_GRAPH_WARMUP_ITERATIONS,
            )
            self.assertEqual(next_launch.probe_status, PREFILL_CUDA_GRAPH_PROBE_PENDING)

    def test_successful_probe_becomes_validated(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            _capture(_make_cache(cache_dir), 2)
            probe = _make_cache(cache_dir)
            self.assertEqual(probe.warmup_iterations, 1)
            _capture(probe, 1)

            payload = _read_payload(probe)
            self.assertEqual(payload["recommended_warmup_iterations"], 1)
            self.assertEqual(
                payload["probe_status"], PREFILL_CUDA_GRAPH_PROBE_VALIDATED
            )
            self.assertIsNone(payload["next_warmup_iterations"])
            validated = _make_cache(cache_dir)
            self.assertTrue(validated.hit)
            self.assertEqual(validated.warmup_iterations, 1)

    def test_probe_failure_is_persistently_disabled(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            cache = _capture(_make_cache(cache_dir), 2)
            cache.mark_runtime_fallback("unit-test failure")
            payload = _read_payload(cache)
            self.assertEqual(payload["status"], "disabled")
            self.assertEqual(payload["probe_status"], PREFILL_CUDA_GRAPH_PROBE_DISABLED)

            disabled = _make_cache(cache_dir)
            self.assertFalse(disabled.hit)
            self.assertEqual(disabled.warmup_iterations, 2)
            self.assertTrue(disabled._probe_disabled)

            # A later successful two-warmup capture retains the disabled state.
            _capture(disabled, 2)
            disabled_again = _make_cache(cache_dir)
            self.assertFalse(disabled_again.hit)
            self.assertEqual(disabled_again.warmup_iterations, 2)
            self.assertEqual(
                disabled_again.probe_status, PREFILL_CUDA_GRAPH_PROBE_DISABLED
            )

    def test_manifest_rejects_recommendation_record_mismatch(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            cache = _capture(_make_cache(cache_dir), 2)
            payload = _read_payload(cache)
            payload["recommended_warmup_iterations"] = 1
            _rewrite_payload(cache, payload)
            loaded = _make_cache(cache_dir)
            self.assertFalse(loaded.hit)
            self.assertIn("recommendation", loaded.miss_reason)

    def test_manifest_rejects_identity_mismatches(self):
        for field, value, reason in (
            ("code_version", "old-code", "code version"),
            ("rank", 7, "rank"),
            ("coordination_digest", "wrong", "coordination"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as cache_dir:
                cache = _capture(_make_cache(cache_dir), 2)
                payload = _read_payload(cache)
                payload[field] = value
                _rewrite_payload(cache, payload)
                loaded = _make_cache(cache_dir)
                self.assertFalse(loaded.hit)
                self.assertIn(reason, loaded.miss_reason)

    def test_coordination_digest_ignores_local_device_index(self):
        first = {
            "device": "cuda:0",
            "gpu_id": 0,
            "world_rank": 0,
            "capability": [10, 0],
        }
        second = {
            "device": "cuda:7",
            "gpu_id": 7,
            "world_rank": 7,
            "capability": [10, 0],
        }
        self.assertEqual(_without_rank_identity(first), _without_rank_identity(second))

    def test_manifest_rejects_extra_or_missing_shapes(self):
        for operation in ("extra", "missing"):
            with (
                self.subTest(operation=operation),
                tempfile.TemporaryDirectory() as cache_dir,
            ):
                cache = _capture(_make_cache(cache_dir), 2)
                payload = _read_payload(cache)
                if operation == "extra":
                    payload["shapes"][shape_key_id({"size": 16})] = {
                        "status": "captured",
                        "shape": {"size": 16},
                        "warmup_iterations": 2,
                    }
                else:
                    del payload["shapes"][shape_key_id(_SHAPES[0])]
                _rewrite_payload(cache, payload)
                loaded = _make_cache(cache_dir)
                self.assertFalse(loaded.hit)
                self.assertIn("shape record", loaded.miss_reason)

    def test_synchronize_rejects_wrong_rank_count(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            _capture(_make_cache(cache_dir), 2)
            cache = _make_cache(cache_dir)
            group = SimpleNamespace(
                world_size=2,
                all_gather_object=lambda _value: [{"hit": True}],
            )
            cache.synchronize(group)
            self.assertFalse(cache.hit)
            self.assertEqual(cache.warmup_iterations, 2)

    def test_synchronize_checks_all_participant_groups_after_a_miss(self):
        """A failed subgroup exchange must not skip later collectives."""

        with tempfile.TemporaryDirectory() as cache_dir:
            _capture(_make_cache(cache_dir), 2)
            cache = _make_cache(cache_dir)
            calls = []

            def first_group_gather(_value):
                calls.append("first")
                return [
                    {
                        "hit": True,
                        "coordination_digest": "wrong",
                        "warmup_iterations": 1,
                        "probe_status": PREFILL_CUDA_GRAPH_PROBE_PENDING,
                    },
                    {"hit": True},
                ]

            def second_group_gather(_value):
                calls.append("second")
                return [{"hit": False}, {"hit": False}]

            cache.synchronize(
                SimpleNamespace(world_size=2, all_gather_object=first_group_gather),
                participant_groups=[
                    SimpleNamespace(world_size=2, all_gather_object=second_group_gather)
                ],
            )
            self.assertEqual(calls, ["first", "second"])
            self.assertFalse(cache.hit)
            self.assertEqual(cache.warmup_iterations, 2)

    def test_coordination_digest_ignores_local_subgroup_membership(self):
        """TP ranks in a TP x PP topology must still share one verdict."""

        rank_zero = {
            "parallel": {
                "tp_size": 2,
                "tp_rank": 0,
                "pp_size": 2,
                "pp_rank": 0,
                "participant_group_topology": {
                    "tp_group": {"ranks": [0, 1], "world_size": 2},
                    "pp_group": {"ranks": [0, 2], "world_size": 2},
                },
            }
        }
        rank_one = {
            "parallel": {
                "tp_size": 2,
                "tp_rank": 1,
                "pp_size": 2,
                "pp_rank": 0,
                "participant_group_topology": {
                    "tp_group": {"ranks": [0, 1], "world_size": 2},
                    "pp_group": {"ranks": [1, 3], "world_size": 2},
                },
            }
        }
        self.assertEqual(
            fingerprint_digest(_without_rank_identity(rank_zero)),
            fingerprint_digest(_without_rank_identity(rank_one)),
        )

    def test_only_backends_with_capture_bookkeeping_use_the_manifest(self):
        """The predicate must not admit a backend lacking the accessors.

        capture_one_shape calls captured_segment_count() and reads
        last_warmup_iterations on any backend the predicate admits, so widening
        it without adding those would AttributeError mid-capture.
        """
        for backend_cls in (
            BreakableCudaGraphBackend,
            FullCudaGraphBackend,
            TcPiecewiseCudaGraphBackend,
        ):
            backend = backend_cls.__new__(backend_cls)
            if not _uses_persistent_prefill_graph_cache(backend):
                continue
            # hasattr is the assertion subject here, not defensive access.
            self.assertTrue(
                hasattr(backend, "captured_segment_count"), backend_cls.__name__
            )
            self.assertTrue(
                hasattr(backend_cls, "last_warmup_iterations")
                or "last_warmup_iterations" in backend_cls.__init__.__code__.co_names,
                backend_cls.__name__,
            )

    def test_tc_piecewise_does_not_use_manifest(self):
        self.assertFalse(
            _uses_persistent_prefill_graph_cache(
                TcPiecewiseCudaGraphBackend.__new__(TcPiecewiseCudaGraphBackend)
            )
        )
        self.assertTrue(
            _uses_persistent_prefill_graph_cache(
                BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
            )
        )


if __name__ == "__main__":
    unittest.main()
