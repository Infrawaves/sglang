"""CPU tests for the offline-analysis expert distribution dumps.

* the circular buffer returns appended steps in order;
* the analysis dump writes a step-ordered global file on rank 0 and chunked
  per-rank files, and keeps the legacy file format when disabled;
* the recorder auto-dumps every N recorded passes.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.eplb import expert_distribution as ed
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_NUM_LAYERS, _NUM_EXPERTS = 2, 4


class TestCircularBufferOrder(unittest.TestCase):
    def test_partial_and_wrapped(self):
        buf = ed._CircularBuffer((1,), 3, dtype=torch.int32, device="cpu")
        self.assertEqual(buf.get_ordered().shape[0], 0)
        for v in (1, 2):
            buf.append(torch.tensor([v], dtype=torch.int32))
        self.assertEqual(buf.get_ordered().flatten().tolist(), [1, 2])
        for v in (3, 4, 5):
            buf.append(torch.tensor([v], dtype=torch.int32))
        self.assertEqual(buf.get_ordered().flatten().tolist(), [3, 4, 5])
        buf.append(torch.tensor([6], dtype=torch.int32))
        self.assertEqual(buf.get_ordered().flatten().tolist(), [4, 5, 6])
        buf.reset()
        self.assertEqual(buf.get_ordered().shape[0], 0)


def _make_stat_accumulator(rank, buffer_size):
    acc = object.__new__(ed._StatAccumulator)
    acc._rank = rank
    acc._enable = False
    acc._expert_location_metadata = SimpleNamespace(
        num_layers=_NUM_LAYERS,
        num_logical_experts=_NUM_EXPERTS,
        num_physical_experts=_NUM_EXPERTS,
        ep_size=2,
        physical_to_logical_map=torch.arange(_NUM_EXPERTS).repeat(_NUM_LAYERS, 1),
        physical_to_logical_map_cpu=torch.arange(_NUM_EXPERTS).repeat(_NUM_LAYERS, 1),
    )
    acc._global_physical_count_of_buffered_step = ed._CircularBuffer(
        (_NUM_LAYERS, _NUM_EXPERTS), buffer_size, dtype=torch.int32, device="cpu"
    )
    acc._first_dump = True
    acc._analysis_dump = True
    acc._pass_metadata = ed.deque(maxlen=buffer_size)
    acc._analysis_dump_seq = 0
    acc.gatherer_kind = "_SelectExpertsSinglePassGatherer"
    return acc


def _env(**values):
    defaults = dict(
        SGLANG_EXPERT_DISTRIBUTION_DUMP_PER_RANK=False,
        SGLANG_EXPERT_DISTRIBUTION_PER_RANK_CHUNK_STEPS=0,
    )
    defaults.update(values)
    return {
        name: SimpleNamespace(get=(lambda v=v: v)) for name, v in defaults.items()
    }


class TestAnalysisDump(unittest.TestCase):
    def _run(self, *, per_rank, chunk, num_steps, buffer_size=4):
        torch.manual_seed(0)
        world = 2
        accs = [_make_stat_accumulator(r, buffer_size) for r in range(world)]
        steps = [
            [
                torch.randint(0, 5, (_NUM_LAYERS, _NUM_EXPERTS), dtype=torch.int32)
                for _ in range(world)
            ]
            for _ in range(num_steps)
        ]
        for i, per_rank_counts in enumerate(steps):
            for r, acc in enumerate(accs):
                acc._global_physical_count_of_buffered_step.append(per_rank_counts[r])
                acc.record_pass_metadata(forward_pass_id=100 + i, num_tokens=10 + r)

        pending = {}

        def fake_all_reduce(tensor, op=None):
            # Ranks run one after another: rank 1 stashes, rank 0 sums.
            if "other" in pending:
                tensor.add_(pending.pop("other"))
            else:
                pending["other"] = tensor.clone()

        written = {}
        envs_patch = _env(
            SGLANG_EXPERT_DISTRIBUTION_DUMP_PER_RANK=per_rank,
            SGLANG_EXPERT_DISTRIBUTION_PER_RANK_CHUNK_STEPS=chunk,
        )
        with (
            patch.multiple(ed.envs, **envs_patch),
            patch.object(ed.torch.distributed, "all_reduce", fake_all_reduce),
            patch.object(
                ed,
                "_dump_to_file_async",
                lambda name, data: written.__setitem__(name, data),
            ),
        ):
            for acc in (accs[1], accs[0]):
                acc._dump_for_analysis()
        return steps, written

    def test_global_file_is_step_ordered_and_summed(self):
        steps, written = self._run(per_rank=False, chunk=0, num_steps=6)
        names = list(written)
        self.assertEqual(len(names), 1)
        self.assertTrue(names[0].startswith("expert_distribution_dump00000_global_"))
        data = written[names[0]]
        kept = steps[-4:]  # buffer of 4 keeps the newest steps
        expected = torch.stack([s[0] + s[1] for s in kept])
        self.assertTrue(torch.equal(data["logical_count"], expected))
        self.assertEqual(data["forward_pass_ids"], [102, 103, 104, 105])
        self.assertEqual(data["num_steps"], 4)
        self.assertEqual(len(data["step_end_time"]), 4)

    def test_per_rank_files_are_chunked_local_counts(self):
        steps, written = self._run(per_rank=True, chunk=3, num_steps=4)
        rank1 = [n for n in written if "_rank001_" in n]
        self.assertEqual(len(rank1), 1)
        data = written[rank1[0]]
        local = torch.stack([s[1] for s in steps])
        expected = torch.stack([local[:3].sum(0), local[3:].sum(0)])
        self.assertTrue(torch.equal(data["local_physical_count_of_chunk"], expected))
        self.assertEqual(data["chunk_steps"], 3)
        self.assertEqual(data["local_num_tokens"], [11] * 4)
        self.assertEqual(data["gatherer"], "_SelectExpertsSinglePassGatherer")
        self.assertEqual(len([n for n in written if "_global_" in n]), 1)

    def test_legacy_file_format_when_disabled(self):
        acc = _make_stat_accumulator(0, 2)
        acc._analysis_dump = False
        acc._enable = False
        acc._global_physical_count_of_buffered_step.append(
            torch.ones((_NUM_LAYERS, _NUM_EXPERTS), dtype=torch.int32)
        )
        written = {}
        with (
            patch.object(ed.torch.distributed, "all_reduce", lambda t, op=None: None),
            patch.object(
                ed, "_dump_to_file", lambda name, data: written.__setitem__(name, data)
            ),
            patch.object(ed.torch, "get_device_module", create=True),
        ):
            acc.dump("file")
        (name,) = written
        self.assertTrue(name.startswith("expert_distribution_recorder_"))
        # Legacy format keeps the whole ring buffer, including empty slots.
        self.assertEqual(written[name]["logical_count"].shape[0], 2)


class TestAutoDump(unittest.TestCase):
    def test_dumps_every_n_recorded_passes(self):
        rec = object.__new__(ed._ExpertDistributionRecorderReal)
        rec._recording = True
        rec._record_current_pass = True
        rec._current_pass_num_tokens = 8
        rec._single_pass_gatherers = {}
        calls = []
        rec._accumulator = SimpleNamespace(
            record_pass_metadata=lambda **kw: calls.append(("meta", kw)),
        )
        rec.dump_record = lambda output_mode: calls.append(("dump", output_mode))
        rec._auto_dump_passes = 3
        rec._passes_since_auto_dump = 0
        for fp in range(7):
            rec._on_forward_pass_end(fp, {})
        dumps = [c for c in calls if c[0] == "dump"]
        self.assertEqual(dumps, [("dump", "file"), ("dump", "file")])
        self.assertEqual(rec._passes_since_auto_dump, 1)

    def test_auto_dump_disabled_with_eplb_or_non_stat(self):
        stat = object.__new__(ed._StatAccumulator)
        envs_patch = {
            "SGLANG_EXPERT_DISTRIBUTION_AUTO_DUMP_PASSES": SimpleNamespace(
                get=lambda: 5
            ),
            "SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR": SimpleNamespace(
                get=lambda: tempfile.gettempdir()
            ),
        }
        with patch.multiple(ed.envs, **envs_patch):
            for enable_eplb, acc, expected in (
                (False, stat, 5),
                (True, stat, 0),
                (False, object(), 0),
            ):
                with patch.object(
                    ed,
                    "get_exec",
                    return_value=SimpleNamespace(
                        moe=SimpleNamespace(enable_eplb=enable_eplb)
                    ),
                ):
                    self.assertEqual(ed._resolve_auto_dump_passes(acc), expected)


class TestAsyncWriter(unittest.TestCase):
    def test_writes_file(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(
                ed.envs,
                "SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR",
                SimpleNamespace(get=lambda: d),
            ):
                ed._dump_to_file_async("x.pt", {"a": torch.ones(2)})
                ed._ASYNC_DUMP_EXECUTOR.submit(lambda: None).result()
            self.assertTrue(torch.equal(torch.load(Path(d) / "x.pt")["a"], torch.ones(2)))


if __name__ == "__main__":
    unittest.main()
