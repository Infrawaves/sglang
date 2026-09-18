"""CPU-only checks for the benchmark's measurement and validity gates."""

import importlib.util
import ast
import contextlib
import copy
import io
import json
import tempfile
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import Mock, patch


class BenchmarkChecks(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).with_name("bench_gpu_prefix.py")
        self.assertTrue(path.exists(), "GPU-prefix benchmark is not implemented")
        spec = importlib.util.spec_from_file_location("bench_gpu_prefix", path)
        self.bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.bench)

    def test_distinct_suffixes_preserve_exact_shared_prefix(self):
        rows = self.bench.make_workload(list(range(100, 200)), 1024, 64, 8)
        self.assertEqual(
            rows, self.bench.make_workload(list(range(100, 200)), 1024, 64, 8)
        )
        self.assertEqual(len({row[1024] for row in rows}), 8)
        self.assertTrue(all(row[:1024] == rows[0][:1024] for row in rows))
        self.assertTrue(all(len(row) == 1088 for row in rows))

    def test_host_hit_is_rejected(self):
        meta = {
            "prompt_tokens": 1088,
            "completion_tokens": 8,
            "cached_tokens": 1024,
            "cached_tokens_details": {"device": 512, "host": 512},
            "finish_reason": {"type": "length"},
        }
        with self.assertRaisesRegex(ValueError, "host|GPU"):
            self.bench.validate_meta(meta, 1024, 64, 8)

    def test_missing_cache_breakdown_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cache|缓存"):
            self.bench.validate_meta(
                {"prompt_tokens": 1088, "completion_tokens": 8}, 1024, 64, 8
            )

    def test_real_gpu_hit_is_accepted_but_duplicate_suffix_is_rejected(self):
        meta = {
            "prompt_tokens": 1088,
            "completion_tokens": 8,
            "cached_tokens": 1024,
            "cached_tokens_details": {"device": 1024, "host": 0},
            "finish_reason": {"type": "length"},
        }
        self.bench.validate_meta(meta, 1024, 64, 8)
        meta["cached_tokens_details"]["device"] = 1088
        meta["cached_tokens"] = 1088
        with self.assertRaises(ValueError):
            self.bench.validate_meta(meta, 1024, 64, 8)

    def test_stream_ignores_metadata_only_frames_and_requires_done(self):
        lines = [
            b'data: {"meta_info": {"completion_tokens": 0}}',
            b'data: {"meta_info": {"completion_tokens": 1}, "text": "x"}',
            b'data: {"meta_info": {"completion_tokens": 8, "finish_reason": {"type": "length"}}, "text": "xyz"}',
            b"data: [DONE]",
        ]
        times = iter([1.0, 2.0, 3.0, 4.0])
        result = self.bench.consume_stream(lines, 0.0, clock=lambda: next(times))
        self.assertEqual(result["ttft_ms"], 2000.0)
        self.assertEqual(result["meta"]["completion_tokens"], 8)
        with self.assertRaisesRegex(ValueError, "DONE|incomplete"):
            self.bench.consume_stream(lines[:-1], 0.0, clock=lambda: 1.0)

    def test_null_cache_source_is_not_treated_as_zero(self):
        meta = {"cached_tokens_details": {"device": 1024, "host": None}}
        with self.assertRaisesRegex(ValueError, "cache"):
            self.bench.validate_meta(meta, 1024, 64, 8)

    def test_transfer_bandwidth_uses_union_and_excludes_failures(self):
        text = "\n".join(
            f"PD_TRANSFER pid=1 pp=0 peer=host bytes={size} descriptors=1 "
            f"start_ns={start} end_ns={end} send_ms=1 effective_GBps=1 ret={ret}"
            for size, start, end, ret in (
                (1_000_000_000, 0, 1_000_000_000, 0),
                (1_000_000_000, 500_000_000, 1_500_000_000, 0),
                (1_000_000_000, 2_000_000_000, 3_000_000_000, 0),
                (9_000_000_000, 0, 9_000_000_000, -1),
            )
        )
        metrics = self.bench.transfer_metrics(text)
        self.assertEqual(metrics["successful_bytes"], 3_000_000_000)
        self.assertEqual(metrics["failed_calls"], 1)
        self.assertEqual(metrics["active_union_ms"], 2500)
        self.assertAlmostEqual(metrics["active_union_GBps"], 1.2)
        self.assertEqual(metrics["span_GBps"], 1)
        self.assertEqual(metrics["call_effective_GBps_median"], 1)
        with self.assertRaisesRegex(ValueError, "failed calls"):
            self.bench.validate_evidence("A", {"transfer_metrics": metrics})

    def test_transfer_instrumentation_preserves_transport_result(self):
        # Load the actual method without importing the GPU serving runtime.
        path = (
            Path(__file__).resolve().parents[1]
            / "python/sglang/srt/disaggregation/mooncake/conn.py"
        )
        tree = ast.parse(path.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "MooncakeKVManager"
        )
        method = next(
            n for n in cls.body if getattr(n, "name", None) == "_transfer_data"
        )
        logger = Mock()
        clock = Mock(side_effect=[0, 1_000_000_000])
        scope = {
            "time": types.SimpleNamespace(monotonic_ns=clock),
            "logger": logger,
            "os": types.SimpleNamespace(getpid=lambda: 123),
        }
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
            scope,
        )
        manager = types.SimpleNamespace(pp_rank=0, engine=Mock(), log_dcp_details=True)
        manager.engine.batch_transfer_sync.return_value = 0
        blocks = [(100, 200, 1_000_000_000)]
        send = scope["_transfer_data"]
        self.assertEqual(send(manager, "peer", blocks), 0)
        manager.engine.batch_transfer_sync.assert_called_once_with(
            "peer", [100], [200], [1_000_000_000]
        )
        fmt, *values = logger.warning.call_args.args
        metrics = self.bench.transfer_metrics(fmt % tuple(values))
        self.assertEqual(metrics["active_union_GBps"], 1)
        logger.reset_mock()
        clock.reset_mock(side_effect=True)
        self.assertEqual(send(manager, "peer", []), 0)
        clock.assert_not_called()
        logger.warning.assert_not_called()
        clock.side_effect = [0, 1_000_000_000]
        manager.engine.batch_transfer_sync.side_effect = RuntimeError(
            "transport failed"
        )
        with self.assertRaisesRegex(RuntimeError, "transport failed"):
            send(manager, "peer", blocks)
        fmt, *values = logger.warning.call_args.args
        self.assertEqual(
            self.bench.transfer_metrics(fmt % tuple(values))["failed_calls"], 1
        )
        manager.log_dcp_details = False
        manager.engine.batch_transfer_sync.side_effect = None
        manager.engine.batch_transfer_sync.return_value = 0
        clock.reset_mock()
        logger.reset_mock()
        self.assertEqual(send(manager, "peer", blocks), 0)
        clock.assert_not_called()
        logger.warning.assert_not_called()
        manager.engine.batch_transfer_sync.return_value = -1
        self.assertEqual(send(manager, "peer", blocks), -1)
        logger.warning.assert_called_once()

    def test_chunk_summary_without_detailed_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prefill.log"
            startup = "DCP window pack PP0: actual=2 x 128 MiB window_tokens=65536\n"
            measured = "DCP window chunk room=1 windows=10 gathers=10 sends=80 tokens=524288 bytes=1048576 ret=0\n"
            path.write_text(startup + measured)
            evidence = self.bench.log_evidence(path, len(startup.encode()))
            self.bench.validate_evidence("B", evidence)
            self.assertEqual(evidence["gathers"], 10)
            self.assertEqual(evidence["sends"], 80)
            # Optional legacy detail lines must not double count summaries.
            with path.open("a") as f:
                f.write("DCP window room=1 gather_ms=1\n")
            self.assertEqual(
                self.bench.log_evidence(path, len(startup.encode()))["gathers"], 10
            )

    def test_ab_run_over_http_and_saved_comparison(self):
        # Actual HTTP/SSE and filesystem I/O; only tokenizer/model inference is fake.
        state = {"mode": "A", "count": 0, "prompts": {"A": [], "B": []}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "prefill.log"
            log.write_text("")

            class Handler(BaseHTTPRequestHandler):
                def do_POST(self):
                    body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                    self.send_response(200)
                    if self.path.startswith("/flush_cache"):
                        state["count"] = 0
                        self.end_headers()
                        return
                    payload = json.loads(body)
                    state["prompts"][state["mode"]].append(payload["input_ids"])
                    state["count"] += 1
                    with log.open("a") as f:
                        start = state["count"] * 2_000_000
                        f.write(
                            f"PD_TRANSFER pid=1 pp=0 peer=host bytes=2048 descriptors=2 "
                            f"start_ns={start} end_ns={start + 1_000_000} "
                            "send_ms=1 effective_GBps=0.002048 ret=0\n"
                        )
                        if state["mode"] == "A":
                            f.write(
                                "PD DCP pack buffer too small; falling back to per-token RDMA\n"
                            )
                        else:
                            f.write("DCP window room=1 gather_ms=1\n")
                            f.write(
                                "DCP window room=1 rank=0 descriptors=3 send_ms=2 ret=0\n"
                            )
                            f.write(
                                "DCP window room=1 tokens=2048 bytes=4096 elapsed_ms=4\n"
                            )
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    hit = 0 if state["count"] == 1 else 1024
                    meta = {
                        "prompt_tokens": len(payload["input_ids"]),
                        "completion_tokens": 8,
                        "cached_tokens": hit,
                        "cached_tokens_details": {"device": hit, "host": 0},
                        "finish_reason": {"type": "length"},
                    }
                    self.wfile.write(
                        b'data: {"meta_info": {"completion_tokens": 0}}\n\n'
                    )
                    self.wfile.write(
                        b"data: "
                        + json.dumps({"meta_info": meta, "text": "ok"}).encode()
                        + b"\n\n"
                    )
                    self.wfile.write(b"data: [DONE]\n\n")

                def log_message(self, *args):
                    pass

            server = HTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            tokenizer = types.SimpleNamespace(
                encode=lambda *a, **k: list(range(100, 140)), all_special_ids=[]
            )
            transformers = types.ModuleType("transformers")
            transformers.AutoTokenizer = types.SimpleNamespace(
                from_pretrained=lambda *a, **k: tokenizer
            )
            args = types.SimpleNamespace(
                prefill_log=str(log),
                results=str(root),
                model="fake",
                prefix_k=1,
                suffix_k=1,
                requests=2,
                output_tokens=8,
                profile_prefill=None,
                base=f"http://127.0.0.1:{server.server_port}",
                mode="A",
            )
            try:
                with (
                    patch.dict("sys.modules", {"transformers": transformers}),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.bench.run(args)
                    args.mode = state["mode"] = "B"
                    log.write_text(
                        "DCP window pack PP0: actual=2 x 128 MiB window_tokens=65536\n"
                    )
                    self.bench.run(args)
                    with self.assertRaisesRegex(ValueError, "already exists"):
                        self.bench.run(args)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
            self.assertEqual(state["prompts"]["A"], state["prompts"]["B"])
            self.assertEqual(len(state["prompts"]["A"]), 4)
            a = json.loads((root / "A-gpu-prefix.json").read_text())
            b = json.loads((root / "B-gpu-prefix.json").read_text())
            self.assertTrue(a["valid"] and b["valid"])
            self.assertEqual(a["summary"]["completed"], 2)
            self.assertEqual(a["log_evidence"]["fallbacks"], 2)
            self.assertEqual(b["log_evidence"]["windows"], 2)
            self.assertEqual(
                b["log_evidence"]["transfer_metrics"]["successful_bytes"], 4096
            )
            self.assertTrue((root / "A-gpu-prefix.prefill.log").exists())
            self.assertEqual(
                json.loads((root / "comparison.json").read_text())["requests"], 2
            )
            bad = copy.deepcopy(b)
            bad["samples"].pop()
            with self.assertRaisesRegex(ValueError, "sample count"):
                self.bench.compare(a, bad)
            bad = copy.deepcopy(b)
            bad["samples"][0]["meta"]["cached_tokens_details"]["device"] = 960
            bad["samples"][0]["meta"]["cached_tokens"] = 960
            with self.assertRaisesRegex(ValueError, "cache hits differ"):
                self.bench.compare(a, bad)
            bad = copy.deepcopy(b)
            bad["profiled"] = True
            with self.assertRaisesRegex(ValueError, "Profiled"):
                self.bench.compare(a, bad)
            with contextlib.redirect_stdout(io.StringIO()):
                for sample in a["samples"]:
                    sample["ttft_ms"] = 200
                for sample in b["samples"]:
                    sample["ttft_ms"] = 100
                result = self.bench.compare(a, b)
            self.assertEqual(result["ttft_reduction_pct"], 50)
            self.assertEqual(result["speedup"], 2)


if __name__ == "__main__":
    unittest.main()
