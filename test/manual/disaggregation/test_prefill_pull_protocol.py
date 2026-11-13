"""CPU integration: real Rust router, Python pull agent and admission ledger.

Build sgl-router first and set SGL_ROUTER_BINARY to its executable. Only the
GPU execution/transfer boundary is simulated; control traffic uses real HTTP.
This manual test needs both language toolchains, unlike the CPU unit suite.
"""

import asyncio
import contextlib
import json
import os
import socket
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import aiohttp
from aiohttp import web
from sglang.srt.managers.prefill_pull_agent import (
    PrefillPullAgent,
    PrefillPullAgentConfig,
)
from sglang.srt.managers.scheduler_components.prefill_work_admission import (
    PrefillCapacity,
    PrefillWorkAdmission,
)
from sglang.test.test_utils import CustomTestCase


class TestPrefillPullProtocol(unittest.IsolatedAsyncioTestCase, CustomTestCase):
    async def asyncSetUp(self):
        self.addAsyncCleanup(self.cleanup)
        self.runners = []
        self.router = None
        self.agent = None
        self.tasks = []
        self.log = self.enterContext(tempfile.TemporaryFile(mode="w+"))
        self.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        self.ledger = PrefillWorkAdmission(max_tokens=16, max_requests=1)
        self.prefills = []
        self.decodes = []
        self.aborts = []
        self.executing = {}
        self.completed = {}
        self.allow_finish = asyncio.Event()
        self.allow_finish.set()
        self.hold_retirement = False
        self.prefill_status = 200
        self.p_url = await self.worker("prefill")
        self.d_url = await self.worker("decode")
        binary = Path(
            os.environ.get(
                "SGL_ROUTER_BINARY",
                Path(__file__).resolve().parents[3]
                / "sgl-router/target/debug/sgl-router",
            )
        )
        self.assertTrue(binary.is_file(), f"Build the router first: {binary}")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.router_url = f"http://127.0.0.1:{port}"
        self.router = await asyncio.create_subprocess_exec(
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--model-id",
            "tiny",
            "--no-tokenizer",
            "--worker-urls",
            self.p_url,
            self.d_url,
            "--chat-routing",
            "reorg",
            "--prefill-dispatch-mode",
            "pull",
            "--worker-api-key",
            "pull-test-secret",
            "--pd-stream-after-prefill",
            "--request-timeout-secs",
            "10",
            "--shutdown-drain-secs",
            "0",
            stdout=self.log,
            stderr=self.log,
        )
        async with asyncio.timeout(15):
            while True:
                if self.router.returncode is not None:
                    self.log.seek(0)
                    self.fail(self.log.read())
                try:
                    async with self.http.get(self.router_url + "/readyz") as response:
                        if response.status == 200:
                            break
                except aiohttp.ClientError:
                    pass
                await asyncio.sleep(0.02)

    async def cleanup(self):
        self.allow_finish.set()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        if self.agent is not None:
            await self.agent.close()
        if self.router is not None and self.router.returncode is None:
            self.router.terminate()
            try:
                await asyncio.wait_for(self.router.wait(), 3)
            except TimeoutError:
                self.router.kill()
                await self.router.wait()
        await self.http.close()
        for runner in self.runners:
            await runner.cleanup()

    async def worker(self, mode):
        async def info(request):
            return web.json_response(
                {
                    "served_model_name": "tiny",
                    "disaggregation_mode": mode,
                    "disaggregation_bootstrap_port": 8997,
                    "dp_size": 1,
                    "optimistic_prefill_attempts": 0,
                    "disaggregation_decode_allocation_policy": "early",
                }
            )

        async def generate(request):
            body = await request.json()
            rid = body["rid"]
            if mode == "prefill":
                error = self.ledger.consume(
                    reservation_id=body.get("prefill_pull_reservation_id"),
                    assignment_id=body.get("prefill_pull_assignment_id"),
                    request_id=rid,
                    tokens=len(body["input_ids"]),
                    now=time.monotonic(),
                )
                if error:
                    return web.json_response({"error": error}, status=403)
                req = SimpleNamespace(
                    kv=SimpleNamespace(kv_allocated_len=0, req_pool_idx=None),
                    metadata_buffer_index=-1,
                )
                self.ledger.attach_request(body["prefill_pull_reservation_id"], req)
                self.executing[rid] = req
                self.prefills.append(body)
                await self.allow_finish.wait()
                if not self.hold_retirement:
                    self.ledger.retire(req)
                self.completed.setdefault(rid, asyncio.Event()).set()
                return web.json_response(
                    {"error": "prefill rejected"}
                    if self.prefill_status != 200
                    else {"text": "", "meta_info": {"id": rid}},
                    status=self.prefill_status,
                )
            self.decodes.append(body)
            await self.completed.setdefault(rid, asyncio.Event()).wait()
            result = {"text": "decoded", "meta_info": {"id": rid}}
            if body.get("stream"):
                return web.Response(
                    text=f"data: {json.dumps(result)}\n\ndata: [DONE]\n\n",
                    content_type="text/event-stream",
                )
            return web.json_response(result)

        async def abort(request):
            self.abort((await request.json())["rid"])
            return web.json_response({})

        app = web.Application()
        app.router.add_get("/server_info", info)
        app.router.add_get("/model_info", info)
        app.router.add_get("/health", info)
        app.router.add_post("/generate", generate)
        app.router.add_post("/abort_request", abort)
        runner = web.AppRunner(app, shutdown_timeout=0.2)
        await runner.setup()
        self.runners.append(runner)
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        return f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"

    def abort(self, rid):
        # An abort notification deliberately does not release in-flight KV.
        self.aborts.append(rid)

    async def start_agent(self):
        async def budget(**kwargs):
            return self.ledger.control(
                **kwargs,
                now=time.monotonic(),
                capacity=PrefillCapacity(tokens=128, requests=4, metadata=4),
            )

        self.agent = PrefillPullAgent(
            PrefillPullAgentConfig(
                router_url=self.router_url,
                worker_url=self.p_url,
                local_url=self.p_url,
                api_key="pull-test-secret",
                router_api_key="pull-test-secret",
                max_inflight=2,
                poll_wait_ms=50,
                retry_interval=0.01,
                status_interval=0.02,
            ),
            budget=budget,
            abort=self.abort,
            ready=lambda: True,
        )
        await self.agent.start()

    def generate(self, *, stream=False):
        async def send():
            async with self.http.post(
                self.router_url + "/generate",
                json={
                    "input_ids": [1, 2, 3],
                    "sampling_params": {"max_new_tokens": 1},
                    "stream": stream,
                },
            ) as response:
                return response.status, await response.text()

        task = asyncio.create_task(send())
        self.tasks.append(task)
        return task

    async def wait_for(self, predicate):
        async with asyncio.timeout(5):
            while not predicate():
                await asyncio.sleep(0.01)

    async def test_pull_pairing_stream_and_safe_capacity_retirement(self):
        self.hold_retirement = True
        first = self.generate(stream=True)
        await asyncio.sleep(0.1)
        self.assertFalse(first.done())
        self.assertEqual((self.prefills, self.decodes), ([], []))
        await self.start_agent()
        status, body = await first
        self.assertEqual(status, 200)
        self.assertIn("data: [DONE]", body)
        self.assertIn("decoded", body)
        p, d = self.prefills[0], self.decodes[0]
        self.assertEqual(p["rid"], d["rid"])
        self.assertEqual(p["bootstrap_room"], d["bootstrap_room"])
        self.assertFalse(p["stream"])
        self.assertEqual(d["disagg_prefill_dp_rank"], 0)
        second = self.generate()
        await asyncio.sleep(0.15)
        self.assertFalse(second.done())
        self.assertEqual(len(self.prefills), 1)
        self.assertEqual(len(self.ledger.reservations), 1)
        self.hold_retirement = False
        self.ledger.retire(self.executing[p["rid"]])
        self.assertEqual((await second)[0], 200)
        self.assertEqual(len(self.prefills), 2)

    async def test_prefill_error_reaches_original_client(self):
        self.prefill_status = 422
        await self.start_agent()
        status, body = await self.generate()
        self.assertEqual(status, 422)
        self.assertEqual(json.loads(body), {"error": "prefill rejected"})

    async def test_disconnect_cancels_without_replaying_or_releasing_kv(self):
        self.allow_finish.clear()
        await self.start_agent()
        request = self.generate()
        await self.wait_for(lambda: bool(self.prefills))
        rid = self.prefills[0]["rid"]
        request.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await request
        await self.wait_for(lambda: rid in self.aborts)
        self.assertEqual(len(self.ledger.reservations), 1)
        next_request = self.generate()
        await asyncio.sleep(0.15)
        self.assertFalse(next_request.done())
        self.assertEqual(len(self.prefills), 1)
        self.allow_finish.set()
        self.assertEqual((await next_request)[0], 200)
        self.assertNotEqual(self.prefills[1]["rid"], rid)


if __name__ == "__main__":
    unittest.main()
