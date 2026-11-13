"""Pull protocol invariants across real HTTP and scheduler-control boundaries.

This fixture needs neither a model nor an inference server. The lightweight
HTTP peers exercise lost responses and cancellation during local tokenization.
"""

import asyncio
import unittest

from aiohttp import web
from sglang.srt.managers.prefill_pull_agent import (
    PrefillPullAgent,
    PrefillPullAgentConfig,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPrefillPullAgent(unittest.IsolatedAsyncioTestCase, CustomTestCase):
    async def asyncSetUp(self):
        self.agent = None
        self.reservations = {}
        self.poll_ids = []
        self.execution_bodies = []
        self.outcomes = []
        self.state = "offered"
        self.commit_state = "committed"
        self.lose_offer = False
        self.lose_commit = False
        self.fail_status = False
        self.fail_outcome = False
        self.delay_intake = False
        self.assignment_offered = False
        self.intake_started = asyncio.Event()
        self.intake_continue = asyncio.Event()
        self.intake_finished = asyncio.Event()
        self.longpoll_started = asyncio.Event()
        self.longpoll_continue = asyncio.Event()
        self.aborted = asyncio.Event()
        self.outcome_attempted = asyncio.Event()
        self.outcome_finished = asyncio.Event()
        self.reservation_released = asyncio.Event()
        self.assignment_rejected = asyncio.Event()
        self.abort_states = []
        self.executed = []
        self.local_auth = []
        self.router_auth = []
        self.local_response = b"raw prefill error\nwith details"
        app = web.Application()
        app.router.add_post("/generate", self._generate)
        app.router.add_post("/internal/pull/{action}", self._control)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"

    async def asyncTearDown(self):
        if self.agent is not None:
            await self.agent.close()
        self.intake_continue.set()
        self.longpoll_continue.set()
        await self.runner.cleanup()

    async def _start(self, max_inflight=1, max_response_bytes=4 * 1024 * 1024):
        self.agent = PrefillPullAgent(
            PrefillPullAgentConfig(
                router_url=self.url,
                worker_url="http://registered-prefill:30000",
                local_url=self.url,
                max_inflight=max_inflight,
                api_key="local-key",
                router_api_key="router-key",
                retry_interval=0.01,
                status_interval=0.01,
                control_timeout=1,
                max_response_bytes=max_response_bytes,
            ),
            budget=self._budget,
            abort=self._abort,
            ready=lambda: True,
        )
        await self.agent.start()

    async def _budget(self, *, action, reservation_id, **kwargs):
        if action == "reserve":
            self.reservations[reservation_id] = "reserved"
            return {"success": True, "data": {"max_tokens": 64}}
        if action == "bind":
            if self.reservations.get(reservation_id) != "reserved":
                return {"success": False}
            self.reservations[reservation_id] = "bound"
        elif action == "release":
            if self.reservations.get(reservation_id) in ("reserved", "bound"):
                self.reservations[reservation_id] = "revoked"
            self.reservation_released.set()
        return {"success": True, "data": {}}

    def _abort(self, request_id):
        self.abort_states.append(tuple(self.reservations.values()))
        self.aborted.set()

    async def _control(self, request):
        body = await request.json()
        self.router_auth.append(request.headers.get("Authorization"))
        action = request.match_info["action"]
        if action == "acquire":
            self.poll_ids.append(body["poll_id"])
            if self.assignment_offered and body["poll_id"] != self.poll_ids[0]:
                self.longpoll_started.set()
                await self.longpoll_continue.wait()
                return web.json_response({"status": "empty"})
            self.assignment_offered = True
            if self.lose_offer:
                self.lose_offer = False
                request.transport.close()
            return web.json_response(
                {
                    "status": "offer",
                    "assignment_id": "assignment-a",
                    "request_id": "router-engine-rid",
                    "path": "/generate",
                    "input_tokens": 3,
                    "body": {"input_ids": [1, 2, 3], "bootstrap_room": 123},
                    "headers": {"Authorization": "untrusted", "Host": "untrusted"},
                    "expires_in_ms": 30000,
                }
            )
        if action == "accept":
            self.state = self.commit_state
            if self.lose_commit:
                self.lose_commit = False
                request.transport.close()
            return web.json_response({"status": self.state})
        if action == "status":
            if self.fail_status:
                raise web.HTTPServiceUnavailable()
            return web.json_response({"status": self.state})
        if action == "reject":
            if self.state in ("offered", "committed"):
                self.state = "expired" if self.state == "offered" else "cancelled"
            self.assignment_rejected.set()
            return web.json_response({"status": self.state})
        if action == "outcome":
            self.outcomes.append(body)
            self.outcome_attempted.set()
            if self.fail_outcome:
                raise web.HTTPServiceUnavailable()
            if self.state == "committed":
                self.state = "completed"
            self.outcome_finished.set()
            return web.json_response({"status": self.state})
        raise web.HTTPNotFound()

    async def _generate(self, request):
        body = await request.json()
        self.execution_bodies.append(body)
        self.local_auth.append(request.headers.get("Authorization"))
        self.intake_started.set()
        if self.delay_intake:
            await self.intake_continue.wait()
        reservation_id = body["prefill_pull_reservation_id"]
        if self.reservations[reservation_id] != "bound":
            self.intake_finished.set()
            return web.Response(status=409, text="revoked")
        self.reservations[reservation_id] = "consumed"
        self.executed.append(body["rid"])
        self.intake_finished.set()
        return web.Response(status=503, body=self.local_response)

    async def test_lost_offer_and_commit_responses_execute_once_and_preserve_outcome(
        self,
    ):
        """Retries must resolve the original assignment before executing it."""
        self.lose_offer = True
        self.lose_commit = True
        await self._start()
        await asyncio.wait_for(self.outcome_finished.wait(), 3)
        self.assertEqual(self.poll_ids[:2], [self.poll_ids[0]] * 2)
        self.assertEqual(self.executed, ["router-engine-rid"])
        self.assertEqual(self.outcomes[0]["status_code"], 503)
        self.assertEqual(self.outcomes[0]["body"], "raw prefill error\nwith details")
        self.assertEqual(self.local_auth, ["Bearer local-key"])
        self.assertEqual(set(self.router_auth), {"Bearer router-key"})
        body = self.execution_bodies[0]
        self.assertEqual(body["prefill_pull_assignment_id"], "assignment-a")
        self.assertEqual(body["bootstrap_room"], 123)
        self.assertEqual(body["routed_dp_rank"], 0)
        self.assertFalse(body["stream"])

    async def test_cancel_revokes_delayed_intake_while_acquisition_is_blocked(self):
        """Status polling must continue while another acquire is long polling."""
        self.delay_intake = True
        await self._start(max_inflight=2)
        await asyncio.wait_for(self.intake_started.wait(), 3)
        await asyncio.wait_for(self.longpoll_started.wait(), 3)
        self.state = "cancelled"
        await asyncio.wait_for(self.aborted.wait(), 3)
        self.assertIn("revoked", self.abort_states[0])
        self.intake_continue.set()
        await asyncio.wait_for(self.intake_finished.wait(), 3)
        self.assertEqual(self.executed, [])
        await asyncio.wait_for(self.outcome_finished.wait(), 3)
        self.assertEqual(self.outcomes[0]["status_code"], 499)

    async def test_unconfirmed_commit_never_executes(self):
        """An accept transport failure cannot grant local execution authority."""
        self.lose_commit = True
        self.commit_state = "expired"
        await self._start()
        await asyncio.wait_for(self.reservation_released.wait(), 3)
        self.assertEqual(self.execution_bodies, [])
        self.assertEqual(self.executed, [])
        self.assertIn("revoked", self.reservations.values())

    async def test_unacknowledged_outcomes_bound_inflight_tasks(self):
        """A completed P request must not create an unbounded outcome backlog."""
        self.fail_outcome = True
        await self._start(max_inflight=1)
        await asyncio.wait_for(self.outcome_attempted.wait(), 3)
        self.assertEqual(self.executed, ["router-engine-rid"])
        self.assertEqual(len(self.poll_ids), 1)
        self.assertEqual(len(self.agent._assignments), 1)
        self.fail_outcome = False
        await asyncio.wait_for(self.outcome_finished.wait(), 3)
        self.assertEqual(self.executed, ["router-engine-rid"])

    async def test_unresolvable_commit_reports_failure_without_execution(self):
        """Losing both ACK and status cannot authorize speculative execution."""
        self.lose_commit = True
        self.fail_status = True
        with self.assertLogs("sglang.srt.managers.prefill_pull_agent", level="WARNING"):
            await self._start()
            await asyncio.wait_for(self.outcome_finished.wait(), 3)
        self.assertEqual(self.execution_bodies, [])
        self.assertEqual(self.outcomes[0]["status_code"], 502)
        self.assertEqual(self.state, "completed")
        self.assertIn("revoked", self.reservations.values())

    async def test_shutdown_revokes_delayed_intake_and_notifies_router(self):
        """Shutdown closes both execution authority and the waiting assignment."""
        self.delay_intake = True
        await self._start()
        await asyncio.wait_for(self.intake_started.wait(), 3)
        await self.agent.close()
        self.assertTrue(self.assignment_rejected.is_set())
        self.assertEqual(self.state, "cancelled")
        self.assertTrue(self.aborted.is_set())
        self.assertIn("revoked", self.abort_states[0])
        self.intake_continue.set()
        await asyncio.wait_for(self.intake_finished.wait(), 3)
        self.assertEqual(self.executed, [])
        self.assertEqual(self.agent._assignments, {})
        self.assertEqual(self.agent._assignment_reservations, {})

    async def test_invalid_utf8_cannot_expand_beyond_outcome_limit(self):
        """Decoding replacement bytes must not create an unreportable outcome."""
        self.local_response = b"\xff" * 32
        with self.assertLogs("sglang.srt.managers.prefill_pull_agent", level="WARNING"):
            await self._start(max_response_bytes=64)
            await asyncio.wait_for(self.outcome_finished.wait(), 3)
        self.assertEqual(self.executed, ["router-engine-rid"])
        self.assertEqual(self.outcomes[0]["status_code"], 502)
        self.assertLessEqual(len(self.outcomes[0]["body"].encode("utf-8")), 64)


if __name__ == "__main__":
    unittest.main()
