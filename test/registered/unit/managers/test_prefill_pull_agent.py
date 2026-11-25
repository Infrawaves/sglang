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
        self.release_calls = 0
        self.stall_scheduler = False
        self.peek_calls = 0
        self.peeks_without_room = 0
        self.peek_max_tokens = 64
        self.offer_needs_tokens = 0
        self.release_polls = asyncio.Event()
        self.empty_polls = 0
        self.empty_reply = {"status": "empty"}
        self.empty_delay = 0.0
        self.peek_gate = None
        self.acquire_status = None
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
        self.offer_started = asyncio.Event()
        self.offer_continue = asyncio.Event()
        self.offer_continue.set()
        self.accept_started = asyncio.Event()
        self.accept_continue = asyncio.Event()
        self.accept_continue.set()
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
        self.local_status = 503
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
        self.release_polls.set()
        self.offer_continue.set()
        self.accept_continue.set()
        await self.runner.cleanup()

    async def _start(
        self,
        max_inflight=1,
        max_response_bytes=4 * 1024 * 1024,
        other_routers=(),
        ready=lambda: True,
        retry_interval=0.01,
        **config,
    ):
        self.agent = PrefillPullAgent(
            PrefillPullAgentConfig(
                router_urls=(self.url, *other_routers),
                worker_url="http://registered-prefill:30000",
                local_url=self.url,
                max_inflight=max_inflight,
                api_key="local-key",
                router_api_key="router-key",
                retry_interval=retry_interval,
                status_interval=0.01,
                control_timeout=1,
                max_response_bytes=max_response_bytes,
                **config,
            ),
            budget=self._budget,
            abort=self._abort,
            ready=ready,
        )
        await self.agent.start()

    async def _budget(self, *, action, reservation_id, **kwargs):
        if action == "peek" and self.stall_scheduler:
            await asyncio.sleep(30)
        if action == "peek":
            self.peek_calls += 1
            if self.peeks_without_room:
                self.peeks_without_room -= 1
                return {"success": False, "error": "no_capacity"}
            if self.peek_gate is not None:
                await self.peek_gate.wait()
            return {"success": True, "data": {"max_tokens": self.peek_max_tokens}}
        if action == "reserve":
            self.reservations[reservation_id] = "reserved"
            return {"success": True, "data": {"max_tokens": 64}}
        if action == "bind":
            if self.reservations.get(reservation_id) != "reserved":
                return {"success": False}
            self.reservations[reservation_id] = "bound"
        elif action == "release":
            self.release_calls += 1
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
            if body["max_tokens"] < self.offer_needs_tokens:
                # Like the router: nothing fits, so the long poll waits.
                await self.release_polls.wait()
                return web.json_response({"status": "empty"})
            if self.empty_polls:
                self.empty_polls -= 1
                await asyncio.sleep(self.empty_delay)
                return web.json_response(self.empty_reply)
            if self.acquire_status is not None:
                return web.Response(status=self.acquire_status, text="refused")
            if self.assignment_offered and body["poll_id"] != self.poll_ids[0]:
                self.longpoll_started.set()
                await self.longpoll_continue.wait()
                return web.json_response({"status": "empty"})
            self.assignment_offered = True
            self.offer_started.set()
            await self.offer_continue.wait()
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
            self.accept_started.set()
            await self.accept_continue.wait()
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
        return web.Response(status=self.local_status, body=self.local_response)

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

    async def test_drain_rejects_an_offer_its_open_poll_brings_back(self):
        """Drain cancelled open polls; an offer the router had already made
        went unanswered until --pull-offer-timeout-ms, delaying its client."""
        self.offer_continue.clear()
        await self._start()
        await asyncio.wait_for(self.offer_started.wait(), 3)
        draining = asyncio.create_task(self.agent.drain())
        await asyncio.sleep(0.05)
        self.offer_continue.set()
        await asyncio.wait_for(draining, 3)
        self.assertTrue(self.assignment_rejected.is_set())
        self.assertEqual(self.state, "expired")
        self.assertEqual(self.execution_bodies, [])

    async def test_drain_stops_waiting_for_a_poll_the_router_never_answers(self):
        """Waiting for open polls is bounded by poll_wait_ms + control_timeout."""
        self.offer_continue.clear()
        await self._start(poll_wait_ms=100)
        await asyncio.wait_for(self.offer_started.wait(), 3)
        await asyncio.wait_for(self.agent.drain(), 3)
        self.assertEqual(self.execution_bodies, [])

    async def test_offer_arriving_after_readiness_loss_is_rejected(self):
        """An open acquire must not dispatch new work after P starts draining."""
        ready = True
        self.offer_continue.clear()
        await self._start(ready=lambda: ready)
        await asyncio.wait_for(self.offer_started.wait(), 3)
        ready = False
        self.offer_continue.set()
        await asyncio.wait_for(self.assignment_rejected.wait(), 3)
        self.assertEqual(self.state, "expired")
        self.assertEqual(self.reservations, {})
        self.assertFalse(self.accept_started.is_set())
        self.assertEqual(self.execution_bodies, [])

    async def test_readiness_lost_during_bind_revokes_before_accept(self):
        """Readiness can change after an offer while its reservation is binding."""
        ready = True
        bound = asyncio.Event()
        resume_bind = asyncio.Event()
        original_budget = self._budget

        async def delayed_bind(**kwargs):
            result = await original_budget(**kwargs)
            if kwargs["action"] == "bind":
                bound.set()
                await resume_bind.wait()
            return result

        self._budget = delayed_bind
        await self._start(ready=lambda: ready)
        try:
            await asyncio.wait_for(bound.wait(), 3)
            ready = False
            resume_bind.set()
            await asyncio.wait_for(self.assignment_rejected.wait(), 3)
            await asyncio.wait_for(self.reservation_released.wait(), 3)
            self.assertEqual(self.state, "expired")
            self.assertEqual(list(self.reservations.values()), ["revoked"])
            self.assertFalse(self.accept_started.is_set())
            self.assertEqual(self.execution_bodies, [])
        finally:
            resume_bind.set()

    async def test_drain_waits_for_accept_execution_and_outcome(self):
        """Shutdown must wait for a committed claim before local intake exists."""
        self.accept_continue.clear()
        self.delay_intake = True
        self.fail_outcome = True
        self.local_status = 200
        self.local_response = b"{}"
        await self._start(max_inflight=2)
        await asyncio.wait_for(self.accept_started.wait(), 3)
        await asyncio.wait_for(self.longpoll_started.wait(), 3)
        claim_loops = [task for tasks in self.agent._routers.values() for task in tasks]
        draining = asyncio.create_task(self.agent.drain())
        # Drain lets open polls finish; this one answers as its wait ends.
        self.longpoll_continue.set()
        try:
            await asyncio.wait_for(
                asyncio.gather(*claim_loops, return_exceptions=True), 3
            )
            self.assertFalse(draining.done())
            self.assertEqual(self.execution_bodies, [])
            self.assertEqual(self.state, "committed")

            self.accept_continue.set()
            await asyncio.wait_for(self.intake_started.wait(), 3)
            self.assertFalse(draining.done())
            self.assertFalse(self.aborted.is_set())

            self.intake_continue.set()
            await asyncio.wait_for(self.outcome_attempted.wait(), 3)
            self.assertFalse(draining.done())
            self.assertEqual(self.executed, ["router-engine-rid"])

            self.fail_outcome = False
            await asyncio.wait_for(draining, 3)
            self.assertEqual(self.state, "completed")
            self.assertFalse(self.aborted.is_set())
            self.assertFalse(self.assignment_rejected.is_set())
            self.assertEqual(self.agent._assignments, {})
        finally:
            self.accept_continue.set()
            self.intake_continue.set()
            self.fail_outcome = False
            if not draining.done():
                await self.agent.close()
            await asyncio.gather(draining, return_exceptions=True)

    async def test_claims_from_every_router_and_reports_to_the_claiming_one(self):
        """Routers keep separate queues; an outcome sent to another router would
        leave the claiming router's client waiting until its deadline."""
        other_polls = []
        other_outcomes = []

        async def other(request):
            body = await request.json()
            action = request.match_info["action"]
            if action == "acquire":
                other_polls.append(body["poll_id"])
                await asyncio.sleep(0.01)
                return web.json_response({"status": "empty"})
            if action == "outcome":
                other_outcomes.append(body)
            return web.json_response({"status": "expired"})

        app = web.Application()
        app.router.add_post("/internal/pull/{action}", other)
        runner = web.AppRunner(app)
        await runner.setup()
        self.addAsyncCleanup(runner.cleanup)
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        other_url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        await self._start(max_inflight=2, other_routers=(other_url,))
        await asyncio.wait_for(self.outcome_finished.wait(), 3)
        self.assertEqual(self.executed, ["router-engine-rid"])
        self.assertTrue(other_polls)
        self.assertEqual(other_outcomes, [])

    async def test_empty_router_long_poll_holds_no_reservation(self):
        """With one request slot, an empty router's long poll that held a
        reservation starved every other router's claims for its whole wait."""
        polling = asyncio.Event()

        async def empty(request):
            if request.match_info["action"] == "acquire":
                polling.set()
                await asyncio.sleep(30)
            return web.json_response({"status": "empty"})

        app = web.Application()
        app.router.add_post("/internal/pull/{action}", empty)
        # Cleanup must not wait for the parked long poll to finish.
        runner = web.AppRunner(app, shutdown_timeout=0.1)
        await runner.setup()
        self.addAsyncCleanup(runner.cleanup)
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        empty_url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        self.agent = PrefillPullAgent(
            PrefillPullAgentConfig(
                router_urls=(empty_url,),
                worker_url="http://registered-prefill:30000",
                local_url=self.url,
                max_inflight=1,
                retry_interval=0.01,
            ),
            budget=self._budget,
            abort=self._abort,
            ready=lambda: True,
        )
        await self.agent.start()
        await asyncio.wait_for(polling.wait(), 3)
        self.assertNotIn("reserved", self.reservations.values())

    async def test_outcome_reaches_a_draining_router_for_as_long_as_it_answers(self):
        """A rolling update removes a router from DNS while it still serves the
        client, possibly for longer than any fixed grace; giving up while it
        still reported the assignment committed left that client to time out."""
        self.fail_outcome = True
        await self._start(router_unreachable_grace=0.05)
        await asyncio.wait_for(self.outcome_attempted.wait(), 3)
        await self.agent._drop_router(self.url)
        # Well past the grace, but the router answers status throughout.
        await asyncio.sleep(0.3)
        self.fail_outcome = False
        await asyncio.wait_for(self.outcome_finished.wait(), 3)
        self.assertEqual(self.state, "completed")

    async def test_unreachable_listed_router_frees_its_inflight_slot(self):
        """A fixed IP or https router is never removed from the list; while its
        outcome retried forever, it held the only in-flight slot and no other
        claim could start."""
        self.fail_outcome = True
        await self._start(max_inflight=1, router_unreachable_grace=0.05)
        await asyncio.wait_for(self.outcome_attempted.wait(), 3)
        self.fail_status = True
        async with asyncio.timeout(3):
            while self.agent._assignments:
                await asyncio.sleep(0.01)
        self.assertIn(self.url, self.agent._routers)
        self.assertEqual(self.state, "committed")
        # The freed slot lets claims resume.
        await asyncio.wait_for(self.longpoll_started.wait(), 3)

    async def test_router_leaves_only_after_repeated_missing_resolutions(self):
        """A transient DNS gap must not stop claims or outcome reporting for a
        router that is still serving clients."""
        self.agent = PrefillPullAgent(
            PrefillPullAgentConfig(
                router_urls=("http://routers.example:8000",),
                worker_url="http://registered-prefill:30000",
                local_url=self.url,
                max_inflight=1,
                retry_interval=0.01,
                router_missing_refreshes=2,
            ),
            budget=self._budget,
            abort=self._abort,
            # Never ready: the claim loops idle without touching the network.
            ready=lambda: False,
        )
        both = {"http://10.0.0.1:8000", "http://10.0.0.2:8000"}
        answers = [both, {"http://10.0.0.1:8000"}, {"http://10.0.0.1:8000"}]

        async def resolve(url):
            return answers.pop(0)

        self.agent._resolve = resolve
        await self.agent._refresh_routers()
        self.assertEqual(self.agent._listed, both)
        await self.agent._refresh_routers()
        self.assertEqual(self.agent._listed, both)
        await self.agent._refresh_routers()
        self.assertEqual(self.agent._listed, {"http://10.0.0.1:8000"})
        # Unlisted is not abandoned: its loops drain its queue while it answers.
        self.assertEqual(set(self.agent._routers), both)

    async def test_a_router_that_left_dns_still_has_its_queue_claimed(self):
        """A router leaving a rolling update drains: requests it queued but
        nobody claimed yet timed out with 504 once the agent stopped claiming
        from it on DNS removal."""
        ready = False
        await self._start(ready=lambda: ready)
        await self.agent._drop_router(self.url)
        self.assertNotIn(self.url, self.agent._listed)
        ready = True
        await asyncio.wait_for(self.outcome_finished.wait(), 3)
        self.assertEqual(self.executed, ["router-engine-rid"])

    async def test_a_stalled_local_scheduler_does_not_retire_a_departed_router(self):
        """Local budget timeouts were counted as the router being unreachable:
        a router that left DNS was dropped while it answered, before a single
        acquire, and the stalled scheduler's recovery never claimed again."""
        self.stall_scheduler = True
        await self._start(router_unreachable_grace=0.05)
        await self.agent._drop_router(self.url)
        # Past the grace, with every peek timing out (control_timeout=1).
        await asyncio.sleep(1.5)
        self.assertIn(self.url, self.agent._routers)
        self.assertEqual(self.poll_ids, [])
        self.stall_scheduler = False
        await asyncio.wait_for(self.outcome_finished.wait(), 5)
        self.assertEqual(self.executed, ["router-engine-rid"])

    async def test_an_unreachable_unlisted_router_stops_being_claimed_from(self):
        """Claiming from a departed router lasts only while it answers."""
        import socket

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            dead = f"http://127.0.0.1:{sock.getsockname()[1]}"
        self.agent = PrefillPullAgent(
            PrefillPullAgentConfig(
                router_urls=(dead,),
                worker_url="http://registered-prefill:30000",
                local_url=self.url,
                max_inflight=1,
                retry_interval=0.01,
                router_unreachable_grace=0.05,
            ),
            budget=self._budget,
            abort=self._abort,
            ready=lambda: True,
        )
        await self.agent.start()
        await self.agent._drop_router(dead)
        async with asyncio.timeout(3):
            while dead in self.agent._routers:
                await asyncio.sleep(0.01)

    async def test_a_finished_assignment_wakes_the_claim_loop_at_once(self):
        """A full worker slept retry_interval before claiming again, so freed
        room sat idle for half of it on average."""
        await self._start(retry_interval=5)
        await asyncio.wait_for(self.outcome_finished.wait(), 3)
        # The next claim starts long before retry_interval has passed.
        await asyncio.wait_for(self.longpoll_started.wait(), 2)

    async def test_budget_freed_mid_request_is_claimed_without_retry_interval(self):
        """Compute budget frees when a request's last chunk is allocated, which
        no local event signals; a worker without room slept retry_interval."""
        self.peeks_without_room = 3
        await self._start(retry_interval=5)
        await asyncio.wait_for(self.outcome_finished.wait(), 2)

    async def test_a_full_worker_peeks_once_per_interval_however_many_loops_ask(
        self,
    ):
        """Loops waking at staggered times each sent their own peek, a TP
        collective, so a full worker peeked claim loops x 20 times a second."""
        self.peeks_without_room = 100
        await self._start(ready=lambda: False, no_capacity_retry_interval=10)
        first = await self.agent._peek()
        await asyncio.sleep(0.01)
        second = await self.agent._peek()
        self.assertEqual((first, second), (first, first))
        self.assertEqual(self.peek_calls, 1)

    async def test_an_open_long_poll_asks_again_once_room_grows(self):
        """A long poll kept the max_tokens it began with: room freed while it
        was open left a fitting queued request waiting out poll_wait_ms."""
        self.peek_max_tokens, self.offer_needs_tokens = 2, 3
        await self._start(poll_refresh_interval=0.05, no_capacity_retry_interval=0)
        while not self.poll_ids:
            await asyncio.sleep(0.01)
        self.peek_max_tokens = 64
        await asyncio.wait_for(self.outcome_finished.wait(), 2)
        # Asked again under the same poll ID, so a lost offer cannot duplicate.
        self.assertEqual(set(self.poll_ids), {self.poll_ids[0]})

    async def test_a_held_worker_peeks_again_without_retry_interval(self):
        """A worker held for a starved request kept its stale grant through a
        whole long poll, then slept retry_interval before peeking again."""
        self.empty_polls = 3
        self.empty_reply = {"status": "empty", "held": True}
        await self._start(retry_interval=5)
        await asyncio.wait_for(self.outcome_finished.wait(), 2)

    async def test_a_poll_that_waited_out_wait_ms_does_not_also_sleep(self):
        """Every empty poll slept retry_interval, even one that had already
        waited wait_ms; requests arriving meanwhile went unclaimed."""
        self.empty_polls, self.empty_delay = 1, 0.2
        await self._start(retry_interval=5, poll_wait_ms=200)
        await asyncio.wait_for(self.outcome_finished.wait(), 2)

    async def test_a_long_poll_outlasting_control_timeout_is_not_a_timeout(self):
        """A poll_wait_ms above control_timeout timed out a router that was
        still legitimately holding the poll, counting it as unreachable."""
        self.empty_polls, self.empty_delay = 1, 1.3
        await self._start(poll_wait_ms=2000)
        await asyncio.wait_for(self.outcome_finished.wait(), 4)
        # The held poll was answered, not retried under the same poll ID.
        self.assertNotEqual(self.poll_ids[0], self.poll_ids[1])

    async def test_zero_wait_polls_back_off_on_empty(self):
        """With poll_wait_ms=0 an empty answer never counted as early, so the
        agent re-acquired at network RTT instead of every retry_interval."""
        self.empty_polls = 1000
        await self._start(retry_interval=0.2, poll_wait_ms=0)
        await asyncio.sleep(0.3)
        self.assertLessEqual(len(self.poll_ids), 3)

    async def test_claim_loops_share_one_scheduler_peek(self):
        """Every claim loop ran its own peek, multiplying scheduler round
        trips (each a TP all-reduce) by claim_concurrency."""
        self.peek_gate = asyncio.Event()
        await self._start(claim_concurrency=4)
        for _ in range(20):
            await asyncio.sleep(0)
        self.peek_gate.set()
        self.assertEqual(self.peek_calls, 1)

    async def test_a_router_refusing_claims_is_not_unreachable(self):
        """A 4xx counted as unreachable: an answering router that left DNS
        was retired, and every loop logged a traceback each retry_interval."""
        self.acquire_status = 401
        with self.assertLogs(
            "sglang.srt.managers.prefill_pull_agent", level="DEBUG"
        ) as logs:
            await self._start(
                router_unreachable_grace=0.05, rejected_retry_interval=0.2
            )
            await self.agent._drop_router(self.url)
            await asyncio.sleep(0.5)
        self.assertIn(self.url, self.agent._routers)
        # Backed off, not retried every retry_interval (0.01).
        self.assertLessEqual(len(self.poll_ids), 4)
        warnings = [r for r in logs.records if r.levelname == "WARNING"]
        self.assertEqual(len(warnings), 1)
        self.assertFalse(warnings[0].exc_info)

    async def test_large_successful_prefill_response_still_reports_success(self):
        """Logprobs or hidden states can make a 200 body exceed the outcome
        limit; relaying it turned a successful prefill into a 502."""
        self.local_status = 200
        self.local_response = b'{"meta_info": "' + b"x" * 1024 + b'"}'
        await self._start(max_response_bytes=64)
        await asyncio.wait_for(self.outcome_finished.wait(), 3)
        self.assertEqual(self.outcomes[0]["status_code"], 200)
        self.assertEqual(self.outcomes[0]["body"], "{}")

    async def test_offers_from_several_routers_respect_max_inflight(self):
        """Every router's loop passes the pre-poll check while its poll is open;
        admitting each returned offer exceeded max_inflight."""
        release = asyncio.Event()
        rejected = []

        async def other(request):
            body = await request.json()
            action = request.match_info["action"]
            if action == "acquire":
                if rejected:
                    await asyncio.sleep(30)
                await release.wait()
                return web.json_response(
                    {
                        "status": "offer",
                        "assignment_id": "assignment-b",
                        "request_id": "other-rid",
                        "path": "/generate",
                        "input_tokens": 3,
                        "body": {"input_ids": [1, 2, 3]},
                        "expires_in_ms": 30000,
                    }
                )
            if action == "reject":
                rejected.append(body["assignment_id"])
            return web.json_response({"status": "expired"})

        app = web.Application()
        app.router.add_post("/internal/pull/{action}", other)
        runner = web.AppRunner(app, shutdown_timeout=0.1)
        await runner.setup()
        self.addAsyncCleanup(runner.cleanup)
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        other_url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        self.delay_intake = True
        await self._start(max_inflight=1, other_routers=(other_url,))
        await asyncio.wait_for(self.intake_started.wait(), 3)
        release.set()
        async with asyncio.timeout(3):
            while not rejected:
                await asyncio.sleep(0.01)
        self.assertEqual(rejected, ["assignment-b"])
        self.assertEqual(list(self.agent._assignments), ["assignment-a"])

    async def test_one_assignment_releases_its_grant_once(self):
        """Each release is a scheduler IPC plus a TP collective; one assignment
        sent up to three."""
        self.local_status = 200
        self.local_response = b"{}"
        await self._start()
        await asyncio.wait_for(self.outcome_finished.wait(), 3)
        async with asyncio.timeout(3):
            while self.agent._assignments:
                await asyncio.sleep(0.005)
        self.assertEqual(self.release_calls, 1)

    async def test_finished_execution_wins_a_tie_with_the_monitor(self):
        """When /generate returns in the same round the monitor gives up, the
        monitor branch won: a successful prefill was aborted and reported 499."""
        self.local_status = 200
        self.local_response = b"{}"
        tie = asyncio.Event()

        async def execute(body, headers):
            await tie.wait()
            return 200, "{}"

        async def monitor(router, identity):
            await tie.wait()

        await self._start()
        self.agent._execute = execute
        self.agent._monitor = monitor
        await asyncio.wait_for(self._wait_committed(), 3)
        tie.set()
        await asyncio.wait_for(self.outcome_finished.wait(), 3)
        self.assertEqual(self.outcomes[0]["status_code"], 200)
        self.assertFalse(self.aborted.is_set())

    async def _wait_committed(self):
        while self.state != "committed":
            await asyncio.sleep(0.005)

    async def test_dual_stack_router_name_is_one_router(self):
        """A name with A and AAAA records (localhost, dual-stack pods) counted
        each router twice and doubled its claim loops."""
        import socket

        agent = PrefillPullAgent(
            PrefillPullAgentConfig(
                router_urls=("http://routers.example:8000",),
                worker_url="http://registered-prefill:30000",
                local_url=self.url,
                max_inflight=1,
            ),
            budget=self._budget,
            abort=self._abort,
            ready=lambda: False,
        )

        async def getaddrinfo(host, port, **kwargs):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", port)),
                (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("fd00::1", port, 0, 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.2", port)),
            ]

        loop = asyncio.get_running_loop()
        original = loop.getaddrinfo
        loop.getaddrinfo = getaddrinfo
        try:
            routers = await agent._resolve("http://routers.example:8000")
        finally:
            loop.getaddrinfo = original
        self.assertEqual(routers, {"http://10.0.0.1:8000", "http://10.0.0.2:8000"})

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
