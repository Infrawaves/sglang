"""Pull admission invariants, independent of GPU execution and network policy.

A dedicated fixture is used because this ledger needs deterministic time and
allocator snapshots, unlike the transfer/queue fixtures in this directory.
"""

import asyncio
import tempfile
import time
import unittest
from array import array
from types import SimpleNamespace

import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.arg_groups.pd_disaggregation_hook import validate_prefill_pull
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.mode import FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    PrefillPullBudgetReq,
    PrefillPullBudgetReqOutput,
    TokenizedGenerateReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.scheduler_components.prefill_work_admission import (
    PrefillCapacity,
    PrefillPullBudgetClient,
    PrefillWorkAdmission,
    SchedulerPrefillWorkAdmission,
    prepare_prefill_pull_request,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _request(*, allocated=0, row=None, metadata=-1, finished=False, sender=None):
    return SimpleNamespace(
        kv=SimpleNamespace(kv_allocated_len=allocated, req_pool_idx=row),
        metadata_buffer_index=metadata,
        disagg_kv_sender=sender,
        finished=lambda: finished,
    )


def _stats(tokens):
    return SimpleNamespace(
        full_available_size=tokens,
        full_evictable_size=0,
        is_hybrid_swa=False,
        is_hybrid_ssm=False,
        is_hisparse=False,
    )


def _tp_admission_worker(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        stats = _stats(80 if rank == 0 else 40)
        adapter = SchedulerPrefillWorkAdmission(
            max_tokens=64,
            max_requests=4,
            safety_margin_tokens=0,
            pool_stats=SimpleNamespace(
                get_pool_stats=lambda: stats,
                storage=SimpleNamespace(
                    usage=lambda: [SimpleNamespace(metrics="full")]
                ),
            ),
            req_pool=SimpleNamespace(available_size=lambda: 4),
            metadata_allocator=SimpleNamespace(available_size=lambda: 4),
            page_size=4,
            cpu_group=dist.group.WORLD,
            group_rank=0,
            group_size=2,
        )
        first = adapter.handle_control(
            PrefillPullBudgetReq(action="reserve", reservation_id="one"), paused=False
        )
        adapter.handle_control(
            PrefillPullBudgetReq(
                action="bind",
                reservation_id="one",
                assignment_id="a",
                request_id="rid",
                tokens=8,
            ),
            paused=False,
        )
        adapter.ledger.consume(
            reservation_id="one",
            assignment_id="a",
            request_id="rid",
            tokens=8,
            now=adapter._now(),
        )
        allocated = 8 if rank == 0 else 4
        adapter.ledger.attach_request(
            "one", _request(allocated=allocated, row=0, metadata=0)
        )
        stats.full_available_size -= allocated
        second = adapter.handle_control(
            PrefillPullBudgetReq(action="reserve", reservation_id="two"), paused=False
        )
        output.put((rank, first.data["max_tokens"], second.data["max_tokens"]))
    finally:
        dist.destroy_process_group()


class TestPrefillPullAdmission(CustomTestCase):
    def setUp(self):
        self.ledger = PrefillWorkAdmission(
            max_tokens=64, max_requests=2, page_size=4, reservation_ttl=10
        )
        self.capacity = PrefillCapacity(tokens=128, requests=2, metadata=2)

    def test_intake_rejection_and_failure_preserve_execution_ownership(self):
        """Only admitted intake runs; a started sender outlives intake failure."""
        adapter = SchedulerPrefillWorkAdmission(
            max_tokens=64,
            max_requests=2,
            safety_margin_tokens=0,
            pool_stats=SimpleNamespace(
                get_pool_stats=lambda: _stats(128),
                storage=SimpleNamespace(
                    usage=lambda: [SimpleNamespace(metrics="full")]
                ),
            ),
            req_pool=SimpleNamespace(available_size=lambda: 2),
            metadata_allocator=SimpleNamespace(available_size=lambda: 2),
            page_size=4,
            cpu_group=None,
            group_rank=0,
            group_size=1,
        )
        req = TokenizedGenerateReqInput(
            rid="rid",
            input_text=None,
            input_ids=array("q", [1, 2]),
            input_embeds=None,
            mm_inputs=None,
            token_type_ids=None,
            sampling_params=SamplingParams(max_new_tokens=1),
            return_logprob=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            prefill_pull_reservation_id="r",
            prefill_pull_assignment_id="a",
        )
        outputs = []
        processed = []
        adapter.handle_generate_request(
            req,
            process_request=lambda: processed.append(req),
            send_control_output=lambda response, original: outputs.append(response),
        )
        self.assertEqual(processed, [])
        self.assertEqual(outputs[0].rid, "rid")
        self.assertEqual(outputs[0].finished_reason["status_code"], 403)

        for sender_started in (False, True):
            with self.subTest(sender_started=sender_started):
                req.prefill_pull_reservation_id = f"r-{sender_started}"
                reservation_id = req.prefill_pull_reservation_id
                now = time.monotonic()
                adapter.ledger.control(
                    action="reserve",
                    reservation_id=reservation_id,
                    now=now,
                    capacity=self.capacity,
                )
                adapter.ledger.control(
                    action="bind",
                    reservation_id=reservation_id,
                    assignment_id="a",
                    request_id="rid",
                    tokens=2,
                    now=now,
                )
                executing = _request(sender=object())

                def fail_intake():
                    if sender_started:
                        adapter.ledger.attach_request(reservation_id, executing)
                    raise RuntimeError("intake failed")

                with self.assertRaisesRegex(RuntimeError, "intake failed"):
                    adapter.handle_generate_request(
                        req,
                        process_request=fail_intake,
                        send_control_output=lambda response, original: self.fail(
                            "admitted intake unexpectedly aborted"
                        ),
                    )
                state = adapter.ledger.control(
                    action="query", reservation_id=reservation_id, now=now
                )["data"]["state"]
                self.assertEqual(state, "consumed" if sender_started else "retired")
                if sender_started:
                    adapter.ledger.retire(executing)

    def reserve(self, reservation_id="r", now=0, capacity=None):
        return self.ledger.control(
            action="reserve",
            reservation_id=reservation_id,
            now=now,
            capacity=capacity or self.capacity,
        )

    def bind(self, tokens=16, now=1):
        return self.ledger.control(
            action="bind",
            reservation_id="r",
            now=now,
            assignment_id="a",
            request_id="rid",
            tokens=tokens,
        )

    def consume(self, tokens=16, now=2, **changes):
        args = dict(
            reservation_id="r",
            assignment_id="a",
            request_id="rid",
            tokens=tokens,
            now=now,
        )
        args.update(changes)
        return self.ledger.consume(**args)

    def test_reservations_compete_before_any_request_exists(self):
        """A stale idle observation must not grant the same work budget twice."""
        self.assertEqual(self.reserve()["data"]["max_tokens"], 64)
        self.assertFalse(self.reserve("second")["success"])
        self.assertTrue(self.bind(tokens=16)["success"])
        self.assertEqual(self.reserve("second")["data"]["max_tokens"], 48)
        self.assertFalse(self.reserve("third")["success"])
        self.assertEqual(self.reserve()["error"], "reservation already bound")

    def test_revocation_expiry_and_execution_ownership(self):
        """Timeout may expire an offer, never a request retaining source KV."""
        self.reserve()
        self.bind()
        self.assertIsNone(self.consume())
        req = _request(finished=True, sender=object())
        self.ledger.attach_request("r", req)
        self.ledger.finish_intake("r", failed=False)
        release = self.ledger.control(action="release", reservation_id="r", now=100)
        self.assertFalse(release["success"])
        self.assertEqual(self.reserve("second", now=100)["data"]["max_tokens"], 48)
        self.ledger.retire(_request())
        self.assertIn("r", self.ledger.reservations)
        self.ledger.retire(req)
        state = self.ledger.control(action="query", reservation_id="r", now=100)
        self.assertEqual(state["data"]["state"], "retired")
        self.assertIsNotNone(self.consume(now=100))
        self.assertFalse(self.reserve(now=100)["success"])

    def test_expired_bound_offer_cannot_execute_or_steal_other_grant(self):
        self.reserve()
        self.bind()
        for changes in (
            {"assignment_id": "other"},
            {"request_id": "other"},
            {"tokens": 17},
        ):
            with self.subTest(changes=changes):
                self.assertIsNotNone(self.consume(**changes))
        self.assertIsNone(self.consume(tokens=8))
        self.assertEqual(self.ledger.reservations["r"].tokens, 8)
        self.assertIsNotNone(self.consume(tokens=8))
        self.reserve("expires", now=2)
        self.assertTrue(
            self.ledger.control(
                action="bind",
                reservation_id="expires",
                assignment_id="x",
                request_id="x",
                tokens=1,
                now=3,
            )["success"]
        )
        result = self.ledger.control(action="query", reservation_id="expires", now=12)
        self.assertEqual(result["data"]["state"], "expired")
        self.assertIsNotNone(
            self.ledger.consume(
                reservation_id="expires",
                assignment_id="x",
                request_id="x",
                tokens=1,
                now=12,
            )
        )

    def test_page_memory_and_metadata_independently_limit_work(self):
        for capacity in (
            PrefillCapacity(tokens=128, requests=0, metadata=2),
            PrefillCapacity(tokens=128, requests=2, metadata=0),
            PrefillCapacity(tokens=4, requests=2, metadata=2),
        ):
            with self.subTest(capacity=capacity):
                self.assertFalse(self.reserve(capacity=capacity)["success"])
        self.assertEqual(
            self.reserve(capacity=PrefillCapacity(24, 2, 2))["data"]["max_tokens"], 20
        )
        self.bind(tokens=8)
        self.consume(tokens=8)
        req = _request(allocated=8, row=0, metadata=0)
        self.ledger.attach_request("r", req)
        # 24 tokens total, 8 already allocated, 4 promised to finish the page.
        # Free memory must not subtract the already allocated 8 a second time.
        result = self.reserve("second", capacity=PrefillCapacity(16, 1, 1))
        self.assertEqual(result["data"]["max_tokens"], 8)

    def test_failed_intake_retires_without_releasing_a_started_sender(self):
        self.reserve()
        self.bind()
        self.consume()
        req = _request(finished=True)
        self.ledger.attach_request("r", req)
        self.ledger.finish_intake("r", failed=False)
        self.assertEqual(self.reserve("next")["data"]["max_tokens"], 64)

    def test_internal_probe_occupancy_blocks_new_promises_until_retired(self):
        reservation_id = self.ledger.admit_internal("health-probe", 64)
        req = _request(sender=object())
        self.ledger.attach_request(reservation_id, req)
        self.assertFalse(self.reserve()["success"])
        self.ledger.retire(req)
        self.assertEqual(self.reserve()["data"]["max_tokens"], 64)

    def test_tp_grant_uses_weakest_rank_after_outstanding_promises(self):
        """Consensus includes rank-local debt, not just raw allocator free space."""
        context = mp.get_context("spawn")
        output = context.Queue()
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = f"file://{directory}/gloo"
            processes = [
                context.Process(
                    target=_tp_admission_worker, args=(rank, rendezvous, output)
                )
                for rank in range(2)
            ]
            for process in processes:
                process.start()
            try:
                results = [output.get(timeout=40) for _ in processes]
                for process in processes:
                    process.join(timeout=10)
                    self.assertEqual(process.exitcode, 0)
                self.assertEqual(sorted(results), [(0, 36, 24), (1, 36, 24)])
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=10)
                output.close()
                output.join_thread()


class TestPrefillPullControlAndInputs(CustomTestCase):
    def test_cancelled_control_cannot_capture_next_reply(self):
        async def scenario():
            sent = []
            client = PrefillPullBudgetClient(sent.append)
            first = asyncio.create_task(
                client.call(action="reserve", reservation_id="first")
            )
            await asyncio.sleep(0)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            second = asyncio.create_task(
                client.call(action="reserve", reservation_id="second")
            )
            await asyncio.sleep(0)
            client.handle_recv(
                PrefillPullBudgetReqOutput(
                    rid=sent[0].rid, success=True, data={"max_tokens": 999}
                )
            )
            self.assertFalse(second.done())
            client.handle_recv(
                PrefillPullBudgetReqOutput(
                    rid=sent[1].rid, success=True, data={"max_tokens": 4}
                )
            )
            self.assertEqual((await second)["data"]["max_tokens"], 4)
            self.assertEqual(client.pending, {})

        asyncio.run(scenario())

    def test_ipc_roundtrip_preserves_authorization_and_reply_correlation(self):
        req = TokenizedGenerateReqInput(
            rid="rid",
            input_text=None,
            input_ids=array("q", [1, 2]),
            input_embeds=None,
            mm_inputs=None,
            token_type_ids=None,
            sampling_params=SamplingParams(max_new_tokens=1),
            return_logprob=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            prefill_pull_reservation_id="r",
            prefill_pull_assignment_id="a",
        )
        decoded = msgpack_decode(msgpack_encode(req))
        self.assertEqual(
            (decoded.prefill_pull_reservation_id, decoded.prefill_pull_assignment_id),
            ("r", "a"),
        )
        self.assertFalse(decoded.prefill_pull_internal)
        control = PrefillPullBudgetReq(
            rid="correlation",
            action="bind",
            reservation_id="r",
            assignment_id="a",
            request_id="rid",
            tokens=2,
        )
        decoded = msgpack_decode(msgpack_encode(control))
        self.assertEqual(
            (decoded.rid, decoded.request_id, decoded.tokens), ("correlation", "rid", 2)
        )

    def test_health_prefix_and_fake_host_are_not_public_bypasses(self):
        def http(path, client="127.0.0.1"):
            return SimpleNamespace(
                url=SimpleNamespace(path=path), client=SimpleNamespace(host=client)
            )

        obj = GenerateReqInput(
            rid=HEALTH_CHECK_RID_PREFIX + "probe",
            input_ids=[1],
            bootstrap_host=FAKE_BOOTSTRAP_HOST,
        )
        with self.assertRaises(ValueError):
            prepare_prefill_pull_request(obj, http("/generate"), starting=False)
        self.assertTrue(
            prepare_prefill_pull_request(obj, http("/health_generate"), starting=False)
        )
        obj.rid = "warmup"
        self.assertTrue(
            prepare_prefill_pull_request(obj, http("/generate"), starting=True)
        )
        with self.assertRaises(ValueError):
            prepare_prefill_pull_request(
                obj, http("/generate", "10.0.0.2"), starting=True
            )
        with self.assertRaises(ValueError):
            prepare_prefill_pull_request(obj, http("/generate"), starting=False)

    def test_configuration_rejects_unsupported_topologies_and_missing_budgets(self):
        base = dict(
            model_path="dummy",
            disaggregation_mode="prefill",
            prefill_pull_router="http://router:8000",
            prefill_pull_worker_url="http://prefill:30000",
            prefill_pull_router_api_key="test-secret",
            prefill_pull_max_tokens=4096,
            prefill_pull_max_requests=4,
        )
        validate_prefill_pull(resolving_view(ServerArgs(**base, tp_size=2)))
        for changes in (
            {"disaggregation_mode": "decode"},
            {"disaggregation_decode_allocation_policy": "prefill_complete"},
            {"optimistic_prefill_attempts": 1},
            {"dp_size": 2},
            {"attn_dp_size": 2},
            {"pp_size": 2},
            {"attn_cp_size": 2},
            {"tokenizer_worker_num": 2},
            {"grpc_port": 9000},
            {"enable_pd_role_switch": True},
            {"dllm_algorithm": "LowConfidence"},
            {"is_embedding": True},
            {"warmups": "voice_chat"},
            {"prefill_pull_max_tokens": None},
            {"prefill_pull_max_requests": 0},
            {"prefill_pull_safety_margin_tokens": -1},
            {"prefill_pull_worker_url": None},
            {"prefill_pull_router_api_key": None},
            {"prefill_pull_router_api_key": ""},
            {"prefill_pull_router_api_key": "   "},
            {"prefill_pull_router_api_key": "a b"},
            {"prefill_pull_router_api_key": "密钥"},
            {"prefill_pull_router_api_key": "key\n"},
        ):
            with (
                self.subTest(changes=changes),
                self.assertRaisesRegex(ValueError, "prefill-pull"),
            ):
                validate_prefill_pull(resolving_view(ServerArgs(**(base | changes))))
        # Default push remains valid without a pull budget.
        validate_prefill_pull(resolving_view(ServerArgs(model_path="dummy")))


if __name__ == "__main__":
    unittest.main()
