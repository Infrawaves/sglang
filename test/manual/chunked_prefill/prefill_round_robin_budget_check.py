"""CPU check of the implemented RR gate and PrefillAdder method bodies.

Run with Python 3, without importing SGLang/GPU dependencies. AST extraction keeps
the existing admission, continuation and charging code intact; cache operations,
requests and the physical page ledger are stand-ins. This is a design check, not
a scheduler, cache-sharing, TP or transport integration test.
"""

import ast
import os
import random
from contextlib import contextmanager, nullcontext
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace as NS

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/managers/schedule_policy.py"
)


def load_adder_methods():
    tree = ast.parse(SOURCE.read_text())
    names = {
        "ceil_paged_tokens",
        "budget_state",
        "_update_prefill_budget",
        "add_chunked_req",
        "add_one_req",
        "add_one_req_ignore_eos",
        "_round_robin_can_admit",
    }
    selected = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ]
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name)
            and t.id in {"CLIP_MAX_NEW_TOKENS", "IGNORE_EOS_RESERVE_TOKENS"}
            for t in node.targets
        ):
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "AddReqResult":
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "PrefillAdder":
            node.body = [
                n
                for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name in names
            ]
            assert {n.name for n in node.body} == names
            selected.append(node)
    namespace = {"os": os, "Enum": Enum, "auto": auto}
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


ACTUAL = load_adder_methods()
Result = ACTUAL["AddReqResult"]
PAGE = 64


def paged(n):
    return (n + PAGE - 1) // PAGE * PAGE


class Request:
    def __init__(self, length, ignore_eos=False, prefix=0):
        self.full_untruncated_fill_ids = range(length)
        self.origin_input_ids = range(length)
        self.prefix_indices = range(prefix)
        self.output_ids = []
        self.kv = NS(kv_allocated_len=prefix)
        self.sampling_params = NS(max_new_tokens=1, ignore_eos=ignore_eos)
        self.host_hit_length = 0
        self.retracted_stain = False
        self.last_node = None
        self.extend_range = None

    def set_extend_range(self, start, end):
        self.extend_range = NS(start=start, end=end, length=end - start)

    def needs_host_load_back(self):
        return False


class Adder(ACTUAL["PrefillAdder"]):
    def __init__(self, free, quantum):
        self.round_robin_requests = None
        self.free = free
        self.page_size = PAGE
        self.rem_total_token_offset = self.cur_rem_token_offset = 0
        self.rem_input_tokens = self.rem_chunk_tokens = quantum
        self.new_token_ratio = 1.0
        self.tree_cache = NS(disable=True)
        self.running_batch = NS(reqs=[])
        self.dllm_config = self.prefill_delayer_single_pass = None
        self.prefill_max_requests = self.rem_mamba_slots = None
        self.is_hybrid_swa = False
        self.can_run_list = []
        self.new_chunked_req = self.req_states = None
        self.log_hit_tokens = self.log_input_tokens = 0
        self.reprocessed_log_hit_tokens = self.reprocessed_log_input_tokens = 0

    @property
    def rem_total_tokens(self):
        return self.free - self.rem_total_token_offset

    @property
    def cur_rem_tokens(self):
        return self.free - self.cur_rem_token_offset

    def _mamba_gap_budget_for_req(self, req):
        return 0

    def _lock_node(self, node):
        return nullcontext()

    def _check_prefill_tile_budget(self, length):
        return None

    def _req_inc_lock_ref(self, req):
        pass

    def _account_prefill_cache_admission(self, req, prefix):
        pass


def reserve(adder, req, end=None):
    end = req.kv.kv_allocated_len if end is None else end
    remaining = len(req.full_untruncated_fill_ids) - end
    if remaining <= 0:
        return 0
    max_new = min(req.sampling_params.max_new_tokens, ACTUAL["CLIP_MAX_NEW_TOKENS"])
    return adder.ceil_paged_tokens(remaining) + adder.page_size + max_new


def simulate(
    lengths, capacity, quantum, ignore_eos, delay, row_limit, *, reserve_enabled=True
):
    requests = [Request(n, ignore_eos) for n in lengths]
    ready = list(requests)  # Logical FIFO; dual-queue equivalence checked separately.
    active, sending, finished = set(), {}, set()
    free = capacity
    for step in range(1000):
        for req, release_step in list(sending.items()):
            if release_step <= step:
                free += paged(req.kv.kv_allocated_len)
                del sending[req]
                finished.add(req)
        if len(finished) == len(requests):
            assert free == capacity
            return
        adder = Adder(free, quantum)
        adder.round_robin_requests = tuple(active) if reserve_enabled else None
        planned = {}
        pressure = False
        for req in list(ready):
            if adder.rem_chunk_tokens <= 0 or adder.rem_input_tokens <= 0:
                break
            if planned and adder.budget_state() != Result.CONTINUE:
                break
            is_new = req not in active
            if is_new:
                if pressure:
                    continue
                if len(active) + len(sending) >= row_limit:
                    pressure = True
                    continue
                before = len(adder.can_run_list)
                result = adder.add_one_req(req, False, None)
                added = len(adder.can_run_list) > before
                if not added:
                    pressure = result == Result.NO_TOKEN
                    if pressure:
                        continue
                    break
                active.add(req)
            else:
                adder.add_chunked_req(req)
            ready.remove(req)
            planned[req] = req.extend_range.end
            assert req.extend_range.length > 0
            if req.extend_range.end < len(req.full_untruncated_fill_ids):
                break  # Applies equally to new and resumed requests.
        if not planned:
            assert sending, (
                "no progress without a transfer to wait for",
                lengths,
                capacity,
            )
        for req, end in planned.items():
            free -= paged(end) - paged(req.kv.kv_allocated_len)
            req.kv.kv_allocated_len = end
            req.prefix_indices = range(end)
            if end == len(req.full_untruncated_fill_ids):
                active.remove(req)
                sending[req] = step + delay + 1
            else:
                ready.append(req)
        # Physical completion space, not the adder's temporary page padding.
        future_pages = sum(
            paged(len(r.full_untruncated_fill_ids)) - paged(r.kv.kv_allocated_len)
            for r in active
        )
        assert free >= future_pages >= 0, (lengths, capacity, step, free, future_pages)
    raise AssertionError("finite requests did not drain")


def main():
    try:
        simulate(
            [200_000, 200_000], 300_032, 16_384, False, 2, 8, reserve_enabled=False
        )
    except AssertionError as exc:
        assert isinstance(exc.args[0], tuple) and len(exc.args[0]) == 5, exc
    else:
        raise AssertionError("negative control failed to detect unreserved overcommit")
    simulate([200_000, 200_000], 300_032, 16_384, False, 2, 8)
    print("PASS: unreserved RR violates completion-space invariant; reserved RR drains")

    cases = 0
    rng = random.Random(0)
    for ignore_eos in (False, True):
        for quantum in (64, 128, 256):
            for pages in range(4, 25):
                capacity = pages * PAGE
                for _ in range(20):
                    lengths = [
                        rng.randrange(1, capacity - 128)
                        for _ in range(rng.randrange(2, 9))
                    ]
                    simulate(
                        lengths,
                        capacity,
                        quantum,
                        ignore_eos,
                        rng.randrange(5),
                        rng.randrange(1, 6),
                    )
                    cases += 1
    print(f"PASS: {cases} seeded paged RR cases using current PrefillAdder methods")

    # With no unfinished competitor, retain original near-capacity admission.
    # Example: length=130, capacity=256 passes the original raw-length gate;
    # ceil(130)+64+1=257 would otherwise stall this single request forever.
    for ignore_eos in (False, True):
        for capacity in (256, 512, 1024):
            for length in range(capacity - 128, capacity - 65):
                simulate([length], capacity, 64, ignore_eos, 2, 1)
    print("PASS: 378 near-capacity single requests retain original admission")

    # Every partial-page offset, including a continuation whose full tail fits.
    count = 0
    for prefix in range(1, 129):
        for tail in (1, 63, 64, 65, 127, 128, 129, 257):
            req = Request(prefix + tail, prefix=prefix)
            actual_pages = paged(prefix + tail) - paged(prefix)
            adder = Adder(paged(tail) + 128, 128)
            assert reserve(adder, req) >= actual_pages
            adder.add_chunked_req(req)
            delta = paged(req.extend_range.end) - paged(prefix)
            assert 0 <= delta <= actual_pages
            count += 1
    print(f"PASS: {count} partial-prefix continuation bounds")

    # New prefix-hit requests use the normal branch even with ignore_eos=True
    # when radix is enabled. The disabled-cache ignore_eos branch assumes zero
    # new-request prefix; simulate() exercises that separate path above.
    count = accepted = rejected = 0
    for ignore_eos in (False, True):
        for prefix in (1, 63, 64, 65, 127, 128):
            for tail in (1, 63, 64, 65, 129):
                for free in range(128, 1025, PAGE):
                    req = Request(prefix + tail, ignore_eos, prefix)
                    other = Request(257, prefix=64)
                    adder = Adder(free, 128)
                    adder.tree_cache.disable = False
                    cost, future = reserve(adder, req), reserve(adder, other)
                    adder.round_robin_requests = (other,)
                    adder.add_one_req(req, False, None)
                    if cost <= adder.free - future:
                        if req in adder.can_run_list:
                            delta = paged(req.extend_range.end) - paged(prefix)
                            remaining = paged(prefix + tail) - paged(
                                req.extend_range.end
                            )
                            other_remaining = paged(257) - paged(64)
                            assert free - delta >= remaining + other_remaining
                            accepted += 1
                    else:
                        assert not adder.can_run_list
                        rejected += 1
                    count += 1
    assert accepted and rejected
    print(
        f"PASS: {count} new prefix-hit gate cases ({accepted} admitted, {rejected} gate-rejected)"
    )

    # Pinning an evictable prefix can change a pass into a rejection. The new
    # gate must run inside the existing lock, not solely before add_one_req.
    adder = Adder(640, 128)
    req, other = Request(257, prefix=128), Request(257, prefix=64)
    demand = reserve(adder, req) + reserve(adder, other)
    assert demand <= adder.rem_total_tokens
    adder.round_robin_requests = (other,)

    @contextmanager
    def pin_prefix(node):
        adder.free -= paged(128)
        yield

    adder._lock_node = pin_prefix
    assert adder.add_one_req(req, False, None) == Result.NO_TOKEN
    assert not adder.can_run_list
    assert demand > adder.rem_total_tokens
    print("PASS: prefix pinning requires the post-lock budget check")

    # The full tail can already lie inside an allocated partial page. Do not
    # replace continuation with a new-request free-token gate.
    req, adder = Request(64, prefix=1), Adder(0, 128)
    assert adder.add_chunked_req(req) is None
    assert paged(req.extend_range.end) - paged(1) == 0
    print("PASS: zero-free-page continuation reuses its allocated tail page")

    # Boundary: passing the additional gate does not bypass existing admission.
    req = Request(64)
    adder = Adder(129, 64)
    assert reserve(adder, req, 0) == adder.rem_total_tokens
    assert adder.add_one_req(req, False, None) == Result.NO_TOKEN
    assert not adder.can_run_list
    print("PASS: existing equal-budget rejection retained")


if __name__ == "__main__":
    main()
