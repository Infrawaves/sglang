"""FIFO rotation model; run directly with Python 3, without SGLang or a GPU.

Models full-prompt reservations and immediate release after completion. Excludes
page slack, prefix sharing, batch packing, GPU overlap, and transfer latency.
"""

from collections import deque


def simulate(lengths, capacity, *, dual=True, quantum=3, arrivals=None, ready_log=None):
    waiting, suspended, fifo = deque(), deque(), deque()
    done, resident, trace = {}, set(), []
    sequence = 0
    current = None
    arrivals = arrivals if arrivals is not None else {0: list(lengths)}
    for step in range(1000):
        # Match the normal PD loop: receive ready arrivals, then yield the
        # previous completed chunk. An unfinished request keeps the current slot
        # between iterations, even though its compute has completed.
        if not waiting and not suspended and not fifo and current is None:
            sequence = 0
        for rid in arrivals.get(step, []):
            assert rid not in done and 0 < lengths[rid] <= capacity
            done[rid] = 0
            (waiting if dual else fifo).append((sequence, rid))
            if ready_log is not None:
                ready_log.append((step, rid, sequence))
            sequence += 1

        if current is not None and (waiting or suspended or fifo):
            (suspended if dual else fifo).append((sequence, current))
            if ready_log is not None:
                ready_log.append((step, current, sequence))
            sequence += 1
            current = None

        if current is not None:
            # No competitor: continue directly, without consuming a new number.
            entry, source = (None, current), None
        elif dual:
            heads = [(q[0], q) for q in (waiting, suspended) if q]
            if not heads:
                if step >= max(arrivals, default=0):
                    return trace
                continue
            entry, source = min(heads, key=lambda item: item[0][0])
        else:
            if not fifo:
                if step >= max(arrivals, default=0):
                    return trace
                continue
            entry, source = fifo[0], fifo

        _, rid = entry
        reserved = sum(lengths[r] for r in resident)
        if rid not in resident and reserved + lengths[rid] > capacity:
            candidates = (
                suspended if dual else [e for e in fifo if e[1] in resident]
            )
            assert candidates, "pressure must leave a resident able to advance"
            entry = candidates[0]
            rid = entry[1]
            source = suspended if dual else fifo

        if source is not None:
            source.remove(entry)
        current = None
        resident.add(rid)
        done[rid] += min(quantum, lengths[rid] - done[rid])
        assert sum(lengths[r] for r in resident) <= capacity
        final = done[rid] == lengths[rid]
        trace.append((rid, done[rid], final))
        if final:
            resident.remove(rid)
        else:
            current = rid

        if dual:
            assert not ({r for _, r in waiting} & {r for _, r in suspended})
            assert current not in {r for _, r in waiting} | {r for _, r in suspended}
            for queue in (waiting, suspended):
                assert list(queue) == sorted(queue)
    raise AssertionError("finite queue failed to finish within model step limit")


def main():
    count = 0
    for capacity in range(4, 25):
        for a in range(1, capacity + 1):
            for b in range(1, capacity + 1):
                lengths = {"A": a, "B": b, "C": 1}
                trace = simulate(lengths, capacity)
                assert trace == simulate(lengths, capacity, dual=False)
                assert {r for r, _, final in trace if final} == set(lengths)
                count += 1
    assert count == 4886
    print(f"PASS: {count} single/dual FIFO capacity cases")

    lengths = {"A": 24, **{f"S{i}": 1 for i in range(8)}}
    arrivals = {0: ["A"], **{i + 1: [f"S{i}"] for i in range(8)}}
    trace = simulate(lengths, 64, arrivals=arrivals)
    assert trace == simulate(lengths, 64, dual=False, arrivals=arrivals)
    a_second = next(i for i, row in enumerate(trace) if row[:2] == ("A", 6))
    s0 = next(i for i, row in enumerate(trace) if row[0] == "S0")
    s1 = next(i for i, row in enumerate(trace) if row[0] == "S1")
    assert s0 < a_second, "ready arrivals precede the previous chunk's yield"
    assert a_second < s1, "later arrivals must not pass an older resume"
    print("PASS: ongoing arrivals preserve resume order")

    for dual in (False, True):
        ready_log = []
        simulate(
            {"A": 1, "B": 1, "C": 1}, 8, dual=dual,
            arrivals={0: ["A"], 1: ["B"], 4: ["C"]}, ready_log=ready_log,
        )
        assert ready_log == [(0, "A", 0), (1, "B", 0), (4, "C", 0)]
        ready_log = []
        boundary_trace = simulate(
            {"A": 6, "B": 1}, 8, dual=dual,
            arrivals={0: ["A"], 1: ["B"]}, ready_log=ready_log,
        )
        assert ready_log == [(0, "A", 0), (1, "B", 1), (1, "A", 2)]
        assert boundary_trace == [("A", 3, False), ("B", 1, True), ("A", 6, True)]
        ready_log = []
        late_trace = simulate(
            {"A": 12, "B": 1}, 16, dual=dual,
            arrivals={0: ["A"], 2: ["B"]}, ready_log=ready_log,
        )
        assert [rid for rid, _, _ in late_trace] == ["A", "A", "B", "A", "A"]
        assert ready_log == [(0, "A", 0), (2, "B", 1), (2, "A", 2)]
        ready_log = []
        solo = simulate({"A": 9}, 16, dual=dual, ready_log=ready_log)
        assert [end for _, end, _ in solo] == [3, 6, 9]
        assert ready_log == [(0, "A", 0)], "uncontested continuation must not renumber"
        ready_log = []
        simulate(
            {"A": 1, "B": 1, "C": 1}, 8, dual=dual,
            arrivals={0: ["A", "B"], 1: ["C"]}, ready_log=ready_log,
        )
        assert ready_log == [(0, "A", 0), (0, "B", 1), (1, "C", 2)]
    print("PASS: empty queues reset numbering; waiting/resuming work prevents reset")
    print("PASS: arrivals precede yield; uncontested continuation keeps its number")

    lengths = {"A": 24, "B": 24, "C": 2}
    ample = simulate(lengths, 64, quantum=8)
    tight = simulate(lengths, 32, quantum=8)
    assert [r for r, _, _ in ample[:3]] == ["A", "B", "C"]
    assert [r for r, _, _ in tight[:5]] == ["A", "A", "A", "B", "C"]
    assert tight[2][2] and tight[4][2] and not tight[3][2]
    print("ample:", ample)
    print("tight:", tight)
    print("PASS: pressure completes A, then serves C before B completes")


if __name__ == "__main__":
    main()
