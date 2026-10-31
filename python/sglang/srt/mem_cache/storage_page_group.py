"""L3 objects spanning several HiCache pages (--hicache-storage-page-size).

Groups are fixed by absolute page position: group j holds pages [jG, jG+G), so
keys depend only on the token chain. A radix node writes only its own pages,
one object per segment and pool component, where a segment is the node's span
cut at group boundaries. A full group is keyed by its last page hash plus
``_g{G}``; a partial segment also carries its start offset in the group, so
segments written by different nodes never collide. Readers chain segments.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Optional, Sequence

import msgspec

# Absolute [lo, hi) page range inside one group.
Segment = tuple[int, int]


def group_segments(start_page: int, num_pages: int, group: int) -> list[Segment]:
    """[start_page, start_page + num_pages) cut at absolute group boundaries."""
    segments = []
    lo, end = start_page, start_page + num_pages
    while lo < end:
        hi = min((lo // group + 1) * group, end)
        segments.append((lo, hi))
        lo = hi
    return segments


def segment_suffix(segment: Segment, group: int) -> str:
    lo, hi = segment
    offset = lo % group
    if offset == 0 and hi - lo == group:
        return f"_g{group}"
    return f"_g{group}s{offset}"


def batch_spans(
    start_page: int, num_pages: int, group: int, batch_pages: int
) -> Iterator[Segment]:
    """Relative [lo, hi) batches whose inner cuts land on absolute group edges.

    ``group`` divides ``batch_pages``, so a batch is never empty.
    """
    lo = 0
    while lo < num_pages:
        hi = min(lo + batch_pages, num_pages)
        if group > 1 and hi < num_pages:
            hi = (start_page + hi) // group * group - start_page
        yield lo, hi
        lo = hi


class GroupReadPlan(msgspec.Struct, frozen=True):
    """Segment chain found by one existence query, in absolute pages."""

    # Reachable segment end -> the segment that reaches it.
    reach: dict[int, Segment]

    def restorable(self, start_page: int) -> list[int]:
        """Restorable prefix lengths in pages, relative to ``start_page``."""
        return sorted(hi - start_page for hi in self.reach)

    def chain(self, lo: int, hi: int) -> tuple[list[Segment], int]:
        """Segments covering [lo, end) for the last reachable end <= hi.

        Only the query's first page may be covered by a segment starting
        before ``lo``; its leading pages are read into scratch.
        """
        ends = [end for end in self.reach if lo < end <= hi]
        if not ends:
            return [], lo
        end = max(ends)
        chain, pos = [], end
        while pos > lo:
            segment = self.reach.get(pos)
            if segment is None:
                return [], lo
            chain.append(segment)
            pos = segment[0]
        chain.reverse()
        return chain, end


def _group_candidates(
    group_idx: int, group: int, start_page: int, end_page: int
) -> list[Segment]:
    """Every segment of one group ending in (start_page, end_page]."""
    base = group_idx * group
    return [
        (base + a, base + b)
        for b in range(1, group + 1)
        if start_page < base + b <= end_page
        for a in range(b)
    ]


def plan_group_reads(
    start_page: int,
    num_pages: int,
    group: int,
    probe: Callable[[list[Segment]], list[bool]],
) -> GroupReadPlan:
    """Longest segment chain from ``start_page`` in at most two probe rounds.

    Round 1 probes the full key of every inner group and every segment of a
    partial edge group. Round 2 probes the segments of a missing inner group
    whose left neighbour holds data, i.e. a radix node boundary. A missing
    group right after another missing one (a cold tail, or two boundaries in
    adjacent groups) is not expanded, so the chain stops there.
    """
    if num_pages == 0:
        return GroupReadPlan(reach={})
    end_page = start_page + num_pages
    groups = range(start_page // group, (end_page - 1) // group + 1)

    def is_inner(g: int) -> bool:
        return g * group >= start_page and (g + 1) * group <= end_page

    def full(g: int) -> Segment:
        return g * group, (g + 1) * group

    round1 = [
        segment
        for g in groups
        for segment in (
            [full(g)]
            if is_inner(g)
            else _group_candidates(g, group, start_page, end_page)
        )
    ]
    exists = dict(zip(round1, probe(round1)))

    def left_has_data(g: int) -> bool:
        left = g - 1
        if left < groups.start:
            return True
        if is_inner(left):
            return exists[full(left)]
        return any(ok and seg[1] == g * group for seg, ok in exists.items())

    round2 = [
        segment
        for g in groups
        if is_inner(g) and not exists[full(g)] and left_has_data(g)
        for segment in _group_candidates(g, group, start_page, end_page)
        if segment != full(g)
    ]
    if round2:
        exists.update(zip(round2, probe(round2)))
    return GroupReadPlan(
        reach=_reach(start_page, [seg for seg, ok in exists.items() if ok])
    )


def _reach(start_page: int, segments: list[Segment]) -> dict[int, Segment]:
    by_end: dict[int, list[Segment]] = {}
    for segment in segments:
        by_end.setdefault(segment[1], []).append(segment)
    reach: dict[int, Segment] = {}
    for hi in sorted(by_end):
        usable = [s for s in by_end[hi] if s[0] == start_page or s[0] in reach]
        lead = [s for s in by_end[hi] if s[0] < start_page]
        if usable:
            # Longest first: fewer objects to read.
            reach[hi] = min(usable)
        elif lead:
            # Shortest lead: fewer pages read into scratch.
            reach[hi] = max(lead)
    return reach


def regroup_layout(
    keys: Sequence[str],
    ptrs: Sequence[Any],
    sizes: Sequence[Any],
    *,
    per_page: int,
    base_page: int,
    segments: Sequence[Segment],
    group: int,
    lead: Optional[Callable[[int], tuple[list[list[int]], list[list[int]]]]] = None,
) -> tuple[list[str], list[list[int]], list[list[int]]]:
    """Fold a page-major layout into one object per segment and component.

    ``keys`` / ``ptrs`` / ``sizes`` hold ``per_page`` entries for every page
    from ``base_page``. A segment starting before ``base_page`` reads those
    pages into ``lead(count)``, which returns per-component scratch segments.
    """
    out_keys, out_ptrs, out_sizes = [], [], []
    for lo, hi in segments:
        suffix = segment_suffix((lo, hi), group)
        rel_lo, rel_hi = lo - base_page, hi - base_page
        lead_ptrs, lead_sizes = lead(-rel_lo) if rel_lo < 0 else (None, None)
        for comp in range(per_page):
            seg_ptrs = list(lead_ptrs[comp]) if lead_ptrs else []
            seg_sizes = list(lead_sizes[comp]) if lead_sizes else []
            for page in range(max(rel_lo, 0), rel_hi):
                idx = page * per_page + comp
                seg_ptrs.extend(_as_list(ptrs[idx]))
                seg_sizes.extend(_as_list(sizes[idx]))
            out_keys.append(keys[(rel_hi - 1) * per_page + comp] + suffix)
            out_ptrs.append(seg_ptrs)
            out_sizes.append(seg_sizes)
    return out_keys, out_ptrs, out_sizes


def _as_list(value: Any) -> list[int]:
    return list(value) if isinstance(value, Sequence) else [value]
