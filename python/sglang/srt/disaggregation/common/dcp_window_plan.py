"""CPU page-level DCP plans; never expand a window into token-index arrays."""

import numpy as np


def dcp_rank_layout(num_tokens, start_token, dcp_size=8):
    """(first window token, token count, packed row offset) for each rank."""
    if num_tokens < 0 or start_token < 0 or dcp_size <= 0:
        raise ValueError("Invalid DCP window geometry")
    layout = []
    offset = 0
    for rank in range(dcp_size):
        first = (rank - start_token) % dcp_size
        count = max(0, (num_tokens - first + dcp_size - 1) // dcp_size)
        layout.append((first, count, offset))
        offset += count
    return layout


class DCPDestinationPlan:
    """Immutable contiguous destination runs, built once per request/peer.

    Wire page tables are immutable for a room's lifetime. Snapshot run geometry
    rather than retaining/expanding a token table for every subsequent window.
    Offsets and lengths are in tokens, independent of layer width.
    """

    def __init__(self, pages, page_size):
        if page_size <= 0:
            raise ValueError("Page size must be positive")
        pages = np.asarray(pages)
        if pages.ndim != 1 or pages.dtype.kind not in "iu":
            raise ValueError(
                "DCP destination pages must be a one-dimensional integer array"
            )
        if pages.size and (
            np.any(pages < 0)
            or int(pages.max()) > (np.iinfo(np.int64).max // page_size) - 1
        ):
            raise ValueError("Invalid DCP destination page")
        pages = pages.astype(np.int64, copy=False)
        self.capacity = pages.size * page_size
        if not pages.size:
            self.starts = self.ends = self.destinations = np.empty(0, dtype=np.int64)
            return
        starts = np.r_[0, np.flatnonzero(np.diff(pages) != 1) + 1]
        self.starts = starts * page_size
        self.ends = np.r_[starts[1:], pages.size] * page_size
        self.destinations = pages[starts] * page_size

    def runs(self, start, count):
        """Return (packed offset, destination token, count), clipped to a window."""
        if start < 0 or count < 0 or (count and start + count > self.capacity):
            raise ValueError("Insufficient destination DCP pages")
        if count == 0:
            return []
        end = start + count
        lo = int(np.searchsorted(self.ends, start, side="right"))
        hi = int(np.searchsorted(self.starts, end, side="left"))
        result = []
        for index in range(lo, hi):
            first = max(start, int(self.starts[index]))
            last = min(end, int(self.ends[index]))
            destination = (
                int(self.destinations[index]) + first - int(self.starts[index])
            )
            result.append((first - start, destination, last - first))
        return result
