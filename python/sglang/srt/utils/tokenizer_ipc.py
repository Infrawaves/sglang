# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Startup-only IPC allocation for the existing CRC32 detokenizer router."""

import errno
import fcntl
import os
import re
import tempfile
import time
import weakref
import zlib
from pathlib import Path

_leases = weakref.WeakSet()


def _close_inherited_leases():
    # MM preprocessing can fork. A child must not keep its parent's slot alive,
    # nor LOCK_UN the shared open-file description and unlock the live parent.
    for lease in list(_leases):
        lease._close_fd()
    _leases.clear()


os.register_at_fork(after_in_child=_close_inherited_leases)


class TokenizerIpcLease:
    """Own one tokenizer slot until the worker exits or explicitly closes it.

    Acquire once during worker startup, before creating preprocessing children.
    Keep a strong reference for the worker's entire lifetime, including shutdown.
    A replacement worker reuses a free slot but gets a fresh IPC address so late
    output for the old worker cannot reach the replacement.
    """

    def __init__(self, slot: int, fd: int):
        self.slot = slot
        self.ipc_name = ""
        self._fd = fd
        self._owner_pid = os.getpid()
        _leases.add(self)

    @classmethod
    def acquire(
        cls,
        instance_id: str,
        tokenizer_worker_num: int,
        detokenizer_worker_num: int,
        *,
        timeout: float = 30.0,
    ):
        if not instance_id or not re.fullmatch(r"[A-Za-z0-9_-]+", instance_id):
            raise ValueError("A nonempty, path-safe server instance_id is required")
        if tokenizer_worker_num < 1 or detokenizer_worker_num < 1 or timeout < 0:
            raise ValueError("Worker counts must be positive and timeout nonnegative")

        # Lock files must stay in place: unlinking a live lock file would let a
        # second worker lock a different inode for the same slot. The instance
        # ID is shared via PortArgs and isolates independent servers on a host.
        directory = Path(tempfile.gettempdir()) / f"sglang-tokenizer-{instance_id}"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout
        while True:
            for slot in range(tokenizer_worker_num):
                fd = os.open(directory / f"slot-{slot}", os.O_CREAT | os.O_RDWR, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    os.close(fd)
                    if exc.errno in (errno.EACCES, errno.EAGAIN):
                        continue
                    raise

                lease = cls(slot, fd)
                try:
                    lease.ipc_name = cls._make_ipc_name(
                        slot % detokenizer_worker_num, detokenizer_worker_num
                    )
                    return lease
                except BaseException:
                    lease.close()
                    raise

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"No free tokenizer slot for instance {instance_id!r} "
                    f"({tokenizer_worker_num} workers). A replacement must wait "
                    "for the old worker to exit."
                )
            time.sleep(min(0.05, remaining))

    @staticmethod
    def _make_ipc_name(bucket: int, num_detokenizers: int) -> str:
        # Reserve unique short paths atomically, including the ipc:// prefix in
        # the hash exactly as MultiDetokenizerRouter._pick does. Rejected names
        # are removed immediately; no work is added to the output hot path.
        for _ in range(max(1024, num_detokenizers * 128)):
            fd, path = tempfile.mkstemp(prefix="sgt-")
            os.close(fd)
            ipc_name = f"ipc://{path}"
            if zlib.crc32(ipc_name.encode()) % num_detokenizers == bucket:
                return ipc_name
            os.unlink(path)
        raise RuntimeError(f"Could not generate an IPC address for bucket {bucket}")

    def _close_fd(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def close(self):
        """Release after the worker's IPC sockets are closed (or startup fails)."""
        if self._owner_pid != os.getpid() or self._fd is None:
            return
        try:
            if self.ipc_name:
                Path(self.ipc_name.removeprefix("ipc://")).unlink(missing_ok=True)
        finally:
            self._close_fd()
            _leases.discard(self)

    def __del__(self):
        self.close()
