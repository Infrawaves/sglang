"""CPU tests for balanced Tokenizer IPC leases and worker replacement."""

import collections
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
import uuid
import zlib
from unittest.mock import patch

from sglang.srt.utils import tokenizer_ipc
from sglang.srt.utils.tokenizer_ipc import TokenizerIpcLease
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


# Load only the dependency-free allocator in workers. Importing the test module
# under multiprocessing spawn would import the full test/torch stack eight times.
_WORKER = r"""
import importlib.util
import json
import os
import signal
import sys

spec = importlib.util.spec_from_file_location("tokenizer_ipc", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

def reply(value):
    print(json.dumps(value), flush=True)

reply({"ready": True})
assert sys.stdin.readline().strip() == "acquire"
lease = module.TokenizerIpcLease.acquire(
    sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), timeout=5.0
)
try:
    reply({"slot": lease.slot, "ipc_name": lease.ipc_name})
    for line in sys.stdin:
        command = line.strip()
        if command == "get":
            reply({"slot": lease.slot, "ipc_name": lease.ipc_name})
        elif command in ("fork", "fork-close", "fork-reuse-fd"):
            original_fd = lease._fd
            read_fd, write_fd = os.pipe()
            child_pid = os.fork()
            if child_pid == 0:
                os.close(read_fd)
                if command == "fork-close":
                    lease.close()
                elif command == "fork-reuse-fd":
                    # The at-fork hook closed this descriptor. Reuse its number
                    # for unrelated child work before cleaning up the old lease.
                    new_fd = os.open(os.devnull, os.O_RDONLY)
                    if new_fd != original_fd:
                        os.dup2(new_fd, original_fd)
                        os.close(new_fd)
                    lease.close()
                    os.fstat(original_fd)
                os.write(write_fd, b"ready")
                os.close(write_fd)
                while True:
                    signal.pause()
            os.close(write_fd)
            assert os.read(read_fd, 5) == b"ready"
            os.close(read_fd)
            reply({"child_pid": child_pid})
        elif command == "close":
            lease.close()
            reply({"closed": True})
        else:
            raise ValueError(command)
finally:
    lease.close()
"""


class TestTokenizerIpcLease(CustomTestCase):
    def _new_instance(self):
        instance_id = uuid.uuid4().hex
        self.addCleanup(
            shutil.rmtree,
            os.path.join(tempfile.gettempdir(), f"sglang-tokenizer-{instance_id}"),
            ignore_errors=True,
        )
        return instance_id

    def _acquire(self, instance_id, tokenizer_num, detokenizer_num, **kwargs):
        lease = TokenizerIpcLease.acquire(
            instance_id, tokenizer_num, detokenizer_num, **kwargs
        )
        self.addCleanup(lease.close)
        return lease

    @staticmethod
    def _stop_worker(process):
        # Include orphaned fork children, even when the original worker was killed.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()

    def _start_worker(self, instance_id, tokenizer_num, detokenizer_num):
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-c",
                _WORKER,
                tokenizer_ipc.__file__,
                instance_id,
                str(tokenizer_num),
                str(detokenizer_num),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        self.addCleanup(self._stop_worker, process)
        return process

    def _read_worker(self, process):
        readable, _, _ = select.select([process.stdout], [], [], 15)
        self.assertTrue(readable, f"Worker {process.pid} did not respond")
        line = process.stdout.readline()
        if not line:
            process.wait(timeout=5)
            self.fail(f"Worker {process.pid} exited: {process.stderr.read()}")
        return json.loads(line)

    @staticmethod
    def _send_worker(process, command):
        process.stdin.write(command + "\n")
        process.stdin.flush()

    def _request(self, process, command):
        self._send_worker(process, command)
        return self._read_worker(process)

    def test_concurrent_allocation_and_killed_worker_replacement(self):
        for detokenizer_num in (4, 8):
            with self.subTest(detokenizers=detokenizer_num):
                instance_id = self._new_instance()
                workers = [
                    self._start_worker(instance_id, 8, detokenizer_num)
                    for _ in range(8)
                ]
                for worker in workers:
                    self.assertEqual(self._read_worker(worker), {"ready": True})
                # All processes wait behind the same start barrier before racing
                # for slots; allocation cannot rely on sequential startup.
                for worker in workers:
                    self._send_worker(worker, "acquire")
                mappings = [self._read_worker(worker) for worker in workers]
                self.assertEqual({item["slot"] for item in mappings}, set(range(8)))
                self.assertEqual(len({item["ipc_name"] for item in mappings}), 8)
                buckets = collections.Counter(
                    zlib.crc32(item["ipc_name"].encode()) % detokenizer_num
                    for item in mappings
                )
                self.assertEqual(
                    buckets,
                    {bucket: 8 // detokenizer_num for bucket in range(detokenizer_num)},
                )
                for item in mappings:
                    self.assertEqual(
                        zlib.crc32(item["ipc_name"].encode()) % detokenizer_num,
                        item["slot"] % detokenizer_num,
                    )

                victim_index = next(
                    i for i, item in enumerate(mappings) if item["slot"] == 3
                )
                victim = workers[victim_index]
                victim.kill()  # SIGKILL deliberately bypasses normal close/atexit.
                victim.wait(timeout=5)
                replacement = self._start_worker(instance_id, 8, detokenizer_num)
                self.assertEqual(self._read_worker(replacement), {"ready": True})
                new_mapping = self._request(replacement, "acquire")
                self.assertEqual(new_mapping["slot"], 3)
                self.assertNotIn(
                    new_mapping["ipc_name"], [item["ipc_name"] for item in mappings]
                )
                self.assertEqual(
                    zlib.crc32(new_mapping["ipc_name"].encode()) % detokenizer_num,
                    3 % detokenizer_num,
                )
                for index, worker in enumerate(workers):
                    if index != victim_index:
                        self.assertEqual(self._request(worker, "get"), mappings[index])

    @unittest.skipUnless(hasattr(os, "fork"), "Requires POSIX fork")
    def test_fork_child_cannot_release_or_retain_parent_lease(self):
        for command in ("fork", "fork-close", "fork-reuse-fd"):
            with self.subTest(command=command):
                instance_id = self._new_instance()
                worker = self._start_worker(instance_id, 1, 8)
                self.assertEqual(self._read_worker(worker), {"ready": True})
                original = self._request(worker, "acquire")
                child_pid = self._request(worker, command)["child_pid"]
                os.kill(child_pid, 0)
                # Child-side close must not unlock the still-live parent's slot.
                with self.assertRaises(TimeoutError):
                    TokenizerIpcLease.acquire(instance_id, 1, 8, timeout=0.1)
                worker.kill()
                worker.wait(timeout=5)
                # Child remains alive, so an inherited lock FD would prevent this.
                os.kill(child_pid, 0)
                replacement = self._acquire(instance_id, 1, 8, timeout=2.0)
                self.assertEqual(replacement.slot, original["slot"])
                self.assertNotEqual(replacement.ipc_name, original["ipc_name"])

    def test_instances_have_independent_slots(self):
        first = self._acquire(self._new_instance(), 1, 4)
        second = self._acquire(self._new_instance(), 1, 4)
        self.assertEqual((first.slot, second.slot), (0, 0))
        self.assertNotEqual(first.ipc_name, second.ipc_name)

    def test_exhaustion_and_idempotent_close(self):
        instance_id = self._new_instance()
        original = self._acquire(instance_id, 1, 4)
        with self.assertRaises(TimeoutError):
            TokenizerIpcLease.acquire(instance_id, 1, 4, timeout=0.1)
        original.close()
        original.close()
        replacement = self._acquire(instance_id, 1, 4, timeout=0.1)
        self.assertEqual(replacement.slot, original.slot)
        self.assertNotEqual(replacement.ipc_name, original.ipc_name)
        # Repeated cleanup of an old lease cannot release the replacement's lock.
        original.close()
        with self.assertRaises(TimeoutError):
            TokenizerIpcLease.acquire(instance_id, 1, 4, timeout=0.1)

    def test_ipc_generation_failure_releases_slot(self):
        instance_id = self._new_instance()
        with patch.object(
            TokenizerIpcLease,
            "_make_ipc_name",
            side_effect=OSError("IPC creation failed"),
        ):
            with self.assertRaisesRegex(OSError, "IPC creation failed"):
                TokenizerIpcLease.acquire(instance_id, 1, 4, timeout=0.1)
        replacement = self._acquire(instance_id, 1, 4, timeout=0.1)
        self.assertEqual(replacement.slot, 0)

    def test_generated_names_support_zmq_round_trip(self):
        import zmq

        context = zmq.Context()
        self.addCleanup(context.term)
        for detokenizer_num in (4, 8):
            with self.subTest(detokenizers=detokenizer_num):
                instance_id = self._new_instance()
                lease = self._acquire(instance_id, 8, detokenizer_num)
                receiver = context.socket(zmq.PULL)
                sender = context.socket(zmq.PUSH)
                self.addCleanup(receiver.close, linger=0)
                self.addCleanup(sender.close, linger=0)
                receiver.setsockopt(zmq.RCVTIMEO, 2000)
                sender.setsockopt(zmq.SNDTIMEO, 2000)
                receiver.bind(lease.ipc_name)
                sender.connect(lease.ipc_name)
                messages = [
                    {"rid": "test-request", "token": token, "finished": False}
                    for token in (42, 13, 57, 9)
                ] + [{"rid": "test-request", "token": 2, "finished": True}]
                for message in messages:
                    sender.send_json(message)
                self.assertEqual([receiver.recv_json() for _ in messages], messages)


if __name__ == "__main__":
    unittest.main()
