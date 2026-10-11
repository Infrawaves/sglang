from __future__ import annotations

# Copyright 2023-2024 SGLang Team
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

"""
Mixin classes and utils for multi-http-worker mode
This file uses multiple processes to handle requests and tokenization, reducing the overhead of python and http server.
"""

import asyncio
import logging
import multiprocessing as multiprocessing
import os
import pickle
import signal
import sys
import threading
import zlib
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

import psutil
import setproctitle
import zmq
import zmq.asyncio

from sglang.srt.disaggregation.utils import TransferBackend
from sglang.srt.managers.disagg_service import start_disagg_service
from sglang.srt.managers.io_struct import (
    BaseBatchReq,
    BaseReq,
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
    ContinueGenerationReqInput,
    FreezeGCReq,
    PauseContinueBroadcastReq,
    PauseGenerationReqInput,
    TokenizerWorkerRegistrationReq,
    async_sock_recv,
    async_sock_send,
    sock_recv,
    sock_send,
    unwrap_from_pickle,
    wrap_as_pickle,
)
from sglang.srt.managers.load_snapshot import (
    create_load_snapshot_reader,
    zmq_reader_owner,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import (
    get_disagg,
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    configure_logger,
    kill_itself_when_parent_died,
    kill_process_tree,
)
from sglang.srt.utils.network import get_zmq_socket
from sglang.utils import get_exception_traceback

if TYPE_CHECKING:
    from sglang.srt.managers.detokenizer_manager import DetokenizerManager

logger = logging.getLogger(__name__)


class SocketMapping:
    def __init__(self):
        self._zmq_context = zmq.Context()
        self._mapping: Dict[str, zmq.Socket] = {}

    def clear_all_sockets(self):
        for socket in self._mapping.values():
            socket.close()
        self._mapping.clear()

    def _register_ipc_mapping(self, ipc_name: str, is_tokenizer: bool):
        type_str = "tokenizer" if is_tokenizer else "detokenizer"
        if ipc_name in self._mapping:
            logger.warning(f"{type_str} already registered {ipc_name=}, skipping...")
            return
        logger.info(f"Registering {type_str} {ipc_name=} in SocketMapping...")
        socket = get_zmq_socket(self._zmq_context, zmq.PUSH, ipc_name, False)
        self._mapping[ipc_name] = socket

    def send_output(self, ipc_name: str, output: Any, is_tokenizer: bool = False):
        if ipc_name is None:
            # Some unhandled cases
            logger.warning(f"IPC name is None, output type={type(output)}, skipping...")
            return

        if ipc_name not in self._mapping:
            self._register_ipc_mapping(ipc_name, is_tokenizer=is_tokenizer)
        sock_send(self._mapping[ipc_name], output)


def _extract_field_by_indices(
    output: Any, field_name: str, indices: List[int], check_length: bool = True
) -> Any:
    """Gather a per-request field at ``indices``; None when the field is absent.

    With ``check_length``, a field too short for any of ``indices`` counts as absent.
    """
    field = getattr(output, field_name, None)
    if field is None:
        return None

    should_wrap_result = field_name in ("customized_info", "time_stats")
    if should_wrap_result:
        field = unwrap_from_pickle(field)
        if field is None:
            return None

    if isinstance(field, dict):
        new_field = {
            k: [v[i] if len(v) > i else None for i in indices] for k, v in field.items()
        }
        if should_wrap_result:
            return wrap_as_pickle(new_field) if new_field else None
        return new_field

    if check_length and len(field) <= max(indices):
        return None

    new_field = [field[i] for i in indices]
    return wrap_as_pickle(new_field) if should_wrap_result else new_field


def _group_indices_by_key(keys: List[str]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    return groups


def _handle_output_by_index(output, i: int):
    return _handle_output_by_indices(output, [i])


def _handle_output_by_indices(output, indices: List[int]):
    """Slice a batch output down to ``indices``; unsupported types pass through.

    NOTE: A maintainable method is better here.
    """
    if isinstance(output, BatchTokenIDOutput):
        new_output = BatchTokenIDOutput(
            rids=[output.rids[i] for i in indices],
            spec_verify_ct=_extract_field_by_indices(output, "spec_verify_ct", indices),
            spec_num_correct_drafts=_extract_field_by_indices(
                output, "spec_num_correct_drafts", indices
            ),
            spec_correct_drafts_histogram=_extract_field_by_indices(
                output, "spec_correct_drafts_histogram", indices
            ),
            spec_num_block_accept_tokens=_extract_field_by_indices(
                output, "spec_num_block_accept_tokens", indices
            ),
            spec_num_cap_tokens=_extract_field_by_indices(
                output, "spec_num_cap_tokens", indices
            ),
            spec_cap_lens_histogram=_extract_field_by_indices(
                output, "spec_cap_lens_histogram", indices
            ),
            time_stats=_extract_field_by_indices(output, "time_stats", indices),
            finished_reasons=_extract_field_by_indices(
                output, "finished_reasons", indices
            ),
            decoded_texts=_extract_field_by_indices(output, "decoded_texts", indices),
            decode_ids=_extract_field_by_indices(output, "decode_ids", indices),
            read_offsets=_extract_field_by_indices(output, "read_offsets", indices),
            output_ids=_extract_field_by_indices(output, "output_ids", indices),
            skip_special_tokens=_extract_field_by_indices(
                output, "skip_special_tokens", indices
            ),
            spaces_between_special_tokens=_extract_field_by_indices(
                output, "spaces_between_special_tokens", indices
            ),
            no_stop_trim=_extract_field_by_indices(output, "no_stop_trim", indices),
            prompt_tokens=_extract_field_by_indices(output, "prompt_tokens", indices),
            completion_tokens=_extract_field_by_indices(
                output, "completion_tokens", indices
            ),
            reasoning_tokens=_extract_field_by_indices(
                output, "reasoning_tokens", indices
            ),
            cached_tokens=_extract_field_by_indices(output, "cached_tokens", indices),
            cached_tokens_details=_extract_field_by_indices(
                output, "cached_tokens_details", indices
            ),
            image_tokens=_extract_field_by_indices(output, "image_tokens", indices),
            audio_tokens=_extract_field_by_indices(output, "audio_tokens", indices),
            video_tokens=_extract_field_by_indices(output, "video_tokens", indices),
            input_token_logprobs_val=_extract_field_by_indices(
                output, "input_token_logprobs_val", indices, check_length=False
            ),
            input_token_logprobs_idx=_extract_field_by_indices(
                output, "input_token_logprobs_idx", indices, check_length=False
            ),
            output_token_logprobs_val=_extract_field_by_indices(
                output, "output_token_logprobs_val", indices, check_length=False
            ),
            output_token_logprobs_idx=_extract_field_by_indices(
                output, "output_token_logprobs_idx", indices, check_length=False
            ),
            input_top_logprobs_val=_extract_field_by_indices(
                output, "input_top_logprobs_val", indices, check_length=False
            ),
            input_top_logprobs_idx=_extract_field_by_indices(
                output, "input_top_logprobs_idx", indices, check_length=False
            ),
            input_top_logprobs_val_flat=_extract_field_by_indices(
                output, "input_top_logprobs_val_flat", indices, check_length=False
            ),
            input_top_logprobs_idx_flat=_extract_field_by_indices(
                output, "input_top_logprobs_idx_flat", indices, check_length=False
            ),
            input_top_logprobs_flat_null_prefix=_extract_field_by_indices(
                output,
                "input_top_logprobs_flat_null_prefix",
                indices,
                check_length=False,
            ),
            output_top_logprobs_val=_extract_field_by_indices(
                output, "output_top_logprobs_val", indices, check_length=False
            ),
            output_top_logprobs_idx=_extract_field_by_indices(
                output, "output_top_logprobs_idx", indices, check_length=False
            ),
            input_token_ids_logprobs_val=_extract_field_by_indices(
                output, "input_token_ids_logprobs_val", indices, check_length=False
            ),
            input_token_ids_logprobs_idx=_extract_field_by_indices(
                output, "input_token_ids_logprobs_idx", indices, check_length=False
            ),
            output_token_ids_logprobs_val=_extract_field_by_indices(
                output, "output_token_ids_logprobs_val", indices, check_length=False
            ),
            output_token_ids_logprobs_idx=_extract_field_by_indices(
                output, "output_token_ids_logprobs_idx", indices, check_length=False
            ),
            output_token_entropy_val=_extract_field_by_indices(
                output, "output_token_entropy_val", indices, check_length=False
            ),
            output_token_sampling_mask=_extract_field_by_indices(
                output, "output_token_sampling_mask", indices, check_length=False
            ),
            output_token_sampling_logprobs=_extract_field_by_indices(
                output, "output_token_sampling_logprobs", indices, check_length=False
            ),
            output_hidden_states=_extract_field_by_indices(
                output, "output_hidden_states", indices, check_length=False
            ),
            routed_experts=_extract_field_by_indices(
                output, "routed_experts", indices, check_length=False
            ),
            indexer_topk=_extract_field_by_indices(
                output, "indexer_topk", indices, check_length=False
            ),
            retraction_counts=_extract_field_by_indices(
                output, "retraction_counts", indices
            ),
            weight_versions=_extract_field_by_indices(
                output, "weight_versions", indices
            ),
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            beam_search_output=_extract_field_by_indices(
                output, "beam_search_output", indices, check_length=True
            ),
            token_steps=_extract_field_by_indices(
                output, "token_steps", indices, check_length=False
            ),
            customized_info=_extract_field_by_indices(
                output, "customized_info", indices, check_length=False
            ),
            dp_ranks=_extract_field_by_indices(
                output, "dp_ranks", indices, check_length=False
            ),
        )
    elif isinstance(output, BatchEmbeddingOutput):
        new_output = BatchEmbeddingOutput(
            rids=[output.rids[i] for i in indices],
            finished_reasons=_extract_field_by_indices(
                output, "finished_reasons", indices
            ),
            embeddings=_extract_field_by_indices(output, "embeddings", indices),
            prompt_tokens=_extract_field_by_indices(output, "prompt_tokens", indices),
            cached_tokens=_extract_field_by_indices(output, "cached_tokens", indices),
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
        )
    elif isinstance(output, BatchStrOutput):
        new_output = BatchStrOutput(
            rids=[output.rids[i] for i in indices],
            spec_verify_ct=_extract_field_by_indices(output, "spec_verify_ct", indices),
            spec_num_correct_drafts=_extract_field_by_indices(
                output, "spec_num_correct_drafts", indices
            ),
            spec_correct_drafts_histogram=_extract_field_by_indices(
                output, "spec_correct_drafts_histogram", indices
            ),
            spec_num_block_accept_tokens=_extract_field_by_indices(
                output, "spec_num_block_accept_tokens", indices
            ),
            spec_num_cap_tokens=_extract_field_by_indices(
                output, "spec_num_cap_tokens", indices
            ),
            spec_cap_lens_histogram=_extract_field_by_indices(
                output, "spec_cap_lens_histogram", indices
            ),
            time_stats=_extract_field_by_indices(output, "time_stats", indices),
            finished_reasons=_extract_field_by_indices(
                output, "finished_reasons", indices
            ),
            output_strs=_extract_field_by_indices(output, "output_strs", indices),
            output_ids=_extract_field_by_indices(output, "output_ids", indices),
            prompt_tokens=_extract_field_by_indices(output, "prompt_tokens", indices),
            completion_tokens=_extract_field_by_indices(
                output, "completion_tokens", indices
            ),
            reasoning_tokens=_extract_field_by_indices(
                output, "reasoning_tokens", indices
            ),
            cached_tokens=_extract_field_by_indices(output, "cached_tokens", indices),
            cached_tokens_details=_extract_field_by_indices(
                output, "cached_tokens_details", indices
            ),
            image_tokens=_extract_field_by_indices(output, "image_tokens", indices),
            audio_tokens=_extract_field_by_indices(output, "audio_tokens", indices),
            video_tokens=_extract_field_by_indices(output, "video_tokens", indices),
            input_token_logprobs_val=_extract_field_by_indices(
                output, "input_token_logprobs_val", indices, check_length=False
            ),
            input_token_logprobs_idx=_extract_field_by_indices(
                output, "input_token_logprobs_idx", indices, check_length=False
            ),
            output_token_logprobs_val=_extract_field_by_indices(
                output, "output_token_logprobs_val", indices, check_length=False
            ),
            output_token_logprobs_idx=_extract_field_by_indices(
                output, "output_token_logprobs_idx", indices, check_length=False
            ),
            input_top_logprobs_val=_extract_field_by_indices(
                output, "input_top_logprobs_val", indices, check_length=False
            ),
            input_top_logprobs_idx=_extract_field_by_indices(
                output, "input_top_logprobs_idx", indices, check_length=False
            ),
            input_top_logprobs_val_flat=_extract_field_by_indices(
                output, "input_top_logprobs_val_flat", indices, check_length=False
            ),
            input_top_logprobs_idx_flat=_extract_field_by_indices(
                output, "input_top_logprobs_idx_flat", indices, check_length=False
            ),
            input_top_logprobs_flat_null_prefix=_extract_field_by_indices(
                output,
                "input_top_logprobs_flat_null_prefix",
                indices,
                check_length=False,
            ),
            output_top_logprobs_val=_extract_field_by_indices(
                output, "output_top_logprobs_val", indices, check_length=False
            ),
            output_top_logprobs_idx=_extract_field_by_indices(
                output, "output_top_logprobs_idx", indices, check_length=False
            ),
            input_token_ids_logprobs_val=_extract_field_by_indices(
                output, "input_token_ids_logprobs_val", indices, check_length=False
            ),
            input_token_ids_logprobs_idx=_extract_field_by_indices(
                output, "input_token_ids_logprobs_idx", indices, check_length=False
            ),
            output_token_ids_logprobs_val=_extract_field_by_indices(
                output, "output_token_ids_logprobs_val", indices, check_length=False
            ),
            output_token_ids_logprobs_idx=_extract_field_by_indices(
                output, "output_token_ids_logprobs_idx", indices, check_length=False
            ),
            output_token_entropy_val=_extract_field_by_indices(
                output, "output_token_entropy_val", indices, check_length=False
            ),
            output_token_sampling_mask=_extract_field_by_indices(
                output, "output_token_sampling_mask", indices, check_length=False
            ),
            output_token_sampling_logprobs=_extract_field_by_indices(
                output, "output_token_sampling_logprobs", indices, check_length=False
            ),
            output_hidden_states=_extract_field_by_indices(
                output, "output_hidden_states", indices, check_length=False
            ),
            routed_experts=_extract_field_by_indices(
                output, "routed_experts", indices, check_length=False
            ),
            indexer_topk=_extract_field_by_indices(
                output, "indexer_topk", indices, check_length=False
            ),
            customized_info=_extract_field_by_indices(
                output, "customized_info", indices, check_length=False
            ),
            dp_ranks=_extract_field_by_indices(
                output, "dp_ranks", indices, check_length=False
            ),
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            retraction_counts=_extract_field_by_indices(
                output, "retraction_counts", indices
            ),
            beam_search_output=_extract_field_by_indices(
                output, "beam_search_output", indices, check_length=True
            ),
            weight_versions=_extract_field_by_indices(
                output, "weight_versions", indices
            ),
            token_steps=_extract_field_by_indices(
                output, "token_steps", indices, check_length=False
            ),
        )
    else:
        new_output = output
    return new_output


class MultiHttpWorkerDetokenizerMixin:
    """Mixin class for DetokenizerManager"""

    def maybe_clear_socket_mapping(self: DetokenizerManager):
        if hasattr(self, "socket_mapping"):
            self.socket_mapping.clear_all_sockets()

    def multi_http_worker_event_loop(self: DetokenizerManager):
        """The event loop that handles requests, for multi multi-http-worker mode"""
        self.socket_mapping = SocketMapping()
        # Watchdog wiring mirrors DetokenizerManager.event_loop: the watchdog is
        # paused while waiting for input and fed once per processed message.
        while True:
            with self.soft_watchdog.disable():
                recv_obj = sock_recv(self.recv_from_scheduler)
            output = self._request_dispatcher(recv_obj)
            if output is not None:
                # Fan out the output back to the originating tokenizer worker(s).
                # In multi-detokenizer mode the upstream MultiDetokenizerRouter may
                # forward either batched or single requests, so handle both shapes.
                if isinstance(recv_obj, BaseBatchReq):
                    groups = _group_indices_by_key(recv_obj.http_worker_ipcs)
                    for ipc_name, indices in groups.items():
                        self.socket_mapping.send_output(
                            ipc_name,
                            _handle_output_by_indices(output, indices),
                            is_tokenizer=True,
                        )
                elif isinstance(recv_obj, BaseReq):
                    self.socket_mapping.send_output(
                        recv_obj.http_worker_ipc, output, is_tokenizer=True
                    )
                else:
                    raise ValueError(
                        f"multi_http_worker_event_loop got unexpected req type {type(recv_obj)}"
                    )
            self.soft_watchdog.feed()


class MultiTokenizerRouter:
    """A router between tokenizer managers and the scheduler/detokenizer manager.

    Forward: tokenizer managers → router → scheduler.
    Backward: detokenizer manager → router → tokenizer managers.
    Also broadcasts pause/continue to all tokenizer managers for consistent is_pause state.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        self.server_args = server_args
        self.startup_time: Optional[Dict[str, Any]] = None
        context = zmq.asyncio.Context(3)
        self.recv_from_detokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.tokenizer_ipc_name, True
        )
        self.send_to_scheduler = get_zmq_socket(
            context, zmq.PUSH, port_args.scheduler_input_ipc_name, True
        )
        self.receive_from_worker = get_zmq_socket(
            context, zmq.PULL, port_args.tokenizer_worker_ipc_name, True
        )
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._task = asyncio.run_coroutine_threadsafe(
            self.router_worker_obj(), self._loop
        )
        self._handle_task = asyncio.run_coroutine_threadsafe(
            print_exception_wrapper(self.handle_loop), self._loop
        )

        # In multi-tokenizer mode the N TokenizerWorker processes cannot each
        # bind the zmq PULL socket used for load snapshots, so the single
        # MultiTokenizerRouter process owns it (zmq -> SHM) and the workers
        # read SHM only. Drain it event-driven via the socket's fd instead of
        # polling on a timer.
        self.load_snapshot_reader = None
        if zmq_reader_owner("MultiTokenizerRouter"):
            self.load_snapshot_reader = create_load_snapshot_reader(
                port_args, caller="MultiTokenizerRouter"
            )
            self._loop.call_soon_threadsafe(self._register_load_snapshot_reader)

        self.disaggregation_bootstrap_server = start_disagg_service()

        # Worker IPC names for pause/continue broadcasting
        self.all_worker_ipcs: set[str] = set()
        # Shared socket mapping (both coroutines run on self._loop, so safe)
        self.socket_mapping = SocketMapping()

    def set_startup_time(self, startup_time: Dict[str, Any]) -> None:
        self.startup_time = startup_time

    def _run_loop(self):
        self._loop.run_forever()

    def _register_load_snapshot_reader(self):
        """Drain zmq load snapshots into SHM whenever the PULL socket is readable.

        zmq exposes an edge-triggered fd; ``poll()`` drains it until empty, which
        also re-arms the fd, so TokenizerWorkers reading SHM stay up to date
        without any timer.
        """
        assert self.load_snapshot_reader is not None
        self._loop.add_reader(
            self.load_snapshot_reader.fileno(), self.load_snapshot_reader.poll
        )
        # Drain anything already queued before the fd was registered.
        self.load_snapshot_reader.poll()

    async def router_worker_obj(self):
        """Forward path: workers → scheduler, with pause/continue broadcast."""
        while True:
            recv_obj = await async_sock_recv(self.receive_from_worker)

            if isinstance(recv_obj, TokenizerWorkerRegistrationReq):
                if recv_obj.worker_ipc_name not in self.all_worker_ipcs:
                    self.all_worker_ipcs.add(recv_obj.worker_ipc_name)
                    logger.info(
                        f"Router registered worker IPC: {recv_obj.worker_ipc_name} "
                        f"(total: {len(self.all_worker_ipcs)})"
                    )
                continue

            if isinstance(
                recv_obj, (PauseGenerationReqInput, ContinueGenerationReqInput)
            ):
                # Broadcast to ALL workers so every worker's is_pause is set
                is_pause = isinstance(recv_obj, PauseGenerationReqInput)
                broadcast = PauseContinueBroadcastReq(is_pause=is_pause)
                for ipc_name in self.all_worker_ipcs:
                    self.socket_mapping.send_output(ipc_name, broadcast)
                # Forward to scheduler rank 0 (it broadcasts to all TP/PP/DP
                # ranks internally). Skip for abort mode which drains via polling.
                if not (
                    isinstance(recv_obj, PauseGenerationReqInput)
                    and recv_obj.mode == "abort"
                ):
                    await async_sock_send(self.send_to_scheduler, recv_obj)
                continue

            await async_sock_send(self.send_to_scheduler, recv_obj)

    async def handle_loop(self):
        """Backward path: detokenizer → route results to correct worker."""
        while True:
            recv_obj = await async_sock_recv(self.recv_from_detokenizer)
            await self._distribute_result_to_workers(recv_obj)

    async def _distribute_result_to_workers(self, recv_obj):
        if isinstance(recv_obj, BaseReq):
            ipc_names = [recv_obj.http_worker_ipc]
        elif isinstance(recv_obj, BaseBatchReq):
            ipc_names = recv_obj.http_worker_ipcs
        else:
            raise ValueError(f"Unknown recv_obj type: {type(recv_obj)}")

        for i, ipc_name in enumerate(ipc_names):
            new_recv_obj = _handle_output_by_index(recv_obj, i)
            self.socket_mapping.send_output(ipc_name, new_recv_obj)


class MultiDetokenizerRouter:
    """Route scheduler outputs to one of N DetokenizerManager workers.

    Hash each request's RID so a hot tokenizer can use all detokenizers while
    every stream stays on one worker. The original ``http_worker_ipc`` remains
    the return address and is the routing fallback for messages without a RID.
    """

    def __init__(self, ipc_name_list: List[str], port_args: PortArgs):
        self.ipc_name_list = ipc_name_list
        self.num_workers = len(ipc_name_list)
        self.socket_mapping = SocketMapping()
        context = zmq.Context(2)
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, port_args.detokenizer_ipc_name, True
        )

        logger.info(
            "Detokenizer routing: RID hash across %s workers (IPC fallback without RID)",
            self.num_workers,
        )

    def _pick(self, rid: Optional[str], http_worker_ipc: str) -> str:
        key = rid or http_worker_ipc
        return self.ipc_name_list[zlib.crc32(key.encode()) % self.num_workers]

    def _send(self, ipc_name: str, obj: Any) -> None:
        self.socket_mapping.send_output(ipc_name, obj, is_tokenizer=False)

    def event_loop(self):
        while True:
            recv_obj = sock_recv(self.recv_from_scheduler)

            # FreezeGCReq must freeze every detokenizer process.
            if isinstance(recv_obj, FreezeGCReq):
                for ipc in self.ipc_name_list:
                    self._send(ipc, recv_obj)
                continue

            # Keep the return address separate from the request routing key.
            if isinstance(recv_obj, BaseReq):
                assert recv_obj.http_worker_ipc is not None, (
                    f"Single req {recv_obj.rid=} missing http_worker_ipc"
                )
                self._send(self._pick(recv_obj.rid, recv_obj.http_worker_ipc), recv_obj)
                continue

            # Batch request.
            if isinstance(recv_obj, BaseBatchReq):
                # Idle/no-op batch (rids=[]): broadcast to all detokenizers
                if not recv_obj.rids:
                    for ipc in self.ipc_name_list:
                        self._send(ipc, recv_obj)
                    continue

                ipcs = recv_obj.http_worker_ipcs
                assert (
                    ipcs is not None
                    and len(ipcs) == len(recv_obj.rids)
                    and all(x is not None for x in ipcs)
                ), f"Batch req {recv_obj.rids=} has invalid http_worker_ipcs"

                # One sub-batch per detokenizer; items keep their tokenizer return address.
                targets = [
                    self._pick(rid=rid, http_worker_ipc=ipc)
                    for rid, ipc in zip(recv_obj.rids, ipcs)
                ]
                for target, indices in _group_indices_by_key(targets).items():
                    part = _handle_output_by_indices(recv_obj, indices)
                    if part is recv_obj:
                        raise TypeError(f"Cannot split {type(recv_obj)}")
                    part.http_worker_ipcs = [ipcs[i] for i in indices]
                    self._send(target, part)
                continue

            raise ValueError(
                f"MultiDetokenizerRouter got unsupported type {type(recv_obj)}"
            )


def run_multi_detokenizer_router_process(
    ipc_name_list: List[str],
    server_args: ServerArgs,
    port_args: PortArgs,
):
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang::detokenizer_router")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    router = None
    try:
        router = MultiDetokenizerRouter(ipc_name_list, port_args)
        router.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"MultiDetokenizerRouter hit an exception: {traceback}")
        if router is not None:
            router.socket_mapping.clear_all_sockets()
        parent_process.send_signal(signal.SIGQUIT)


class TokenizerWorker(TokenizerManager):
    """Tokenizer Worker in multi-http-worker mode"""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        setproctitle.setproctitle(f"sglang::tokenizer_worker:{os.getpid()}")
        import torch

        torch.set_num_threads(1)
        super().__init__(
            server_args,
            port_args,
            start_pd_bootstrap_service=False,
        )

        self.worker_id = os.getpid()
        self.tokenizer_ipc_name = port_args.tokenizer_ipc_name

        # For PD disaggregation
        self.disaggregation_transfer_backend = TransferBackend(
            get_disagg().disaggregation_transfer_backend
        )

        # Register this worker with the router for pause/continue broadcasting
        reg = TokenizerWorkerRegistrationReq(worker_ipc_name=self.tokenizer_ipc_name)
        self._dispatch_to_scheduler(reg)

        # Future for awaiting pause/continue broadcast confirmation
        self._pause_continue_future: Optional[asyncio.Future] = None

        # Register PauseContinueBroadcastReq in the result dispatcher so
        # handle_loop routes it to _handle_pause_continue_broadcast
        from sglang.utils import TypeBasedDispatcher

        self._result_dispatcher += TypeBasedDispatcher(
            [(PauseContinueBroadcastReq, self._handle_pause_continue_broadcast)]
        )

    async def pause_generation(self, obj: PauseGenerationReqInput):
        loop = asyncio.get_event_loop()
        self._pause_continue_future = loop.create_future()
        # Send to router which will broadcast to all workers
        # (router also handles forwarding to scheduler for non-abort modes)
        self._dispatch_to_scheduler(obj)
        await self._pause_continue_future

        if obj.mode == "abort":
            # Abort polling: only the originator checks its own lock state
            while True:
                self.abort_request(abort_all=True)
                is_locked = await self.model_update_lock.is_locked()
                if not is_locked:
                    break
                await asyncio.sleep(1.0)

    async def continue_generation(self, obj: ContinueGenerationReqInput):
        loop = asyncio.get_event_loop()
        self._pause_continue_future = loop.create_future()
        self._dispatch_to_scheduler(obj)
        await self._pause_continue_future

    def _handle_pause_continue_broadcast(self, obj: PauseContinueBroadcastReq):
        """Called from handle_loop when a broadcast arrives from the router."""
        loop = asyncio.get_event_loop()
        loop.create_task(self._apply_pause_continue_broadcast(obj))

    async def _apply_pause_continue_broadcast(self, obj: PauseContinueBroadcastReq):
        """Apply pause/continue state under the condition lock."""
        async with self.is_pause_cond:
            if obj.is_pause:
                self.is_pause = True
            else:
                self.is_pause = False
                self.is_pause_cond.notify_all()

        # Resolve the pending future if this worker initiated the pause/continue
        if self._pause_continue_future and not self._pause_continue_future.done():
            self._pause_continue_future.set_result(True)
            self._pause_continue_future = None


def get_tokenizer_worker_class(server_args: ServerArgs) -> Type[TokenizerWorker]:
    worker_class = server_args.get_tokenizer_worker_class()
    if not isinstance(worker_class, type) or not issubclass(
        worker_class, TokenizerWorker
    ):
        raise TypeError(
            "ServerArgs.get_tokenizer_worker_class() must return a TokenizerWorker "
            f"subclass, got {worker_class!r}"
        )

    return worker_class


async def print_exception_wrapper(func):
    """
    Sometimes an asyncio function does not print exception.
    We do another wrapper to handle the exception.
    """
    try:
        await func()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"MultiTokenizerRouter hit an exception: {traceback}")
        if hasattr(func, "__self__") and isinstance(
            func.__self__, MultiTokenizerRouter
        ):
            func.__self__.dump_requests_before_crash()
        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(1)


def get_main_process_id() -> int:
    """Get the main process ID."""
    return multiprocessing.current_process()._parent_pid


def write_to_shared_memory(obj, name: str) -> shared_memory.SharedMemory:
    """Write data to shared memory"""
    serialized = pickle.dumps(obj)
    size = len(serialized)
    try:
        # Try to open existing shared memory
        shm = shared_memory.SharedMemory(name=name)
        # If size is insufficient, close and recreate
        if shm.size < size:
            shm.close()
            shm.unlink()
            shm = shared_memory.SharedMemory(create=True, size=size, name=name)
    except FileNotFoundError:
        # If not present, create new shared memory
        shm = shared_memory.SharedMemory(create=True, size=size, name=name)

    shm.buf[:size] = serialized
    return shm


def read_from_shared_memory(name: str) -> Any:
    """Read data from shared memory"""
    try:
        shm = shared_memory.SharedMemory(name=name)
        data = pickle.loads(bytes(shm.buf))
        shm.close()
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Shared memory {name} not found")


def write_data_for_multi_tokenizer(
    port_args: PortArgs,
    server_args: ServerArgs,
    scheduler_info: Dict,
):
    """Write args information to share memory for multi-tokenizer"""
    # get main process ID
    main_pid = get_main_process_id()
    current_pid = os.getpid()
    logger.info(f"main process ID: {main_pid}, current process ID: {current_pid}")
    args = (port_args, server_args, scheduler_info)
    args_shm = write_to_shared_memory(args, f"multi_tokenizer_args_{current_pid}")
    args_shm.close()

    return args_shm
