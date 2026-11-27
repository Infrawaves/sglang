"""Demand image preprocessing between the scheduler and the tokenizer.

Scheduler: one MaterializeImagesReq per newly needed image set, features
attached only from the reply the request waits for, and every other reply
(late, stale, for a finished or aborted request) released. Tokenizer: every
MaterializeImagesReq gets exactly one ImagesReadyReq, and prepared transport
leases are cancelled whenever the reply carries no features.
"""

import asyncio
import gc
import weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import msgspec
import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    GenerateReqInput,
    ImagesReadyReq,
    MaterializeImagesReq,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.req_types import MultimodalProcessorOutput  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.tokenizer_manager import (  # noqa: E402
    ReqState,
    TokenizerManager,
)
from sglang.srt.multimodal.demand_preprocess import (  # noqa: E402
    DemandImageSources,
    ImageDemandStatus,
)

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _Item:
    def __init__(self, offsets, hash_=1, feature=None):
        self.offsets = offsets
        self.hash = hash_
        self.pad_value = hash_ + 100
        self.feature = feature
        self.model_specific_data = {}
        self.release_transport_proxies = Mock()


class _Req:
    """Weak-referenceable stand-in for the scheduler's Req."""

    def __init__(self, rid, spans):
        self.rid = rid
        self.mm_demand = None
        self.multimodal_inputs = SimpleNamespace(
            mm_items=[_Item([span], hash_=index) for index, span in enumerate(spans)]
        )
        self._finished = False

    def finished(self):
        return self._finished


_SPANS = [(3, 6), (10, 13), (20, 29)]


def _scheduler():
    scheduler = object.__new__(Scheduler)
    scheduler._image_demand_reqs = weakref.WeakValueDictionary()
    scheduler.ipc_channels = SimpleNamespace(send_to_tokenizer=Mock())
    return scheduler


def _registered(scheduler, rid="r", ticket=3):
    req = _Req(rid, _SPANS)
    scheduler._register_image_demand(req, ticket)
    return req


def _reply(req, materialization_id, indices, **kwargs):
    targets = req.multimodal_inputs.mm_items
    items = [
        _Item(list(targets[index].offsets), targets[index].hash, torch.ones(2))
        for index in indices
    ]
    return ImagesReadyReq(
        rid=req.rid,
        ticket=req.mm_demand.ticket,
        materialization_id=materialization_id,
        image_indices=list(indices),
        mm_inputs=SimpleNamespace(mm_items=items),
        **kwargs,
    )


def _wait(scheduler, req, prefix_len):
    failures = []
    assert req.mm_demand.blocks(prefix_len)
    scheduler._await_demand_images(req, failures)
    return failures


def test_waiting_request_asks_once_per_newly_needed_image_set():
    scheduler = _scheduler()
    req = _registered(scheduler)
    send = scheduler.ipc_channels.send_to_tokenizer.send_output

    assert _wait(scheduler, req, 12) == []
    assert _wait(scheduler, req, 12) == []
    assert send.call_count == 1
    message, owner = send.call_args.args
    assert isinstance(message, MaterializeImagesReq)
    assert (message.rid, message.ticket) == ("r", 3)
    assert (message.materialization_id, message.image_indices) == (0, [1, 2])
    assert owner is req

    # The prefix shrinks below image 0: only image 0 is asked for.
    _wait(scheduler, req, 0)
    assert send.call_count == 2
    assert send.call_args.args[0].image_indices == [0]


def test_reply_attaches_features_and_unblocks_admission():
    scheduler = _scheduler()
    req = _registered(scheduler)
    _wait(scheduler, req, 12)
    reply = _reply(req, 0, [1, 2])
    sent = list(reply.mm_inputs.mm_items)

    scheduler.handle_images_ready(reply)

    targets = req.multimodal_inputs.mm_items
    assert targets[0].feature is None
    assert torch.equal(targets[1].feature, torch.ones(2))
    assert torch.equal(targets[2].feature, torch.ones(2))
    assert all(item.feature is None for item in sent)
    assert not any(item.release_transport_proxies.called for item in sent)
    assert not req.mm_demand.blocks(12)


@pytest.mark.parametrize(
    "late",
    [
        "finished",  # finished meanwhile (e.g. client abort while waiting)
        "aborted",  # release_mm_inputs_on_abort dropped the inputs
        "gone",  # the request object no longer exists
        "stale_ticket",  # reply for an older incarnation of the rid
        "duplicate",  # the round was already answered
    ],
)
def test_late_or_stale_reply_only_releases_its_features(late):
    scheduler = _scheduler()
    req = _registered(scheduler)
    _wait(scheduler, req, 0)
    reply = _reply(req, 0, [0, 1, 2])
    sent = list(reply.mm_inputs.mm_items)
    if late == "finished":
        req._finished = True
    elif late == "aborted":
        req.multimodal_inputs = None
    elif late == "gone":
        # The mock's call records are the last strong references.
        scheduler.ipc_channels.send_to_tokenizer.reset_mock()
        del req
        gc.collect()
        assert "r" not in scheduler._image_demand_reqs
    elif late == "stale_ticket":
        reply.ticket += 1
    elif late == "duplicate":
        scheduler.handle_images_ready(_reply(req, 0, [0, 1, 2]))

    scheduler.handle_images_ready(reply)

    for item in sent:
        item.release_transport_proxies.assert_called_once_with()
        assert item.feature is None


def test_error_reply_fails_the_request_with_its_status():
    scheduler = _scheduler()
    req = _registered(scheduler)
    _wait(scheduler, req, 0)
    reply = ImagesReadyReq(
        rid="r",
        ticket=3,
        materialization_id=0,
        image_indices=[0, 1, 2],
        error_msg="Could not decode image: image is corrupted or truncated",
        status_code=400,
    )

    scheduler.handle_images_ready(reply)

    demand = req.mm_demand
    assert demand.status == [ImageDemandStatus.FAILED] * 3
    assert (demand.error_msg, demand.status_code) == (reply.error_msg, 400)
    failures = []
    assert demand.blocks(0)
    scheduler._await_demand_images(req, failures)
    assert failures == [req]


def test_reply_that_disagrees_with_the_metadata_fails_and_is_released():
    scheduler = _scheduler()
    req = _registered(scheduler)
    _wait(scheduler, req, 0)
    reply = _reply(req, 0, [0, 1, 2])
    reply.mm_inputs.mm_items[1].offsets = [(11, 14)]
    sent = list(reply.mm_inputs.mm_items)

    scheduler.handle_images_ready(reply)

    assert all(item.feature is None for item in req.multimodal_inputs.mm_items)
    for item in sent:
        item.release_transport_proxies.assert_called_once_with()
    assert req.mm_demand.status_code == 500
    assert req.mm_demand.failed(0)


# --------------------------------------------------------------------------
# Tokenizer side
# --------------------------------------------------------------------------


def _demand(ticket=3):
    return DemandImageSources(
        ticket=ticket,
        sources=[b"x"],
        content_digests=["sha256:" + "00" * 32],
        artifacts=[None],
        offsets=[(3, 6)],
    )


def _tokenizer(materialize):
    manager = object.__new__(TokenizerManager)
    manager.rid_to_state = {}
    manager.mm_processor = SimpleNamespace(
        materialize_images_async=materialize,
        finalize_cache_identities=Mock(),
    )
    manager.cuda_vmm_feature_transport = SimpleNamespace(
        prepare_for_dispatch_async=AsyncMock(return_value=["lease"]),
        cancel_for_dispatch=Mock(),
    )
    manager._dispatch_to_scheduler = Mock()
    return manager


def _state(demand):
    state = ReqState(
        out_list=[], finished=False, event=asyncio.Event(), obj=Mock(), time_stats=None
    )
    state.mm_demand = demand
    return state


def _materialize_req(ticket=3):
    return MaterializeImagesReq(
        rid="r", ticket=ticket, materialization_id=0, image_indices=[0]
    )


def _run(manager, recv_req):
    with envs.SGLANG_MM_PRECOMPUTE_HASH.override(False):
        asyncio.run(manager._materialize_images(recv_req))
    manager._dispatch_to_scheduler.assert_called_once()
    reply = manager._dispatch_to_scheduler.call_args.args[0]
    assert isinstance(reply, ImagesReadyReq)
    assert (reply.rid, reply.ticket, reply.image_indices) == ("r", 3, [0])
    return reply


def test_tokenizer_replies_with_features():
    output = MultimodalProcessorOutput(mm_items=[])
    manager = _tokenizer(AsyncMock(return_value=output))
    manager.rid_to_state["r"] = _state(_demand())

    reply = _run(manager, _materialize_req())

    assert reply.error_msg is None and reply.mm_inputs is not None
    manager.mm_processor.finalize_cache_identities.assert_called_once_with(output)
    manager.cuda_vmm_feature_transport.cancel_for_dispatch.assert_not_called()


@pytest.mark.parametrize("gone", ["missing", "finished", "aborted", "new_ticket"])
def test_tokenizer_replies_400_for_a_request_that_is_gone(gone):
    materialize = AsyncMock()
    manager = _tokenizer(materialize)
    recv_req = _materialize_req()
    if gone != "missing":
        state = _state(_demand(ticket=4 if gone == "new_ticket" else 3))
        state.finished = gone == "finished"
        state.abort_sent = gone == "aborted"
        manager.rid_to_state["r"] = state

    reply = _run(manager, recv_req)

    materialize.assert_not_called()
    assert (reply.mm_inputs, reply.status_code) == (None, 400)


def test_tokenizer_cancels_leases_when_the_request_ends_during_preprocessing():
    manager = _tokenizer(None)
    state = _state(_demand())
    manager.rid_to_state["r"] = state

    async def materialize(demand, indices):
        state.abort_sent = True
        return MultimodalProcessorOutput(mm_items=[])

    manager.mm_processor.materialize_images_async = materialize

    reply = _run(manager, _materialize_req())

    assert (reply.mm_inputs, reply.status_code) == (None, 400)
    manager.cuda_vmm_feature_transport.cancel_for_dispatch.assert_called_once_with(
        ["lease"]
    )


@pytest.mark.parametrize(
    "error, status, message",
    [
        (
            ValueError("Could not decode image: image is corrupted or truncated"),
            400,
            "Could not decode image: image is corrupted or truncated",
        ),
        (RuntimeError("boom"), 500, "Image preprocessing failed: RuntimeError: boom"),
    ],
)
def test_tokenizer_maps_preprocessing_errors(error, status, message):
    manager = _tokenizer(AsyncMock(side_effect=error))
    manager.rid_to_state["r"] = _state(_demand())

    reply = _run(manager, _materialize_req())

    assert (reply.mm_inputs, reply.status_code, reply.error_msg) == (
        None,
        status,
        message,
    )


def test_tokenizer_cancels_leases_when_dispatch_fails():
    manager = _tokenizer(AsyncMock(return_value=MultimodalProcessorOutput(mm_items=[])))
    manager.rid_to_state["r"] = _state(_demand())
    manager._dispatch_to_scheduler.side_effect = RuntimeError("socket closed")

    with pytest.raises(RuntimeError, match="socket closed"):
        with envs.SGLANG_MM_PRECOMPUTE_HASH.override(False):
            asyncio.run(manager._materialize_images(_materialize_req()))

    manager.cuda_vmm_feature_transport.cancel_for_dispatch.assert_called_once_with(
        ["lease"]
    )


def test_register_attaches_the_sources_to_the_request_state():
    manager = _tokenizer(None)
    manager.rid_to_state["r"] = _state(None)
    tokenized = SimpleNamespace(rid="r", mm_demand_ticket=None)
    demand = _demand()

    manager._register_mm_demand(tokenized, demand)

    assert tokenized.mm_demand_ticket == 3
    assert manager.rid_to_state["r"].mm_demand is demand


def test_switch_off_keeps_every_request_on_the_full_path():
    manager = _tokenizer(None)
    manager._mm_demand_supported = None
    request = GenerateReqInput(text="hi", image_data=["image.png"], rid="r")
    manager.rid_to_state["r"] = _state(None)

    with (
        envs.SGLANG_K3_IMAGE_DEMAND_PREPROCESS.override(False),
        patch("sglang.srt.managers.tokenizer_manager.logger") as logger,
    ):
        assert not manager._mm_demand_eligible(request)

    assert manager._mm_demand_supported is False
    logger.warning.assert_not_called()
    # Without a ticket the scheduler never builds demand state for a request.
    defaults = {
        field.name: field.default
        for field in msgspec.structs.fields(TokenizedGenerateReqInput)
    }
    assert defaults["mm_demand_ticket"] is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"sampling_params": {"n": 2}},
        {"sampling_params": {"beam_width": 4}},
        {"session_params": {"id": "s"}},
        {"video_data": ["clip.mp4"]},
    ],
)
def test_requests_that_need_features_up_front_are_not_eligible(overrides):
    manager = _tokenizer(None)
    manager._mm_demand_supported = True
    manager.rid_to_state["r"] = _state(None)
    request = GenerateReqInput(
        text="hi", image_data=["image.png"], rid="r", **overrides
    )

    assert not manager._mm_demand_eligible(request)
