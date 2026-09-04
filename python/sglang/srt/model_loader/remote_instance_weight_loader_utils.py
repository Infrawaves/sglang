# SPDX-License-Identifier: Apache-2.0

import enum
import importlib
import importlib.util
import logging
import time
from typing import List, Optional

import msgspec
import requests

from sglang.srt.environ import envs
from sglang.srt.utils.network import NetworkAddress
from sglang.srt.utils.nvlink_fabric_utils import NvlinkFabricIdentity

logger = logging.getLogger(__name__)

# Mooncake transport names. "nvlink" is MNNVL, reaching one fabric clique;
# "nvlink_intra" is CUDA IPC, reaching one host.
MNNVL_PROTOCOL = "nvlink"
INTRA_NVLINK_PROTOCOL = "nvlink_intra"
# Fastest first, which is also fewest peers first.
NVLINK_PROTOCOLS = (INTRA_NVLINK_PROTOCOL, MNNVL_PROTOCOL)

# Arbitrary; enough for the ranks unregistering concurrently to drain.
_DEREGISTER_MAX_ATTEMPTS = 5
_DEREGISTER_BACKOFF_S = 0.5
# mooncake's ERR_ADDRESS_NOT_REGISTERED. Its batch unregister returns the first
# error even after freeing most of the list, and the addresses it did free answer
# a retry with this -- they are done, not stuck.
_MOONCAKE_ERR_ADDRESS_NOT_REGISTERED = -3


def _iter_manifest_parameters(model):
    """Every name a parameter answers to, not just the canonical one.

    post_load_weights aliases parameters onto their parent
    (`self._bfa_f_b_w = self.f_b_proj.weight` in kimi_k3), and
    named_parameters() de-duplicates by identity keeping the parent's name, so
    the child's vanishes. The client is a fresh skeleton that has run neither
    the aliasing nor post_load_weights, and looks the child name up.
    """
    return model.named_parameters(remove_duplicate=False)


class RemoteInstanceWeightLoaderBackend(str, enum.Enum):
    NCCL = "nccl"
    TRANSFER_ENGINE = "transfer_engine"
    MODELEXPRESS = "modelexpress"


def trigger_init_weights_send_group_for_remote_instance_request(
    remote_instance_weight_loader_seed_instance_ip: str,
    remote_instance_weight_loader_seed_instance_service_port: int,
    remote_instance_weight_loader_send_weights_group_ports: List[int],
    remote_instance_weight_loader_client_id: str,
):
    seed_instance_service_url = f"http://{remote_instance_weight_loader_seed_instance_ip}:{remote_instance_weight_loader_seed_instance_service_port}"
    # Only support loading weights from instance with same parallelism strategy.
    # Per TP rank pair between seed and dst instances will build a communication group for sending weights.
    # i.e. seed TP 0 <-> dst TP 0, seed TP 1 <-> dst TP 1, etc.
    # Each communication group will have a world size 2.
    try:
        requests.post(
            f"{seed_instance_service_url}/init_weights_send_group_for_remote_instance",
            json={
                "master_address": remote_instance_weight_loader_seed_instance_ip,
                "ports": (
                    ",".join(
                        str(p)
                        for p in remote_instance_weight_loader_send_weights_group_ports
                    )
                ),
                "group_rank": 0,
                "world_size": 2,
                "group_name": f"send_weights_{remote_instance_weight_loader_client_id}",
                "backend": "nccl",
            },
        )
    except Exception as e:
        logger.error(
            f"Failed to trigger init_weights_send_group_for_remote_instance_request to seed instance {seed_instance_service_url}: {e}."
        )
        raise


def trigger_transferring_weights_request(
    remote_instance_weight_loader_seed_instance_ip: str,
    remote_instance_weight_loader_seed_instance_service_port: int,
    remote_instance_weight_loader_send_weights_group_ports: List[int],
    remote_instance_weight_loader_client_id: str,
):
    seed_instance_service_url = f"http://{remote_instance_weight_loader_seed_instance_ip}:{remote_instance_weight_loader_seed_instance_service_port}"
    try:
        requests.post(
            f"{seed_instance_service_url}/send_weights_to_remote_instance",
            json={
                "master_address": remote_instance_weight_loader_seed_instance_ip,
                "ports": (
                    ",".join(
                        str(p)
                        for p in remote_instance_weight_loader_send_weights_group_ports
                    )
                ),
                "group_name": f"send_weights_{remote_instance_weight_loader_client_id}",
            },
        )
    except Exception as e:
        logger.error(f"Failed to trigger send weights to remote instance request: {e}")
        raise


class SeedTransferEngineInfo(msgspec.Struct, frozen=True):
    # protocol and fabric_identity are None when the seed predates them; the
    # client then assumes the seed matches its own configured transport.
    session_id: str
    weights_info_dict: dict
    protocol: Optional[str] = None
    fabric_identity: Optional[NvlinkFabricIdentity] = None
    # Other endpoints for the same weights. The primary above stays the NIC one,
    # which is where a client too old to read this field lands.
    alternates: tuple["SeedTransferEngineInfo", ...] = ()

    def endpoint_for(
        self,
        *,
        protocol: str,
        fabric: Optional[NvlinkFabricIdentity] = None,
        local_host: Optional[str] = None,
    ) -> Optional["SeedTransferEngineInfo"]:
        # Each NVLink transport is gated on a different thing: CUDA IPC on the
        # peer being on this host, a fabric handle on it being in this clique.
        # The NIC routes anywhere, so it is gated on neither.
        for endpoint in (self, *self.alternates):
            if endpoint.protocol != protocol:
                continue
            if protocol == MNNVL_PROTOCOL and endpoint.fabric_identity != fabric:
                continue
            if protocol == INTRA_NVLINK_PROTOCOL and not endpoint.is_on_host(
                local_host
            ):
                continue
            return endpoint
        return None

    def is_on_host(self, host: Optional[str]) -> bool:
        # The session id is the seed's own host:port, so no second round trip.
        if host is None:
            return False
        try:
            return NetworkAddress.parse(self.session_id).host == host
        except ValueError:
            logger.warning(
                "Cannot read a host out of seed session id %r.", self.session_id
            )
            return False

    def offered_protocols(self) -> tuple[str, ...]:
        return tuple(
            e.protocol for e in (self, *self.alternates) if e.protocol is not None
        )


def get_remote_instance_transfer_engine_info_per_rank(
    seed_url: str, rank: int
) -> Optional[SeedTransferEngineInfo]:
    try:
        response = requests.get(
            f"{seed_url}/get_remote_instance_transfer_engine_info",
            params={
                "rank": rank,
            },
        )

        if response.status_code != 200:
            logger.error(f"request.get failed: {response.status_code}")
            return None

        data = response.json()
        if "remote_instance_transfer_engine_info" not in data:
            logger.error(
                "Failed to get `remote_instance_transfer_engine_info` in response."
            )
            return None

        return _parse_seed_transfer_engine_info(
            data["remote_instance_transfer_engine_info"]
        )
    except Exception as e:
        logger.error(f"Exception: {e}")
        return None


def _parse_seed_transfer_engine_info(raw) -> Optional[SeedTransferEngineInfo]:
    # A positional list, only ever appended to, so a client can be newer than
    # the seed it reads.
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        logger.error("Malformed transfer engine info from the seed: %r", raw)
        return None
    session_id, weights_info_dict = raw[0], raw[1]
    if session_id is None or weights_info_dict is None:
        return None
    protocol = raw[2] if len(raw) > 2 else None
    fabric = raw[3] if len(raw) > 3 else None
    alternates = raw[4] if len(raw) > 4 else None
    return SeedTransferEngineInfo(
        session_id=session_id,
        weights_info_dict=weights_info_dict,
        protocol=protocol,
        fabric_identity=(
            msgspec.convert(fabric, NvlinkFabricIdentity) if fabric else None
        ),
        alternates=_parse_alternates(alternates),
    )


def _parse_alternates(raw) -> tuple[SeedTransferEngineInfo, ...]:
    if not isinstance(raw, (list, tuple)):
        return ()
    parsed = []
    for entry in raw:
        # Leaves only: nesting would let a malformed payload recurse.
        endpoint = _parse_seed_transfer_engine_info(entry[:4] if entry else entry)
        if endpoint is not None:
            parsed.append(endpoint)
    return tuple(parsed)


def _registration_frees_no_memory(protocol: Optional[str]) -> bool:
    # ibv_reg_mr pins pages; an NVLink registration is a metadata entry for
    # memory the model already owns.
    if protocol is None:
        protocol = (
            envs.SGLANG_REMOTE_INSTANCE_PROTOCOL.get() or envs.MOONCAKE_PROTOCOL.get()
        )
    return protocol in NVLINK_PROTOCOLS


def _registration_failure_hint() -> str:
    # The mooncake-side cause only reaches its stderr glog.
    protocol = (
        envs.SGLANG_REMOTE_INSTANCE_PROTOCOL.get() or envs.MOONCAKE_PROTOCOL.get()
    )
    if protocol not in NVLINK_PROTOCOLS:
        return ""
    return (
        f" The transport is mooncake protocol={protocol!r}, whose registration "
        "goes through the CUDA driver API and needs a current context on the "
        "calling thread; a mooncake log line reading "
        "'cuMemGetAddressRange failed ... (error 201)' means it had none."
    )


def register_memory_region(model, transfer_engine):
    if importlib.util.find_spec("torch") is None:
        return register_memory_region_v1(model, transfer_engine)
    else:
        return register_memory_region_v2(model, transfer_engine)


def deregister_memory_region(
    transfer_engine, registered_blocks, protocol: Optional[str] = None
) -> bool:
    """Release the reader's registration of the given blocks.

    Reader side only: the seed keeps its registration, that is what it serves
    handles from. ``protocol`` is the transport the engine was built with; it
    decides whether releasing reclaims anything at all.
    """
    if not registered_blocks:
        return True
    addresses = [address for address, _ in registered_blocks]
    try:
        if transfer_engine.batch_unregister_memory(addresses) == 0:
            return True
    except Exception as e:
        # EAGAIN when the batch call fans out over too many blocks at once, and
        # AttributeError on mooncake builds without the batch entry point.
        logger.debug("Batch deregistration raised (%s).", e)

    if _registration_frees_no_memory(protocol):
        # NVLink tracks a registration by cudaMalloc segment base while these
        # blocks are per-tensor, so the retry below would fail for seconds.
        logger.debug(
            "Leaving %d NVLink weight registrations in place; they hold no "
            "memory to reclaim.",
            len(addresses),
        )
        return True

    # A non-zero batch return is the normal path, not a failure: mooncake reports
    # its first error even after freeing most of the list, so this pass sweeps
    # whatever it left. Retrying past that pays off only while it makes progress
    # -- a driver busy with the other ranks clears, a permanent refusal does not.
    logger.debug(
        "Batch deregistration reported an error; sweeping the %d region(s).",
        len(addresses),
    )
    pending = addresses
    for attempt in range(_DEREGISTER_MAX_ATTEMPTS):
        if attempt:
            time.sleep(_DEREGISTER_BACKOFF_S * attempt)
        failed = []
        for address in pending:
            try:
                ret = transfer_engine.unregister_memory(address)
            except Exception:
                failed.append(address)
                continue
            if ret != 0 and ret != _MOONCAKE_ERR_ADDRESS_NOT_REGISTERED:
                failed.append(address)
        if not failed:
            return True
        stalled = len(failed) == len(pending)
        logger.warning(
            "%d of %d weight memory regions still pinned after attempt %d.",
            len(failed),
            len(addresses),
            attempt + 1,
        )
        pending = failed
        if stalled:
            logger.warning(
                "Attempt %d freed none of them, so the remaining addresses are "
                "being refused rather than deferred; not retrying.",
                attempt + 1,
            )
            break

    logger.error(
        "Failed to deregister %d of %d weight memory regions; that memory "
        "stays pinned for the life of the process, and the KV pool sizes "
        "itself around it.",
        len(pending),
        len(addresses),
    )
    return False


def register_memory_region_v1(model, transfer_engine):
    start_tic = time.time()

    weight_mr_dict = {}
    registered_blocks = []
    seen_blocks = set()
    try:
        for name, weight in _iter_manifest_parameters(model):
            size = weight.numel() * weight.element_size()
            block = (weight.data_ptr(), size)
            # One registration per byte range; aliases would otherwise register
            # twice, and the second unregister_memory would fail.
            if block not in seen_blocks:
                ret = transfer_engine.register_memory(weight.data_ptr(), size)
                if ret != 0:
                    raise RuntimeError(
                        f"register memory failed for weight {name}, error: {ret}."
                        f"{_registration_failure_hint()}"
                    )
                seen_blocks.add(block)
                registered_blocks.append(block)
            weight_mr_dict[name] = (
                weight.data_ptr(),
                weight.numel(),
                weight.element_size(),
            )
    except BaseException:
        # The caller only learns about blocks we return, so a partial
        # registration would stay pinned with nobody holding a handle.
        deregister_memory_region(transfer_engine, registered_blocks)
        raise

    end_tic = time.time()
    logger.debug(f"Register memory region time: {(end_tic - start_tic):.4f}s")
    return weight_mr_dict, registered_blocks


def register_memory_region_v2(model, transfer_engine):
    start_tic = time.time()

    weight_mr_dict = {}
    weight_addr_set = set()
    for name, weight in _iter_manifest_parameters(model):
        weight_mr_dict[name] = (
            weight.data_ptr(),
            weight.numel(),
            weight.element_size(),
        )
        weight_addr_set.add(weight.data_ptr())

    import torch

    memory_snapshot = torch.cuda.memory.memory_snapshot()
    weight_blocks_for_reg_mr = []
    # Blocks in each segment have continuous physical addresses,
    # so they can be merged for memory registration.
    for segment in memory_snapshot:
        current_weight_block = None
        blocks = segment.get("blocks", [])
        for block in blocks:
            address = block.get("address", -1)
            size = block.get("size", -1)
            state = block.get("state", "")
            if address < 0 or size < 0 or state == "":
                continue
            # Only register active allocated memory blocks that hold weights.
            if state == "active_allocated":
                if address in weight_addr_set:
                    if current_weight_block is None:
                        current_weight_block = (address, size)
                    elif current_weight_block[0] + current_weight_block[1] == address:
                        current_weight_block = (
                            current_weight_block[0],
                            current_weight_block[1] + size,
                        )
                    else:
                        weight_blocks_for_reg_mr.append(current_weight_block)
                        current_weight_block = (address, size)
        if current_weight_block is not None:
            weight_blocks_for_reg_mr.append(current_weight_block)

    # Register merged memory blocks that hold weights.
    registered_blocks = []
    try:
        for weight_block in weight_blocks_for_reg_mr:
            address, size = weight_block
            ret = transfer_engine.register_memory(address, size)
            if ret != 0:
                raise RuntimeError(
                    f"register memory failed for weight block at address "
                    f"{address} with size {size}, error: {ret}."
                    f"{_registration_failure_hint()}"
                )
            registered_blocks.append(weight_block)
    except BaseException:
        # The caller only learns about blocks we return, so a partial
        # registration would stay pinned with nobody holding a handle.
        deregister_memory_region(transfer_engine, registered_blocks)
        raise

    end_tic = time.time()
    logger.debug(f"Register memory region v2 time: {(end_tic - start_tic):.4f}s")
    return weight_mr_dict, registered_blocks
