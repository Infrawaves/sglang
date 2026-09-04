# SPDX-License-Identifier: Apache-2.0
"""NVLink fabric (MNNVL) identity.

An NVL72 rack shares one NVLink domain across nodes, but partitions it into
cliques at runtime. A fabric handle only resolves between two GPUs that agree on
both the cluster UUID and the clique id.
"""

from __future__ import annotations

import logging
from typing import Optional

import msgspec

logger = logging.getLogger(__name__)

# nvmlGpuFabricState_t: NVML_GPU_FABRIC_STATE_COMPLETED.
_FABRIC_STATE_COMPLETED = 3


class NvlinkFabricIdentity(msgspec.Struct, frozen=True):
    cluster_uuid: str
    clique_id: int

    def __str__(self) -> str:
        return f"{self.cluster_uuid}/clique{self.clique_id}"


def get_nvlink_fabric_identity(gpu_id: int) -> Optional[NvlinkFabricIdentity]:
    handle = _open_nvml_handle(gpu_id)
    if handle is None:
        return None

    import pynvml

    info = _query_fabric_info(pynvml, handle, gpu_id)
    if info is None:
        return None

    state, status = info.state, info.status
    if state != _FABRIC_STATE_COMPLETED or status != 0:
        logger.info(
            "GPU %s NVLink fabric is not usable (state=%r, status=%r).",
            gpu_id,
            state,
            status,
        )
        return None

    cluster_uuid = _format_cluster_uuid(info.clusterUuid)
    clique_id = int(info.cliqueId)
    # A standalone node answers COMPLETED/Success with a zeroed UUID and clique
    # 0. Taken as an identity it would match every other such node, so two
    # unrelated hosts would agree they share a clique.
    if clique_id == 0 and not cluster_uuid.strip("0"):
        logger.info(
            "GPU %s reports no NVLink fabric (zeroed cluster UUID); this node is "
            "not part of an MNNVL domain.",
            gpu_id,
        )
        return None

    return NvlinkFabricIdentity(cluster_uuid=cluster_uuid, clique_id=clique_id)


def _query_fabric_info(pynvml, handle, gpu_id: int):
    # The versioned call takes the struct as an out-param and reads its version
    # field to pick a layout, so that has to be set first; older pynvml wraps the
    # unversioned one, which returns the struct instead.
    get_info_v = getattr(pynvml, "nvmlDeviceGetGpuFabricInfoV", None)
    struct_type = getattr(pynvml, "c_nvmlGpuFabricInfoV_t", None)
    if get_info_v is not None and struct_type is not None:
        version = getattr(pynvml, "nvmlGpuFabricInfo_v3", None) or getattr(
            pynvml, "nvmlGpuFabricInfo_v2", None
        )
        try:
            info = struct_type()
            if version is not None:
                info.version = version
            get_info_v(handle, pynvml.byref(info))
            return info
        except Exception as e:
            logger.debug(
                "nvmlDeviceGetGpuFabricInfoV failed on GPU %s (%s).", gpu_id, e
            )

    get_info = getattr(pynvml, "nvmlDeviceGetGpuFabricInfo", None)
    if get_info is None:
        logger.debug("This pynvml exposes no GPU fabric info entry point.")
        return None
    try:
        return get_info(handle)
    except Exception as e:
        # Raises on GPUs not attached to a fabric at all, so not worth a warning.
        logger.debug("No NVLink fabric info for GPU %s (%s).", gpu_id, e)
        return None


def _open_nvml_handle(gpu_id: int):
    try:
        import pynvml

        from sglang.srt.utils.numa_utils import _get_nvml_device_index

        pynvml.nvmlInit()
        # gpu_id is a CUDA logical index; NVML indexes physical GPUs, and the
        # two differ under a reordered CUDA_VISIBLE_DEVICES.
        return pynvml.nvmlDeviceGetHandleByIndex(_get_nvml_device_index(gpu_id))
    except ModuleNotFoundError:
        logger.debug("pynvml not installed; cannot probe the NVLink fabric.")
        return None
    except Exception as e:
        logger.debug("Cannot open GPU %s through NVML (%s).", gpu_id, e)
        return None


def _format_cluster_uuid(raw) -> str:
    # pynvml hands back bytes or an already-formatted str depending on version.
    # Both sides of a comparison come through here, so only determinism matters.
    if isinstance(raw, str):
        return raw
    try:
        return bytes(raw).hex()
    except (TypeError, ValueError):
        return str(raw)
