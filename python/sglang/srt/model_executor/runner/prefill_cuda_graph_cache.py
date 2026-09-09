# Copyright 2023-2026 SGLang Team
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
"""Persistent compatibility metadata for prefill CUDA-graph capture.

CUDA does not currently expose a supported API that serializes a
``cudaGraphExec_t`` (and the breakable backend also owns Python eager-break
closures).  This module therefore stores *only* a strict, rank-scoped capture
manifest.  A completed manifest is a warmup hint for a later process; it is
never treated as an executable graph or as permission to reuse a GPU address.

The manifest is deliberately conservative:

* every shape, backend, model/runtime and parallelism fact participates in the
  fingerprint;
* all ranks in a tensor-parallel group must agree on the hit/miss verdict;
* writes are atomic and incomplete captures are never accepted; and
* a successful two-warmup capture only arms a one-warmup probe. The faster hint
  is marked validated after that probe succeeds, while a failed probe is
  permanently disabled for the exact fingerprint.

The schema has fields for future stable-VA graph relocation work, but this
version's artifact kind is explicitly ``metadata_only``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import logging
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

PREFILL_CUDA_GRAPH_CACHE_SCHEMA_VERSION = 1
PREFILL_CUDA_GRAPH_CACHE_CODE_VERSION = "warmup-hint-v3"
PREFILL_CUDA_GRAPH_CACHE_ARTIFACT_KIND = "metadata_only"
DEFAULT_PREFILL_CUDA_GRAPH_WARMUP_ITERATIONS = 2
CACHED_PREFILL_CUDA_GRAPH_WARMUP_ITERATIONS = 1
_VALID_PREFILL_CUDA_GRAPH_WARMUP_ITERATIONS = frozenset((1, 2))
_PREFILL_CUDA_GRAPH_PROBE_PENDING = "pending"
_PREFILL_CUDA_GRAPH_PROBE_VALIDATED = "validated"
_PREFILL_CUDA_GRAPH_PROBE_DISABLED = "disabled"
_VALID_PREFILL_CUDA_GRAPH_PROBE_STATES = frozenset(
    (
        _PREFILL_CUDA_GRAPH_PROBE_PENDING,
        _PREFILL_CUDA_GRAPH_PROBE_VALIDATED,
        _PREFILL_CUDA_GRAPH_PROBE_DISABLED,
    )
)
# Public names keep the on-disk state machine easy to inspect in diagnostics
# and unit tests without exposing any executable CUDA-graph handles.
PREFILL_CUDA_GRAPH_PROBE_PENDING = _PREFILL_CUDA_GRAPH_PROBE_PENDING
PREFILL_CUDA_GRAPH_PROBE_VALIDATED = _PREFILL_CUDA_GRAPH_PROBE_VALIDATED
PREFILL_CUDA_GRAPH_PROBE_DISABLED = _PREFILL_CUDA_GRAPH_PROBE_DISABLED

_CACHE_SUBDIR = Path("cuda_graph") / "prefill"
_SHAPE_FIELDS = ("size", "stream_idx", "variant_label", "dsa_variant")
_PARALLEL_FIELDS = (
    "world_size",
    "world_rank",
    "tp_size",
    "tp_rank",
    "pp_size",
    "pp_rank",
    "dp_size",
    "dp_rank",
    "attn_tp_size",
    "attn_tp_rank",
    "attn_cp_size",
    "attn_cp_rank",
    "attn_dcp_size",
    "attn_dcp_rank",
    "attn_dp_size",
    "attn_dp_rank",
    "moe_ep_size",
    "moe_ep_rank",
    "moe_tp_size",
    "moe_tp_rank",
    "moe_dp_size",
    "moe_dp_rank",
    "dcp_enabled",
    "gpu_id",
)
_PARTICIPANT_GROUP_FIELDS = (
    "tp_group",
    "pp_group",
    "dcp_group",
    "attn_tp_group",
    "moe_ep_group",
    "moe_tp_group",
)
_MODEL_FIELDS = (
    "model_path",
    "revision",
    "model_impl",
    "quantization",
    "dtype",
    "context_len",
    "hidden_size",
    "hc_hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "vocab_size",
    "is_multimodal",
)
_HF_CONFIG_FIELDS = (
    "model_type",
    "architectures",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "vocab_size",
    "torch_dtype",
)
_RUNNER_FIELDS = (
    "prefill_backend_name",
    "capture_num_tokens",
    "max_num_tokens",
    "max_bs",
    "capture_forward_mode",
    "capture_hidden_mode",
    "capture_return_pooled_hidden_states",
    "mamba_track_enabled",
    "enable_lora",
    "_capture_lora",
    "_capture_req_slots",
    "_is_full_backend",
    "enable_cp_v2_bcg_capture",
    "_capture_chunked_prefix",
    "_prefix_chunk_len",
    "_prefix_chunk_capacity",
    "_prefix_capture_variants",
)
_SERVER_ARG_FIELDS = (
    "model_path",
    "revision",
    "model_impl",
    "model_checksum",
    "weight_version",
    "attention_backend",
    "moe_runner_backend",
    "dtype",
    "quantization",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "data_parallel_size",
    "expert_parallel_size",
    "context_length",
    "max_running_requests",
    "chunked_prefill_size",
    "disable_chunked_prefix_cache",
    "enable_lora",
    "enable_mamba",
    "enable_mamba_extra_buffer",
    "enable_dp_attention",
    "attention_tp_size",
    "attention_cp_size",
    "decode_context_parallel_size",
    "enable_memory_saver",
    "enable_cudagraph_gc",
    "debug_cuda_graph",
    "enable_profile_cuda_graph",
    "enable_torch_compile",
    "speculative_algorithm",
)
_ENV_MARKERS = (
    "CUDA_GRAPH",
    "GRAPH_",
    "FLASHINFER",
    "DEEPEP",
    "MOE",
    "MAMBA",
    "DCP",
    "PREFILL_CP",
    "LORA",
    "TRTLLM",
    "NVLINK",
    "VMM",
)


def _qualified_name(value: Any) -> str | None:
    if value is None:
        return None
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    """Convert common runtime/config values into deterministic JSON data.

    Fingerprinting must not include object addresses (``repr`` of many torch
    objects does), so unknown objects are represented by their qualified type
    and a bounded set of public scalar fields.
    """

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if depth >= 5:
        return _qualified_name(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"length": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Enum):
        return _json_safe(value.value, depth=depth + 1)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return _json_safe(dataclasses.asdict(value), depth=depth + 1)
        except (TypeError, ValueError, RecursionError):
            return _qualified_name(value)
    if isinstance(value, Mapping):
        items = []
        for key in sorted(value, key=lambda item: str(item)):
            if len(items) >= 128:
                break
            items.append(
                (
                    str(key),
                    _json_safe(value[key], depth=depth + 1),
                )
            )
        return {key: item for key, item in items}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, depth=depth + 1) for item in value[:256]]
    if isinstance(value, (set, frozenset)):
        safe = [_json_safe(item, depth=depth + 1) for item in value]
        return sorted(safe, key=lambda item: str(item))[:256]

    # torch.dtype and similar scalar-like objects have stable string forms.
    module_name = type(value).__module__
    if module_name.startswith("torch"):
        text = str(value)
        if "0x" not in text:
            return text

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _json_safe(to_dict(), depth=depth + 1)
        except (AttributeError, TypeError, ValueError, RecursionError) as exc:
            logger.debug(
                "Unable to convert %s to a dict for cache fingerprint: %s",
                type(value),
                exc,
            )

    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        result = {"__class__": _qualified_name(value)}
        for key in sorted(attrs):
            if key.startswith("_") or len(result) >= 65:
                continue
            item = attrs[key]
            if callable(item):
                continue
            safe_item = _json_safe(item, depth=depth + 1)
            if isinstance(safe_item, (type(None), bool, int, float, str, list, dict)):
                result[key] = safe_item
        return result
    return _qualified_name(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def fingerprint_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def shape_key_to_dict(shape_key: Any) -> dict[str, Any]:
    """Return the stable part of a ``ShapeKey`` without importing its module."""

    if isinstance(shape_key, Mapping):
        source = shape_key
        size = source.get("size")
        result = {field: source.get(field) for field in _SHAPE_FIELDS}
    else:
        size = getattr(shape_key, "size", None)
        result = {field: getattr(shape_key, field, None) for field in _SHAPE_FIELDS}
    if size is None:
        raise ValueError(f"shape key has no size: {shape_key!r}")
    result["size"] = int(size)
    return _json_safe(result)


def _normalized_shapes(shapes: Sequence[Any]) -> list[dict[str, Any]]:
    unique = {
        _canonical_json(shape_key_to_dict(shape)): shape_key_to_dict(shape)
        for shape in shapes
    }
    return [unique[key] for key in sorted(unique)]


def shape_key_id(shape_key: Any) -> str:
    return _canonical_json(shape_key_to_dict(shape_key))


def _is_valid_warmup_iterations(value: Any) -> bool:
    """Return whether ``value`` is an actual schema-supported integer.

    ``bool`` is an ``int`` subclass, so an explicit type check is intentional:
    accepting ``True`` as one warmup would make a hand-edited manifest alter
    capture behavior unexpectedly.
    """

    return type(value) is int and value in _VALID_PREFILL_CUDA_GRAPH_WARMUP_ITERATIONS


def _attrs(obj: Any, names: Sequence[str]) -> dict[str, Any]:
    if obj is None:
        return {}
    result = {}
    for name in names:
        try:
            value = getattr(obj, name)
        except Exception as exc:  # noqa: BLE001 - runtime properties are user-defined
            logger.debug(
                "Unable to read %s.%s for cache fingerprint: %s", type(obj), name, exc
            )
            continue
        if callable(value):
            continue
        result[name] = _json_safe(value)
    return result


def _object_signature(obj: Any, names: Sequence[str]) -> Any:
    if obj is None:
        return None
    return {"class": _qualified_name(obj), "fields": _attrs(obj, names)}


def _parallel_values() -> dict[str, Any]:
    try:
        parallel = get_parallel()
    except (AssertionError, AttributeError, RuntimeError):
        return {}
    result = _attrs(parallel, _PARALLEL_FIELDS)
    topology = {}
    for name in _PARTICIPANT_GROUP_FIELDS:
        try:
            group = getattr(parallel, name, None)
            ranks = getattr(group, "ranks", None)
            world_size = getattr(group, "world_size", None)
        except (AssertionError, AttributeError, RuntimeError):
            continue
        if ranks is None:
            continue
        try:
            topology[name] = {
                "ranks": [int(rank) for rank in ranks],
                "world_size": int(world_size),
            }
        except (TypeError, ValueError):
            continue
    if topology:
        result["participant_group_topology"] = topology
    return result


def _rank_for_runner(runner: Any) -> int:
    values = _parallel_values()
    rank = values.get("world_rank")
    if rank is None:
        try:
            import torch

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
        except (AssertionError, AttributeError, RuntimeError, TypeError, ValueError):
            rank = None
    if rank is None:
        rank = getattr(getattr(runner, "model_runner", None), "rank", 0)
    try:
        return int(rank)
    except (TypeError, ValueError):
        return 0


def _device_fingerprint(runner: Any) -> dict[str, Any]:
    model_runner = getattr(runner, "model_runner", None)
    device = getattr(runner, "device", getattr(model_runner, "device", "unknown"))
    gpu_id = getattr(model_runner, "gpu_id", 0)
    result: dict[str, Any] = {"device": str(device), "gpu_id": _json_safe(gpu_id)}
    if str(device).split(":", 1)[0].lower() != "cuda":
        return result
    try:
        import torch

        result["torch_cuda"] = torch.version.cuda
        result["torch_hip"] = torch.version.hip
        driver_version_fn = getattr(torch._C, "_cuda_getDriverVersion", None)
        if callable(driver_version_fn):
            driver_version = int(driver_version_fn())
            if driver_version > 0:
                result["cuda_driver"] = driver_version
        if not torch.cuda.is_available():
            result["availability"] = False
            return result
        properties = torch.cuda.get_device_properties(gpu_id)
        result.update(
            {
                "name": getattr(properties, "name", None),
                "uuid": str(getattr(properties, "uuid", "unknown")),
                "capability": list(torch.cuda.get_device_capability(gpu_id)),
                "total_memory": int(getattr(properties, "total_memory", 0)),
            }
        )
    except (
        AssertionError,
        AttributeError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        # A cache miss is safer than making startup depend on optional driver
        # probing (and this path is also exercised by CPU-only unit tests).
        result["availability"] = "unknown"
    return _json_safe(result)


def _package_versions() -> dict[str, Any]:
    versions = {}
    for package in (
        "sglang",
        "torch",
        "flashinfer-python",
        "sglang-kernel",
        "cuda-python",
        "cuda-bindings",
        "triton",
        "nvidia-cutlass-dsl",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            versions[package] = "unknown"
    try:
        import torch

        versions["torch_runtime"] = torch.__version__
        versions["cuda_runtime"] = torch.version.cuda
        versions["hip_runtime"] = torch.version.hip
    except (AttributeError, ImportError, RuntimeError):
        logger.debug("Unable to read torch runtime versions for cache fingerprint")
    return _json_safe(versions)


def _implementation_digest() -> str | None:
    """Invalidate metadata when this cache implementation changes in place."""

    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return None


def _source_digest(value: Any) -> str | None:
    """Return a source-file digest for a runner/backend implementation."""

    try:
        source_path = inspect.getsourcefile(value)
        if source_path is None:
            return None
        return hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    except (OSError, TypeError):
        return None


def _relevant_environment() -> dict[str, str]:
    result = {}
    for name, value in os.environ.items():
        if name == "SGLANG_CACHE_DIR" or not name.startswith("SGLANG_"):
            continue
        if any(marker in name for marker in _ENV_MARKERS):
            result[name] = value
    return dict(sorted(result.items()))


def build_prefill_cuda_graph_fingerprint(
    runner: Any, expected_shapes: Sequence[Any]
) -> dict[str, Any]:
    """Build the compatibility payload used by ``PrefillCudaGraphCache``."""

    model_runner = getattr(runner, "model_runner", None)
    model_config = getattr(model_runner, "model_config", None)
    model = getattr(model_runner, "model", None)
    hf_config = getattr(model_config, "hf_config", None)
    server_args = getattr(model_runner, "server_args", None)
    parallel = _parallel_values()

    graph_backend = getattr(runner, "prefill_backend_name", None)
    fingerprint = {
        "schema_version": PREFILL_CUDA_GRAPH_CACHE_SCHEMA_VERSION,
        "code_version": PREFILL_CUDA_GRAPH_CACHE_CODE_VERSION,
        "backend": _json_safe(graph_backend),
        "expected_shapes": _normalized_shapes(expected_shapes),
        "model": {
            "config": _attrs(model_config, _MODEL_FIELDS),
            "hf_config": _attrs(hf_config, _HF_CONFIG_FIELDS),
            "model_class": _qualified_name(model),
            "quant_config": _object_signature(
                getattr(model, "quant_config", None),
                (
                    "name",
                    "method",
                    "quantization_method",
                    "weight_block_size",
                    "group_size",
                    "is_checkpoint_serialized",
                    "is_checkpoint_nvfp4_serialized",
                    "is_checkpoint_fp8_serialized",
                ),
            ),
        },
        "parallel": parallel,
        "device": _device_fingerprint(runner),
        "runtime": {
            "packages": _package_versions(),
            "implementation_digest": _implementation_digest(),
            "runner_source_digest": _source_digest(type(runner)),
            "backend_source_digest": _source_digest(
                type(getattr(runner, "backend", None))
            ),
            "attention_backend": _qualified_name(
                getattr(model_runner, "attn_backend", None)
            ),
            "moe_backend": _qualified_name(getattr(model_runner, "moe_runner", None)),
        },
        "runner": _attrs(runner, _RUNNER_FIELDS),
        "server_args": _attrs(server_args, _SERVER_ARG_FIELDS),
        "environment": _relevant_environment(),
    }
    return _json_safe(fingerprint)


def _without_rank_identity(value: Any, key: str = "") -> Any:
    """Remove per-rank identity while retaining hardware compatibility facts."""

    if isinstance(value, Mapping):
        result = {}
        for name, item in value.items():
            name_text = str(name).lower()
            if name_text == "participant_group_topology" and isinstance(item, Mapping):
                # Each rank sees the global-rank membership of its own PP/TP
                # subgroup.  Those lists legitimately differ between ranks in
                # a 2-D topology, but the capture contract only needs the
                # subgroup widths to agree for coordination.  The rank/size
                # fields in ``parallel`` retain the local topology identity.
                result[name] = {
                    str(group_name): {"world_size": group_data.get("world_size")}
                    for group_name, group_data in item.items()
                    if isinstance(group_data, Mapping)
                }
                continue
            if name_text in {
                "world_rank",
                "tp_rank",
                "pp_rank",
                "dp_rank",
                "attn_tp_rank",
                "attn_cp_rank",
                "attn_dcp_rank",
                "dcp_rank",
                "attn_dp_rank",
                "moe_ep_rank",
                "moe_tp_rank",
                "moe_dp_rank",
                "gpu_id",
                "uuid",
            }:
                continue
            if name_text == "device" and isinstance(item, str):
                # ``runner.device`` is commonly ``cuda:<local-rank>``. The
                # local index is process identity, not a graph-compatibility
                # property; GPU model/capability remain in the sibling fields.
                item = item.split(":", 1)[0]
            result[name] = _without_rank_identity(item, name_text)
        return result
    if isinstance(value, list):
        return [_without_rank_identity(item, key) for item in value]
    return value


def _cache_root(cache_dir: str | os.PathLike[str] | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    try:
        configured = envs.SGLANG_CACHE_DIR.get()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        configured = os.path.expanduser("~/.cache/sglang")
    return Path(os.path.expanduser(str(configured)))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> bool:
    """Atomically publish a JSON manifest, returning False on cache I/O errors."""

    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(payload, output, sort_keys=True, separators=(",", ":"))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return True
    except (OSError, TypeError, ValueError) as exc:
        logger.debug("Unable to write prefill CUDA graph cache %s: %s", path, exc)
        return False
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.debug("Unable to remove prefill CUDA graph cache %s: %s", path, exc)


class PrefillCudaGraphCache:
    """Rank-scoped manifest and warmup policy for one prefill runner."""

    def __init__(
        self,
        *,
        fingerprint: Mapping[str, Any],
        expected_shapes: Sequence[Any],
        cache_dir: str | os.PathLike[str] | None = None,
        rank: int = 0,
    ) -> None:
        self.fingerprint = _json_safe(dict(fingerprint))
        self.fingerprint_digest = fingerprint_digest(self.fingerprint)
        self.coordination_digest = fingerprint_digest(
            _without_rank_identity(self.fingerprint)
        )
        self.expected_shapes = _normalized_shapes(expected_shapes)
        self._expected_shape_ids = {
            _canonical_json(shape): shape for shape in self.expected_shapes
        }
        self.rank = int(rank)
        root = _cache_root(cache_dir) / _CACHE_SUBDIR
        self.path = root / f"{self.fingerprint_digest}.rank{self.rank}.json"
        self._staging_path = self.path.with_suffix(self.path.suffix + ".inprogress")
        self._records = {}
        self._capture_started = False
        self._runtime_fallback = False
        self._writable = True
        self.hit = False
        self.miss_reason = "cache file not found"
        self.warmup_iterations = DEFAULT_PREFILL_CUDA_GRAPH_WARMUP_ITERATIONS
        # ``recommended_warmup_iterations`` describes the policy that was
        # actually observed for every shape. ``probe_status`` controls whether
        # the next launch may try the faster policy once.
        self.probe_status = _PREFILL_CUDA_GRAPH_PROBE_PENDING
        self.next_warmup_iterations: int | None = None
        self._probe_disabled = False
        self._load()

    @classmethod
    def from_runner(
        cls,
        runner: Any,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
    ) -> PrefillCudaGraphCache:
        expected_shapes: list[dict[str, Any]] = []
        for size in getattr(runner, "capture_num_tokens", ()):
            expected_shapes.append({"size": int(size)})
        for variant in getattr(runner, "_prefix_capture_variants", ()):
            for size in getattr(runner, "capture_num_tokens", ()):
                expected_shapes.append(
                    {"size": int(size), "variant_label": f"chunked_prefix:{variant}"}
                )
        return cls(
            fingerprint=build_prefill_cuda_graph_fingerprint(runner, expected_shapes),
            expected_shapes=expected_shapes,
            cache_dir=cache_dir,
            rank=_rank_for_runner(runner),
        )

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            with self.path.open("r", encoding="utf-8") as source:
                payload = json.load(source)
        except (OSError, ValueError, TypeError) as exc:
            self.miss_reason = f"malformed manifest ({type(exc).__name__})"
            return
        if not isinstance(payload, Mapping):
            self.miss_reason = "manifest root is not an object"
            return
        if payload.get("schema_version") != PREFILL_CUDA_GRAPH_CACHE_SCHEMA_VERSION:
            self.miss_reason = "schema version mismatch"
            return
        if payload.get("code_version") != PREFILL_CUDA_GRAPH_CACHE_CODE_VERSION:
            self.miss_reason = "code version mismatch"
            return
        if payload.get("artifact_kind") != PREFILL_CUDA_GRAPH_CACHE_ARTIFACT_KIND:
            self.miss_reason = "unsupported artifact kind"
            return
        status = payload.get("status")
        if status not in ("complete", "disabled"):
            self.miss_reason = "manifest has an unsupported status"
            return
        if payload.get("fingerprint_digest") != self.fingerprint_digest:
            self.miss_reason = "fingerprint mismatch"
            return
        if _json_safe(payload.get("fingerprint")) != self.fingerprint:
            self.miss_reason = "fingerprint payload mismatch"
            return
        if not _is_valid_warmup_iterations(
            payload.get("recommended_warmup_iterations")
        ):
            self.miss_reason = "invalid warmup recommendation"
            return
        if type(payload.get("rank")) is not int or payload.get("rank") != self.rank:
            self.miss_reason = "rank mismatch"
            return
        if payload.get("coordination_digest") != self.coordination_digest:
            self.miss_reason = "coordination digest mismatch"
            return
        try:
            manifest_shapes = _normalized_shapes(payload.get("expected_shapes", []))
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            self.miss_reason = "invalid expected shape list"
            return
        if manifest_shapes != self.expected_shapes:
            self.miss_reason = "capture shape list mismatch"
            return
        records = payload.get("shapes")
        recommendation = payload["recommended_warmup_iterations"]
        probe_status = payload.get("probe_status")
        next_warmup = payload.get("next_warmup_iterations")
        if probe_status not in _VALID_PREFILL_CUDA_GRAPH_PROBE_STATES:
            self.miss_reason = "invalid probe status"
            return
        if status == "disabled" and probe_status != _PREFILL_CUDA_GRAPH_PROBE_DISABLED:
            self.miss_reason = "disabled manifest has an invalid probe status"
            return
        if probe_status == _PREFILL_CUDA_GRAPH_PROBE_PENDING:
            if recommendation != DEFAULT_PREFILL_CUDA_GRAPH_WARMUP_ITERATIONS:
                self.miss_reason = "pending probe has an invalid recommendation"
                return
            if not _is_valid_warmup_iterations(next_warmup) or (
                next_warmup != CACHED_PREFILL_CUDA_GRAPH_WARMUP_ITERATIONS
            ):
                self.miss_reason = "pending probe has an invalid next warmup"
                return
        elif probe_status == _PREFILL_CUDA_GRAPH_PROBE_VALIDATED:
            if recommendation != CACHED_PREFILL_CUDA_GRAPH_WARMUP_ITERATIONS:
                self.miss_reason = "validated probe has an invalid recommendation"
                return
            if next_warmup is not None:
                self.miss_reason = "validated probe has an unexpected next warmup"
                return
        else:
            if recommendation != DEFAULT_PREFILL_CUDA_GRAPH_WARMUP_ITERATIONS:
                self.miss_reason = "disabled probe has an invalid recommendation"
                return
            if next_warmup is not None:
                self.miss_reason = "disabled probe has an unexpected next warmup"
                return
            if status == "complete":
                # A complete disabled manifest still has to describe the
                # capture that produced it, even though it will not arm a
                # future one-warmup probe.
                valid, reason = self._validate_records(records, recommendation)
                if not valid:
                    self.miss_reason = reason
                    return
            elif not isinstance(records, Mapping):
                self.miss_reason = "disabled manifest has invalid shape records"
                return
            # Disabled markers intentionally do not carry a trusted shape
            # table; the next capture rebuilds it from the live runner.
            self._records = {}
            self.probe_status = _PREFILL_CUDA_GRAPH_PROBE_DISABLED
            self.next_warmup_iterations = None
            self._probe_disabled = True
            self.hit = False
            self.warmup_iterations = DEFAULT_PREFILL_CUDA_GRAPH_WARMUP_ITERATIONS
            self.miss_reason = "one-warmup probe disabled"
            return
        valid, reason = self._validate_records(records, recommendation)
        if not valid:
            self.miss_reason = reason
            return
        self._records = {str(key): dict(value) for key, value in records.items()}
        self.probe_status = probe_status
        self.next_warmup_iterations = next_warmup
        self.hit = True
        # A pending manifest has validated the two-warmup path, but the next
        # process is explicitly probing one warmup. Only a validated manifest
        # may skip that probe state transition.
        self.warmup_iterations = (
            next_warmup
            if probe_status == _PREFILL_CUDA_GRAPH_PROBE_PENDING
            else recommendation
        )
        self.miss_reason = "cache hit"

    def _validate_records(self, records: Any, recommendation: Any) -> tuple[bool, str]:
        """Validate the shape table and its warmup policy as one transaction."""

        if not isinstance(records, Mapping):
            return False, "shape records are missing"
        expected_ids = set(self._expected_shape_ids)
        record_ids = set(records)
        if record_ids != expected_ids:
            missing = expected_ids - record_ids
            extra = record_ids - expected_ids
            if missing:
                return False, f"shape record missing: {next(iter(missing))}"
            return False, f"unexpected shape record: {next(iter(extra))}"
        if not _is_valid_warmup_iterations(recommendation):
            return False, "invalid warmup recommendation"
        for shape_id, record in records.items():
            if not isinstance(shape_id, str):
                return False, "shape record key is not a string"
            if not isinstance(record, Mapping) or record.get("status") != "captured":
                return False, f"invalid shape record: {shape_id}"
            shape = record.get("shape")
            if not isinstance(shape, Mapping):
                return False, f"shape payload missing: {shape_id}"
            try:
                canonical_id = shape_key_id(shape)
            except (AttributeError, TypeError, ValueError, KeyError):
                return False, f"invalid shape payload: {shape_id}"
            if canonical_id != shape_id:
                return False, f"shape key mismatch: {shape_id}"
            warmup = record.get("warmup_iterations")
            if not _is_valid_warmup_iterations(warmup):
                return False, f"invalid warmup record: {shape_id}"
            if warmup != recommendation:
                return False, f"warmup recommendation mismatch: {shape_id}"
        return True, ""

    def synchronize(
        self, tp_group: Any, *, participant_groups: Sequence[Any] | None = None
    ) -> None:
        """Make rank-local policy safe for every group used by graph capture.

        ``tp_group`` is kept as the required compatibility argument.  When
        pipeline/context/expert parallel groups also participate in the outer
        graph-capture context, callers may pass them in ``participant_groups``
        so a miss on one stage cannot leave another stage on a different
        warmup policy.
        """

        groups = [tp_group]
        if participant_groups is not None:
            groups.extend(participant_groups)
        seen: set[int] = set()
        all_groups_compatible = True
        for group in groups:
            if group is None or id(group) in seen:
                continue
            seen.add(id(group))
            if not self._synchronize_group(group):
                # Every rank must execute the same sequence of collectives.
                # In particular, a failed TP subgroup can overlap a later PP/
                # CP subgroup; returning here would let ranks outside the
                # failed subgroup enter that later collective alone and hang.
                # Keep exchanging verdicts for the remaining groups, then let
                # the caller capture with the resulting conservative policy.
                all_groups_compatible = False

        if self.hit and all_groups_compatible:
            logger.info(
                "Prefill CUDA graph cache hit on all capture ranks (digest=%s, "
                "warmup=%d, probe=%s)",
                self.fingerprint_digest[:16],
                self.warmup_iterations,
                self.probe_status,
            )

    def _synchronize_group(self, group: Any) -> bool:
        try:
            world_size = int(getattr(group, "world_size", 1))
        except (TypeError, ValueError):
            world_size = 1
        if world_size <= 1:
            return True
        gather = getattr(group, "all_gather_object", None)
        if not callable(gather):
            self.disable("capture group cannot exchange cache verdicts")
            return False
        local = {
            "hit": self.hit,
            "coordination_digest": self.coordination_digest,
            "warmup_iterations": self.warmup_iterations,
            "probe_status": self.probe_status,
        }
        try:
            verdicts = gather(local)
        except (
            Exception
        ) as exc:  # noqa: BLE001 - distributed backends expose arbitrary errors
            self.disable(f"cache verdict exchange failed ({type(exc).__name__})")
            return False
        if not isinstance(verdicts, Sequence) or len(verdicts) != world_size:
            self.disable("cache verdict exchange returned an invalid rank count")
            return False
        if any(
            isinstance(verdict, Mapping)
            and verdict.get("probe_status") == _PREFILL_CUDA_GRAPH_PROBE_DISABLED
            for verdict in verdicts
        ):
            # A rank that has already observed a failed probe must prevent the
            # remaining ranks from arming that probe again.
            self.probe_status = _PREFILL_CUDA_GRAPH_PROBE_DISABLED
            self.next_warmup_iterations = None
            self._probe_disabled = True
            self.disable("one-warmup probe disabled on a capture rank")
            return False
        compatible = all(
            isinstance(verdict, Mapping)
            and verdict.get("hit") is True
            and verdict.get("coordination_digest") == self.coordination_digest
            and verdict.get("warmup_iterations") == self.warmup_iterations
            and verdict.get("probe_status") == self.probe_status
            for verdict in verdicts
        )
        if not compatible:
            self.disable("capture-rank cache verdict disagreement")
            return False
        return True

    def disable(self, reason: str) -> None:
        self.hit = False
        self.warmup_iterations = DEFAULT_PREFILL_CUDA_GRAPH_WARMUP_ITERATIONS
        self.miss_reason = reason
        logger.info("Prefill CUDA graph cache miss: %s", reason)

    def begin_capture(self) -> None:
        self._capture_started = True
        self._records = {}
        staging_payload = self._payload(status="capturing")
        self._writable = _atomic_write_json(self._staging_path, staging_payload)
        logger.info(
            "Prefill CUDA graph cache %s (path=%s, warmup_iterations=%d)",
            "hit" if self.hit else f"miss: {self.miss_reason}",
            self.path,
            self.warmup_iterations,
        )

    def record_shape(
        self,
        shape_key: Any,
        *,
        segment_count: int | None = None,
        elapsed_s: float | None = None,
        warmup_iterations: int | None = None,
    ) -> None:
        try:
            shape = shape_key_to_dict(shape_key)
        except (AttributeError, TypeError, ValueError, KeyError) as exc:
            self.disable(f"invalid captured shape ({type(exc).__name__})")
            self._runtime_fallback = True
            return
        shape_id = _canonical_json(shape)
        if shape_id not in self._expected_shape_ids:
            self.disable(f"unexpected captured shape: {shape_id}")
            self._runtime_fallback = True
            return
        actual_warmup = (
            warmup_iterations
            if warmup_iterations is not None
            else self.warmup_iterations
        )
        if not _is_valid_warmup_iterations(actual_warmup):
            self.disable(f"invalid warmup count for shape: {shape_id}")
            self._runtime_fallback = True
            return
        record: dict[str, Any] = {
            "status": "captured",
            "shape": shape,
            "warmup_iterations": actual_warmup,
            "captured_at": time.time(),
        }
        if segment_count is not None:
            try:
                segment_count = int(segment_count)
            except (TypeError, ValueError):
                segment_count = None
            if segment_count is not None and segment_count >= 0:
                record["segment_count"] = segment_count
        if elapsed_s is not None:
            try:
                if math.isfinite(elapsed_s):
                    record["elapsed_s"] = float(elapsed_s)
            except (TypeError, ValueError):
                pass
        self._records[shape_id] = record

    def mark_runtime_fallback(self, reason: str) -> None:
        """Persistently disable a one-warmup probe after incompatibility."""

        self._runtime_fallback = True
        self.probe_status = _PREFILL_CUDA_GRAPH_PROBE_DISABLED
        self.next_warmup_iterations = None
        self._probe_disabled = True
        self.disable(reason)
        # Publish a marker immediately. The current capture may fail before
        # all shape records exist, so use the dedicated ``disabled`` status;
        # the next launch will retain the historical two-warmup policy and
        # never arm the probe again for this fingerprint.
        published = _atomic_write_json(self.path, self._payload(status="disabled"))
        if not published:
            self._writable = False

    def _payload(
        self,
        *,
        status: str,
        recommended_warmup_iterations: int | None = None,
    ) -> dict[str, Any]:
        if recommended_warmup_iterations is None:
            recommended_warmup_iterations = self.warmup_iterations
        return {
            "schema_version": PREFILL_CUDA_GRAPH_CACHE_SCHEMA_VERSION,
            "code_version": PREFILL_CUDA_GRAPH_CACHE_CODE_VERSION,
            "artifact_kind": PREFILL_CUDA_GRAPH_CACHE_ARTIFACT_KIND,
            "status": status,
            "rank": self.rank,
            "fingerprint": self.fingerprint,
            "fingerprint_digest": self.fingerprint_digest,
            "coordination_digest": self.coordination_digest,
            "expected_shapes": self.expected_shapes,
            "shapes": self._records,
            # This records the policy that actually succeeded for every shape;
            # it never implies that an executable graph can be restored.
            "recommended_warmup_iterations": recommended_warmup_iterations,
            "probe_status": self.probe_status,
            "next_warmup_iterations": self.next_warmup_iterations,
            "updated_at": time.time(),
        }

    def commit(self) -> bool:
        if not self._capture_started or self._runtime_fallback:
            self.abort("capture was not eligible for commit")
            return False
        missing = set(self._expected_shape_ids) - set(self._records)
        if missing:
            self.abort(f"incomplete capture ({len(missing)} shape records missing)")
            return False
        recommendation = self._recommendation_from_records()
        if recommendation is None:
            self.abort("inconsistent or invalid warmup records")
            return False
        if self._probe_disabled:
            probe_status = _PREFILL_CUDA_GRAPH_PROBE_DISABLED
            next_warmup_iterations = None
        elif recommendation == CACHED_PREFILL_CUDA_GRAPH_WARMUP_ITERATIONS:
            probe_status = _PREFILL_CUDA_GRAPH_PROBE_VALIDATED
            next_warmup_iterations = None
        else:
            probe_status = _PREFILL_CUDA_GRAPH_PROBE_PENDING
            next_warmup_iterations = CACHED_PREFILL_CUDA_GRAPH_WARMUP_ITERATIONS
        self.probe_status = probe_status
        self.next_warmup_iterations = next_warmup_iterations
        if self._writable:
            published = _atomic_write_json(
                self.path,
                self._payload(
                    status="complete",
                    recommended_warmup_iterations=recommendation,
                ),
            )
            if not published:
                self._writable = False
        _unlink(self._staging_path)
        if self._writable:
            logger.info(
                "Committed prefill CUDA graph cache manifest (path=%s, shapes=%d)",
                self.path,
                len(self._records),
            )
        return self._writable

    def _recommendation_from_records(self) -> int | None:
        """Choose only a warmup policy observed for every captured shape."""

        valid, _ = self._validate_records(self._records, self.warmup_iterations)
        if not valid:
            # The current runner may have changed its policy after a fallback;
            # validate the records independently before selecting a policy.
            if not isinstance(self._records, Mapping):
                return None
            values = [
                record.get("warmup_iterations")
                for record in self._records.values()
                if isinstance(record, Mapping)
            ]
            if (
                not values
                or len(values) != len(self._records)
                or not all(_is_valid_warmup_iterations(value) for value in values)
                or len(set(values)) != 1
            ):
                return None
            recommendation = values[0]
            valid, _ = self._validate_records(self._records, recommendation)
            if not valid:
                return None
            return recommendation
        return self.warmup_iterations

    def abort(self, reason: str = "capture aborted") -> None:
        if reason:
            logger.info("Prefill CUDA graph cache not committed: %s", reason)
        _unlink(self._staging_path)
