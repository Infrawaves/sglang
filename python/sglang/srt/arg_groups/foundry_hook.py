# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for the Foundry CUDA-graph persistence extension.

Foundry persists captured decode CUDA graphs to disk and restores them on a
later start, skipping capture. Restoration is only sound when the SAVE process
and the LOAD process walk the same device-allocation sequence, so this handler
forces off the features whose allocation pattern Foundry cannot reproduce, and
rejects the ones it cannot force.

Runs late in the resolution pipeline: it overrides what
``handle_cuda_graph_config`` resolved and reads the resolved speculative
algorithm, so it must come after both.
"""

from __future__ import annotations

import logging
from typing import Any

from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase

logger = logging.getLogger(__name__)


def handle_foundry_graph_extension(server_args: Any):
    """Force the capture configuration Foundry's SAVE/LOAD contract requires.

    Decode is pinned to ``full`` — the only capture shape Foundry serializes.
    Prefill capture is disabled: Foundry has no prefill archive format, and
    ``FullCudaGraphBackend`` is shared between the decode and prefill runners,
    so leaving prefill on ``full`` would feed prefill's token-count shape keys
    into the decode archive.
    """
    cfg = resolving_view(server_args)
    if not cfg.foundry_graph_extension_config_path:
        return

    _reject_features_foundry_cannot_honor(server_args)

    config = cfg.cuda_graph_config
    if config.decode.backend == Backend.DISABLED:
        raise ValueError(
            "Foundry persists decode CUDA graphs, but decode CUDA graph capture "
            "is disabled. Drop --disable-cuda-graph / "
            "--cuda-graph-backend-decode=disabled, or drop "
            "--foundry-graph-extension-config-path."
        )

    if config.decode.backend != Backend.FULL:
        logger.info(
            "Foundry: forcing cuda_graph_config[decode].backend from %r to 'full'.",
            config.decode.backend,
        )
    if config.prefill.backend != Backend.DISABLED:
        logger.warning(
            "Foundry: disabling prefill CUDA graph capture (was %r). Foundry "
            "persists decode graphs only, and the 'full' backend object is "
            "shared with the prefill runner.",
            config.prefill.backend,
        )

    config = with_phase(config, Phase.DECODE, backend=Backend.FULL)
    config = with_phase(config, Phase.PREFILL, backend=Backend.DISABLED)

    declare_resolution(
        server_args,
        "handle_foundry_graph_extension",
        cuda_graph_config=config,
        # The capture-time profiler allocates on SAVE; LOAD never enters the
        # capture loop and cannot reproduce those allocations.
        enable_profile_cuda_graph=False,
        # Autotuning picks kernels from timing, so SAVE and LOAD can choose
        # differently for one shape while the restored graph holds SAVE's pick.
        disable_flashinfer_autotune=True,
    )

    _warn_on_unpinned_mamba_cache(server_args)


def _reject_features_foundry_cannot_honor(server_args: Any) -> None:
    """Fail fast rather than corrupt an archive or serve wrong logits."""
    cfg = resolving_view(server_args)

    if cfg.enable_pdmux:
        raise ValueError(
            "Foundry does not support --enable-pdmux: PD-multiplexing captures "
            "one graph per stream, and Foundry's archive keys a graph by batch "
            "size alone."
        )

    if cfg.enable_memory_saver:
        raise ValueError(
            "Foundry does not support --enable-memory-saver: Foundry captures "
            "through its own graph context, bypassing the memory-saver "
            "adapter's tagged allocations."
        )

    if cfg.speculative_algorithm is not None:
        raise ValueError(
            "Foundry does not support speculative decoding "
            f"(--speculative-algorithm={cfg.speculative_algorithm!r}): "
            "draft/verify capture keys graphs by token count and adds "
            "per-variant graphs Foundry's archive cannot express."
        )


def _warn_on_unpinned_mamba_cache(server_args: Any) -> None:
    """Warn when a linear-attention model leaves ``max_mamba_cache_size`` unset.

    Unpinned, the size is solved from measured free GPU memory, which differs
    between the SAVE and LOAD processes. It is not carried in
    ``MemoryPoolConfig``, so Foundry's saved-pool-config short-circuit does not
    restore it: the state pool would be sized differently on LOAD and every
    later allocation would shift, leaving restored graphs pointing at the wrong
    addresses. Foundry records and replays the value too, so this warns rather
    than raises.
    """
    cfg = resolving_view(server_args)
    if cfg.max_mamba_cache_size is not None:
        return
    if mambaish_config(server_args.get_model_config()) is None:
        return

    logger.warning(
        "Foundry: --max-mamba-cache-size is unset on a linear-attention model. "
        "Its default is solved from measured free GPU memory, which differs "
        "between SAVE and LOAD. Foundry will record the SAVE value and replay "
        "it, but pinning it explicitly on both runs is the reliable fix."
    )
