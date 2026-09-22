# SPDX-License-Identifier: Apache-2.0
"""Activation entry point for the Foundry CUDA-graph persistence extension.

Foundry keeps its SGLang integration in its own package
(``foundry.integration.sglang``); the only thing it needs from this tree is a
module it can rely on being importable as ``sglang.srt.foundry_shim``. All this
file does is install Foundry's runtime monkey-patches.

Two responsibilities live elsewhere on purpose:

* **Flag forcing** is a resolution concern, so it is a pipeline handler
  (``sglang.srt.arg_groups.foundry_hook``). Writing a field here would be
  invisible to the published config bags the runtime actually reads.
* **Per-process installation** is the caller's job. ``spawn`` does not inherit
  Python state and ``resolve_once`` short-circuits in a child (the record
  arrives already resolved), so every process that needs the patches calls
  :func:`apply_server_args` itself.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def apply_server_args(server_args) -> None:
    """Install Foundry's runtime patches in this process, if Foundry is on.

    Idempotent: Foundry guards ``install_hooks`` with a module-level flag, so
    repeated calls in one process are no-ops.

    Call this in every process that either builds a ``ModelRunner`` or spawns a
    process that does. A spawner needs it because ``LD_PRELOAD`` takes effect at
    process start: the hook library has to be in the *parent's* environment
    before the child is forked, and Foundry sets that from its spawn-site
    patches.
    """
    if not server_args.foundry_graph_extension_config_path:
        return

    try:
        from foundry.integration.sglang.hooks import install_hooks
    except ImportError as e:
        raise ImportError(
            "--foundry-graph-extension-config-path was set but the 'foundry' "
            "package is not importable. Install it with "
            "`pip install -e . --no-build-isolation` from the foundry checkout."
        ) from e

    install_hooks(server_args)
