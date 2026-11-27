"""Walle canonicalization worker, run as a standalone script by walle_pool.py.

Writes ``{"ready": true}`` once walle is loaded, then answers each stdin line
(one schema JSON) with one stdout line
``{"canonical": str | null, "warning": str | null, "error": str | null}``,
``canonical`` with the properties back in declaration order. Exits at stdin EOF.

libwalle embeds a Go runtime that breaks after fork and must be called from the
main thread, so it only ever runs here, in a fresh interpreter that imports
nothing from sglang.
"""

import importlib.util
import json
import sys
import traceback
from pathlib import Path

from walle import WalleValidator


def _load_restore_property_order():
    path = Path(__file__).with_name("kimi_k3_property_order.py")
    spec = importlib.util.spec_from_file_location("kimi_k3_property_order", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.restore_property_order


def _declared_order(*, canonical: str, original: str, restore) -> str:
    if canonical == "{}":
        return canonical
    try:
        restored = restore(
            canonical=json.loads(canonical), original=json.loads(original)
        )
    except Exception:
        # Walle's own order still constrains correctly; only the prompt differs.
        traceback.print_exc()
        return canonical
    return json.dumps(restored, ensure_ascii=False, separators=(",", ":"))


def main() -> None:
    restore = _load_restore_property_order()
    validator = WalleValidator()
    # Loads libwalle before the first schema, so its start-up is not billed to it.
    validator.canonical_schema("{}")
    out = sys.stdout.buffer
    out.write(b'{"ready": true}\n')
    out.flush()
    for line in sys.stdin.buffer:
        schema = line.rstrip(b"\n").decode("utf-8")
        try:
            canonical, warning = validator.canonical_schema(schema)
            canonical = _declared_order(
                canonical=canonical, original=schema, restore=restore
            )
            reply = {"canonical": canonical, "warning": warning, "error": None}
        except ValueError as exc:
            reply = {"canonical": None, "warning": None, "error": str(exc)}
        out.write(json.dumps(reply).encode("ascii") + b"\n")
        out.flush()


if __name__ == "__main__":
    main()
