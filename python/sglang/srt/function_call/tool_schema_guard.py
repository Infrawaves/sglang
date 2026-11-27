"""Bounds for merging a single anyOf branch in the Kimi K3 tool grammar."""

from typing import Any, Dict


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# min > max makes walle degrade the enclosing schema to {}
# (SimplifyDegradeEnclosingSchema); XGrammar would refuse to compile it.
_BOUND_PAIRS = (
    ("minLength", "maxLength"),
    ("minimum", "maximum"),
    ("minItems", "maxItems"),
)


_BOUND_KEYS = frozenset(key for pair in _BOUND_PAIRS for key in pair)


def _has_contradicting_bounds(schema: Dict[str, Any]) -> bool:
    for low_key, high_key in _BOUND_PAIRS:
        low, high = schema.get(low_key), schema.get(high_key)
        if _is_number(low) and _is_number(high) and low > high:
            return True
    return False


def merge_anyof_branch(
    parent: Dict[str, Any], branch: Dict[str, Any]
) -> Dict[str, Any]:
    """``parent`` with its only ``anyOf`` branch merged in, as walle does: each
    bound keeps the stricter value, and min > max degrades the branch to ``{}``."""
    merged = {**parent, **branch}
    for low_key, high_key in _BOUND_PAIRS:
        for key, stricter in ((low_key, max), (high_key, min)):
            values = (parent.get(key), branch.get(key))
            if all(_is_number(value) for value in values):
                merged[key] = stricter(*values)
    if _has_contradicting_bounds(merged):
        # walle moved the parent's bounds into the branch it degraded.
        return {key: value for key, value in parent.items() if key not in _BOUND_KEYS}
    return merged
