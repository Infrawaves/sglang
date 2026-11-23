"""Strict K3 tool schema checks ported from MoonshotAI/walle (validator.go).

The Kimi Vendor Verifier holds vendors to walle's verdicts. Where the port has to
choose, it accepts: a schema walle accepts must never get a 400 here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import msgspec

# Keeps a pathological ref graph from stalling the event loop; arbitrary,
# walle's own cap (maxRefWalkSteps) counts differently.
_MAX_REF_WALK_STEPS = 100_000

# Keywords whose values are data, not subschemas.
_LITERAL_KEYWORDS = frozenset({"const", "enum", "default", "examples"})


class ToolSchemaError(ValueError):
    """A strict tool's parameters schema that cannot be served."""


def _resolve_ref(ref: str, root: Dict[str, Any]) -> Optional[Any]:
    if ref == "#":
        return root
    if not ref.startswith("#/"):
        return None
    node: Any = root
    for part in ref[2:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and key in node:
            node = node[key]
        elif isinstance(node, list) and key.isdigit() and int(key) < len(node):
            node = node[int(key)]
        else:
            return None
    return node


class _Verdict(msgspec.Struct, frozen=True):
    finite: bool  # the schema has a finite instance
    hit_path_stack: bool  # the answer depended on the path stack: not memoized


_FINITE = _Verdict(finite=True, hit_path_stack=False)
_INFINITE = _Verdict(finite=False, hit_path_stack=False)


class _TerminationChecker:
    def __init__(self, root: Dict[str, Any]):
        self.root = root
        # Verdict per fully expanded $ref, valid only when the walk that produced
        # it never hit the path stack (walle caches under the same rule).
        self.memo: Dict[str, bool] = {}
        self.steps = 0

    def terminates(self, schema: Any, visited: Set[str]) -> _Verdict:
        self.steps += 1
        if self.steps > _MAX_REF_WALK_STEPS:
            raise ToolSchemaError(
                "tool parameters schema: reference graph is too complex to validate"
            )
        if schema is False:
            return _INFINITE
        if not isinstance(schema, dict):
            return _FINITE

        types = _type_list(schema)
        if types and any(t not in ("object", "array") for t in types):
            return _FINITE
        check_type = types[-1] if types else None

        hit_path_stack = False
        stuck = False

        if check_type == "array":
            min_items = schema.get("minItems")
            # Without a positive minItems the empty array is an instance.
            if not (_is_number(min_items) and min_items > 0):
                return _FINITE
            items = schema.get("items")
            if items is None or items == {} or items is True:
                return _FINITE
            verdict = self.terminates(schema=items, visited=visited)
            if verdict.finite:
                return verdict
            hit_path_stack = verdict.hit_path_stack
            stuck = True

        if check_type == "object":
            required = schema.get("required")
            if not isinstance(required, list) or not required:
                return _FINITE
            properties = schema.get("properties")
            if not isinstance(properties, dict) or not properties:
                return _FINITE
            required_names = {name for name in required if isinstance(name, str)}
            # Every required property must terminate; siblings pointing at the
            # same $ref are not a cycle, so each walks its own visited set.
            for name, sub_schema in properties.items():
                if name not in required_names:
                    continue
                verdict = self.terminates(schema=sub_schema, visited=set(visited))
                if not verdict.finite:
                    return verdict
                hit_path_stack = hit_path_stack or verdict.hit_path_stack
            return _Verdict(finite=True, hit_path_stack=hit_path_stack)

        any_of = schema.get("anyOf")
        if isinstance(any_of, list):
            if not any_of:
                return _FINITE
            seen_in_branches: Set[str] = set()
            for branch in any_of:
                branch_visited = set(visited)
                verdict = self.terminates(schema=branch, visited=branch_visited)
                if verdict.finite:
                    visited |= branch_visited
                    return verdict
                hit_path_stack = hit_path_stack or verdict.hit_path_stack
                seen_in_branches |= branch_visited
            visited |= seen_in_branches
            stuck = True

        ref = schema.get("$ref")
        if isinstance(ref, str):
            if ref in visited:
                return _Verdict(finite=False, hit_path_stack=True)
            if ref in self.memo:
                return _Verdict(finite=self.memo[ref], hit_path_stack=False)
            visited.add(ref)
            target = _resolve_ref(ref=ref, root=self.root)
            if target is None:
                # Not ours to judge: an unresolvable ref is left to the schema
                # validators and the grammar compiler.
                return _FINITE
            verdict = self.terminates(schema=target, visited=visited)
            if not verdict.hit_path_stack:
                self.memo[ref] = verdict.finite
            return verdict

        # Walle answers "no" for any other non-empty schema reaching this point;
        # only the array/anyOf dead ends above are certain, so accept the rest.
        return _Verdict(finite=not stuck, hit_path_stack=hit_path_stack)


def check_tool_schema_terminates(schema: Any) -> None:
    """Raise ToolSchemaError when a ``$ref`` walle checks has no finite instance.

    Like walle's TraverseAndCheckRefs, this walks only the keywords walle keeps
    (properties, items, anyOf); properties under an explicit empty ``required``
    are not checked, and an optional property's ref still is.
    """
    if not isinstance(schema, dict):
        return
    checker = _TerminationChecker(schema)

    def walk(node: Any, need_check: bool, path: List[str]) -> None:
        if not isinstance(node, dict):
            return
        ref = node.get("$ref")
        if need_check and isinstance(ref, str):
            if not checker.terminates(schema=node, visited=set()).finite:
                where = "/".join(path) or "<root>"
                raise ToolSchemaError(
                    "Invalid tool parameters schema: detected infinite recursion "
                    f"without termination condition at {where} ($ref {ref!r} "
                    "admits no finite instance)."
                )
        properties = node.get("properties")
        if isinstance(properties, dict):
            child_check = need_check and node.get("required") != []
            for name, child in properties.items():
                walk(
                    node=child,
                    need_check=child_check,
                    path=path + ["properties", name],
                )
        if isinstance(node.get("items"), dict):
            walk(node=node["items"], need_check=need_check, path=path + ["items"])
        any_of = node.get("anyOf")
        if isinstance(any_of, list):
            for index, child in enumerate(any_of):
                walk(
                    node=child, need_check=need_check, path=path + ["anyOf", str(index)]
                )

    try:
        walk(node=schema, need_check=True, path=[])
    except RecursionError:
        # A ref chain deeper than the interpreter stack is accepted, like
        # anything else the port cannot decide.
        return


def _type_list(schema: Dict[str, Any]) -> Optional[List[str]]:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return [schema_type]
    if isinstance(schema_type, list):
        return [t for t in schema_type if isinstance(t, str)]
    return None


def _allows_object(schema: Dict[str, Any]) -> bool:
    types = _type_list(schema)
    return types is None or "object" in types


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _matches_type(value: Any, json_type: str) -> bool:
    if json_type == "null":
        return value is None
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "integer":
        return _is_number(value) and float(value).is_integer()
    if json_type == "number":
        return _is_number(value)
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "array":
        return isinstance(value, list)
    if json_type == "object":
        return isinstance(value, dict)
    return True


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


def canonicalize_tool_schema(schema: Any) -> Any:
    """Copy of ``schema`` with the walle repairs the K3 grammar needs."""
    if isinstance(schema, list):
        return [canonicalize_tool_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    result: Dict[str, Any] = {}
    for key, value in schema.items():
        if key in _LITERAL_KEYWORDS:
            result[key] = value
        elif key == "properties" and isinstance(value, dict):
            # Property names are not keywords: recurse into the values only.
            result[key] = {
                name: canonicalize_tool_schema(sub) for name, sub in value.items()
            }
        else:
            result[key] = canonicalize_tool_schema(value)

    if _has_contradicting_bounds(result):
        return {}

    types = _type_list(result)
    enum = result.get("enum")
    if types and isinstance(enum, list):
        kept = [
            value
            for value in enum
            if any(_matches_type(value=value, json_type=t) for t in types)
        ]
        if kept:
            result["enum"] = kept
        else:
            # The type stays: it is the tighter of the two.
            result.pop("enum")

    required = result.get("required")
    if isinstance(required, list):
        properties = result.get("properties")
        if not _allows_object(result) or not isinstance(properties, dict):
            result.pop("required")
        else:
            seen: Set[str] = set()
            kept_names = []
            for name in required:
                if isinstance(name, str) and name in properties and name not in seen:
                    seen.add(name)
                    kept_names.append(name)
            if kept_names:
                result["required"] = kept_names
            else:
                result.pop("required")
    return result
