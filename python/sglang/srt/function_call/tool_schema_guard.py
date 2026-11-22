"""Walle-compatible guards for strict tool parameter schemas.

Moonshot's own serving stack judges tool schemas with MoonshotAI/walle, and the
Kimi Vendor Verifier (``tests/tool_call_json_schema``) holds vendors to walle's
verdicts. Two of them matter before a schema reaches the K3 structural tag:

* Rejection: a schema in which some ``$ref`` can never terminate has no finite
  instance. XGrammar compiles it anyway, and a forced call then runs to
  ``max_tokens``. Walle rejects it (``PostValidateRefs``), so we answer 400.
* Canonicalization: what walle repairs instead of rejecting must not break the
  grammar either. ``required`` naming undeclared properties made the K3 builder
  raise, and ``get_structure_constraint`` swallowed the error -- the call then
  ran with no constraint at all; contradicting bounds and an enum with no value
  of the schema's type made XGrammar refuse to compile (request aborted).

This is a narrow port of walle's ``CheckRefTermination`` /
``TraverseAndCheckRefs`` (validator.go), ``SimplifyPruneRequired``,
``simplifyIntersectEnumWithType`` and ``SimplifyDegradeEnclosingSchema``. Where the
port has to choose, it errs towards accepting: a schema walle accepts must never
get a 400 here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

# A pathological ref graph must not stall the event loop; walle has the same cap.
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


def _is_positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


class _TerminationChecker:
    def __init__(self, root: Dict[str, Any]):
        self.root = root
        # Verdict per fully expanded $ref, valid only when the walk that produced
        # it never hit the path stack (walle caches under the same rule).
        self.memo: Dict[str, bool] = {}
        self.steps = 0

    def terminates(self, schema: Any, visited: Set[str]) -> Tuple[bool, bool]:
        """(has a finite instance, verdict depended on the path stack)"""
        self.steps += 1
        if self.steps > _MAX_REF_WALK_STEPS:
            raise ToolSchemaError(
                "tool parameters schema: reference graph is too complex to validate"
            )
        if schema is False:
            return False, False
        if not isinstance(schema, dict):
            return True, False

        check_type = None
        schema_type = schema.get("type")
        if isinstance(schema_type, str):
            if schema_type not in ("object", "array"):
                return True, False
            check_type = schema_type
        elif isinstance(schema_type, list):
            for item in schema_type:
                if isinstance(item, str) and item not in ("object", "array"):
                    return True, False
                check_type = item

        cut = False
        stuck = False

        if check_type == "array":
            # Without a positive minItems the empty array is an instance.
            if not _is_positive_number(schema.get("minItems")):
                return True, False
            items = schema.get("items")
            if items is None or items == {} or items is True:
                return True, False
            ok, items_cut = self.terminates(items, visited)
            if ok:
                return True, items_cut
            cut = cut or items_cut
            stuck = True

        if check_type == "object":
            required = schema.get("required")
            if not isinstance(required, list) or not required:
                return True, False
            properties = schema.get("properties")
            if not isinstance(properties, dict) or not properties:
                return True, False
            required_names = {name for name in required if isinstance(name, str)}
            # Every required property must terminate; siblings pointing at the
            # same $ref are not a cycle, so each walks its own visited set.
            for name, sub_schema in properties.items():
                if name not in required_names:
                    continue
                ok, prop_cut = self.terminates(sub_schema, set(visited))
                if not ok:
                    return False, prop_cut
                cut = cut or prop_cut
            return True, cut

        any_of = schema.get("anyOf")
        if isinstance(any_of, list):
            if not any_of:
                return True, False
            seen_in_branches: Set[str] = set()
            for branch in any_of:
                branch_visited = set(visited)
                ok, branch_cut = self.terminates(branch, branch_visited)
                if ok:
                    visited |= branch_visited
                    return True, branch_cut
                cut = cut or branch_cut
                seen_in_branches |= branch_visited
            visited |= seen_in_branches
            stuck = True

        ref = schema.get("$ref")
        if isinstance(ref, str):
            if ref in visited:
                return False, True
            if ref in self.memo:
                return self.memo[ref], False
            visited.add(ref)
            target = _resolve_ref(ref, self.root)
            if target is None:
                # Not ours to judge: an unresolvable ref is left to the schema
                # validators and the grammar compiler, as before this check.
                return True, False
            ok, target_cut = self.terminates(target, visited)
            if not target_cut:
                self.memo[ref] = ok
            return ok, target_cut

        # Walle answers "no" for any other non-empty schema reaching this point;
        # only the array/anyOf dead ends above are certain, so accept the rest.
        return (not stuck), cut


def check_tool_schema_terminates(schema: Any) -> None:
    """Raise ToolSchemaError when a reachable ``$ref`` has no finite instance.

    Mirrors walle's TraverseAndCheckRefs: every ``$ref`` met while walking the
    schema (``$defs`` are only reached through refs) must terminate, including
    one under an optional property -- an optional property can still be emitted.
    ``additionalProperties`` and the properties of an object with an explicit
    empty ``required`` are not checked, as in walle.
    """
    if not isinstance(schema, dict):
        return
    checker = _TerminationChecker(schema)

    def walk(node: Any, need_check: bool, path: List[str]) -> None:
        if not isinstance(node, dict):
            return
        ref = node.get("$ref")
        if need_check and isinstance(ref, str):
            ok, _ = checker.terminates(node, set())
            if not ok:
                where = "/".join(path) or "<root>"
                raise ToolSchemaError(
                    "Invalid tool parameters schema: detected infinite recursion "
                    f"without termination condition at {where} ($ref {ref!r} "
                    "admits no finite instance)."
                )
        for key, value in node.items():
            if key in ("$defs", "definitions") or key in _LITERAL_KEYWORDS:
                continue
            if isinstance(value, dict):
                child_check = need_check
                if key == "properties" and node.get("required") == []:
                    child_check = False
                elif key == "additionalProperties":
                    child_check = False
                walk(value, child_check, path + [key])
            elif key == "anyOf" and isinstance(value, list):
                for index, item in enumerate(value):
                    walk(item, need_check, path + [f"anyOf/{index}"])

    walk(schema, True, [])


def _allows_object(schema: Dict[str, Any]) -> bool:
    schema_type = schema.get("type")
    if schema_type is None:
        return True
    if isinstance(schema_type, str):
        return schema_type == "object"
    if isinstance(schema_type, list):
        return "object" in schema_type
    return True


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


# Pairs walle treats as unsatisfiable when min > max (keyword_validators.go).
# The enclosing schema degrades to {} (SimplifyDegradeEnclosingSchema) rather
# than failing: XGrammar refuses to compile such a schema and the request would
# otherwise be aborted.
_BOUND_PAIRS = (
    ("minLength", "maxLength"),
    ("minimum", "maximum"),
    ("minItems", "maxItems"),
)


def _has_contradicting_bounds(schema: Dict[str, Any]) -> bool:
    for low_key, high_key in _BOUND_PAIRS:
        low, high = schema.get(low_key), schema.get(high_key)
        if _is_number(low) and _is_number(high) and low > high:
            return True
    return False


def canonicalize_tool_schema(schema: Any) -> Any:
    """Copy of ``schema`` with the walle canonicalizations the K3 grammar needs.

    * bounds with min > max: the enclosing schema becomes ``{}``;
    * ``enum`` keeps only values matching ``type``, and is dropped when none
      does (the type stays: it is the tighter of the two);
    * ``required`` keeps only names declared in ``properties``, and is dropped
      when nothing is left, when there is no ``properties``, or when the schema
      cannot be an object.

    Literal values (enum/const/default/examples) are never rewritten.
    """
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

    schema_type = result.get("type")
    type_list = (
        [schema_type]
        if isinstance(schema_type, str)
        else [t for t in schema_type if isinstance(t, str)]
        if isinstance(schema_type, list)
        else None
    )
    enum = result.get("enum")
    if type_list and isinstance(enum, list):
        kept = [v for v in enum if any(_matches_type(v, t) for t in type_list)]
        if kept:
            result["enum"] = kept
        else:
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
