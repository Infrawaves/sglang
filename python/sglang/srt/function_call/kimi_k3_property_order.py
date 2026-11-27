"""Declared property order for walle's canonical schemas.

Walle emits properties sorted by name. The K3 grammar, and XGrammar inside a
nested object, admit declared properties only in declaration order, which is
also the order the prompt shows; restore_property_order puts it back.

Stdlib only: walle_worker.py loads this file by path, outside the sglang
package, so the restore runs within walle's execution timeout and never on the
serving event loop. Every walk is iterative and memoized, and visits only the
keywords walle keeps: walle accepts an original nested deeply under a keyword
it drops (not, allOf, x-...).
"""

from collections import deque
from functools import partial
from typing import Any, Callable, Deque, Dict, FrozenSet, List, Optional, Tuple

# The subschema keywords walle keeps below a schema's root. A schema with none
# of them declares nothing a canonical object could take its order from.
_STRUCTURE_KEYWORDS = frozenset(
    {"properties", "items", "additionalProperties", "anyOf"}
)
_DEFS_REF_PREFIX = "#/$defs/"
_NO_TYPES = object()

_Sources = List[dict]


def restore_property_order(*, canonical: Any, original: Any) -> Any:
    """Reorder, in place, each canonical object's properties as the original
    declares them, and return ``canonical``.

    A canonical object is matched to the original object(s) it came from by
    path, through walle's $ref inlining, anyOf distribution and dropped anyOf
    branches; a property no match declares falls back to where the original
    first declares that name."""
    if isinstance(canonical, dict) and isinstance(original, dict):
        _PropertyOrderRestorer(original).restore(canonical)
    return canonical


class _PropertyOrderRestorer:
    def __init__(self, original: Dict[str, Any]):
        self.root = original
        self.first_declared = _first_declarations(original)
        # Canonical $defs name -> the original schemas a use site matched it to.
        self.def_sources: Dict[str, _Sources] = {}
        self.reached_defs: Deque[str] = deque()
        # id(original schema) -> the structural schemas it and its $ref chain
        # declare with, targets first; shared between the chain's members.
        self._expansions: Dict[int, _Sources] = {}
        self._types: Dict[int, Any] = {}

    def restore(self, canonical: Dict[str, Any]) -> None:
        self._restore_tree(
            node=canonical, sources_fn=partial(self._expand, nodes=[self.root])
        )
        defs = canonical.get("$defs")
        if not isinstance(defs, dict):
            return
        done = set()
        by_name = iter(defs)
        while len(done) < len(defs):
            # Defs a $ref has reached are matched there; the rest by name.
            name = None
            while self.reached_defs and name is None:
                candidate = self.reached_defs.popleft()
                if candidate in defs and candidate not in done:
                    name = candidate
            if name is None:
                name = next(candidate for candidate in by_name if candidate not in done)
            done.add(name)
            if not isinstance(defs[name], dict):
                continue
            # A def keeping its original $defs name copies that schema. Walle
            # renames the defs it hoists from other pointers (definitions/...);
            # those take the sources of the use site that reached them.
            target = self._def_target(name)
            if target is not None:
                sources_fn = partial(self._expand, nodes=[target])
            else:
                sources_fn = partial(_constant, value=self.def_sources.get(name, []))
            self._restore_tree(node=defs[name], sources_fn=sources_fn)

    def _restore_tree(
        self, *, node: Dict[str, Any], sources_fn: Callable[[], _Sources]
    ) -> None:
        stack = [(node, sources_fn, None, None)]
        while stack:
            node, sources_fn, producer, own = stack.pop()
            children = self._restore_node(
                node=node, sources_fn=sources_fn, producer=producer, own=own
            )
            stack.extend(reversed(children))

    def _restore_node(
        self,
        *,
        node: Dict[str, Any],
        sources_fn: Callable[[], _Sources],
        producer: Optional[int],
        own: Any,
    ) -> List[Tuple[dict, Callable[[], _Sources], Optional[int], Any]]:
        """Reorder this node's properties; its children with their sources,
        each computed only if that child needs them.

        For an anyOf branch, ``own`` is the original branch it came from and
        ``producer`` the id of the source whose anyOf held it: that anyOf is
        spent, and must not align an anyOf nested in the branch."""
        ref = node.get("$ref")
        reaches_def = False
        if isinstance(ref, str) and ref.startswith(_DEFS_REF_PREFIX):
            name = ref[len(_DEFS_REF_PREFIX) :]
            reaches_def = (
                name not in self.def_sources and self._def_target(name) is None
            )
        structural = not _STRUCTURE_KEYWORDS.isdisjoint(node)
        if not (reaches_def or structural):
            return []
        sources = sources_fn()
        if reaches_def:
            self.def_sources[name] = sources
            self.reached_defs.append(name)
        if not structural:
            return []
        children: List[Tuple[dict, Callable[[], _Sources], Optional[int], Any]] = []
        for key, value in node.items():
            if key == "properties" and isinstance(value, dict):
                node[key] = self._ordered_properties(properties=value, sources=sources)
                for name, child in node[key].items():
                    if isinstance(child, dict):
                        children.append(
                            (
                                child,
                                partial(
                                    self._property_sources, sources=sources, name=name
                                ),
                                None,
                                None,
                            )
                        )
            elif key in ("items", "additionalProperties") and isinstance(value, dict):
                children.append(
                    (
                        value,
                        partial(self._keyword_sources, sources=sources, key=key),
                        None,
                        None,
                    )
                )
            elif key == "anyOf" and isinstance(value, list):
                aligned, branch_producer = self._aligned_branches(
                    branches=value, sources=sources, producer=producer, own=own
                )
                for branch, original in zip(value, aligned):
                    if isinstance(branch, dict):
                        children.append(
                            (
                                branch,
                                partial(
                                    self._branch_sources,
                                    sources=sources,
                                    original=original,
                                ),
                                branch_producer,
                                original,
                            )
                        )
        return children

    def _property_sources(self, *, sources: _Sources, name: str) -> _Sources:
        return self._expand(
            nodes=[
                source["properties"].get(name)
                for source in sources
                if isinstance(source.get("properties"), dict)
            ]
        )

    def _keyword_sources(self, *, sources: _Sources, key: str) -> _Sources:
        return self._expand(nodes=[source.get(key) for source in sources])

    def _branch_sources(self, *, sources: _Sources, original: Any) -> _Sources:
        # Walle distributes the parent's keywords into every branch, so the
        # parent's declarations come first, then the branch's own.
        if original is None:
            return sources
        return _merged(first=sources, second=self._expand(nodes=[original]))

    def _aligned_branches(
        self, *, branches: list, sources: _Sources, producer: Optional[int], own: Any
    ) -> Tuple[List[Any], Optional[int]]:
        """The original anyOf branch each canonical branch came from (or None),
        and the id of the source whose anyOf they are."""
        # The most specific source first: a branch's own original, then the
        # parent's sources whose anyOf did not already produce this branch.
        candidates = self._expand(nodes=[own]) if own is not None else []
        taken = {id(source) for source in candidates}
        if producer is not None:
            taken.add(producer)
        candidates = candidates + [s for s in sources if id(s) not in taken]
        for source in candidates:
            original = _anyof(source)
            if len(original) == len(branches):
                return list(original), id(source)
        for source in candidates:
            original = _anyof(source)
            if len(original) > len(branches):
                aligned = self._subsequence(
                    branches=branches, original=original, source=source, sources=sources
                )
                return aligned, id(source)
        return [None] * len(branches), None

    def _subsequence(
        self, *, branches: list, original: list, source: dict, sources: _Sources
    ) -> List[Any]:
        # Walle only drops branches and keeps the rest in order, so each
        # canonical branch is the next compatible original one.
        parent_names = set()
        for parent in sources:
            declared = parent.get("properties")
            if isinstance(declared, dict):
                parent_names.update(declared)
        parent_types = self._declared_types(source)
        aligned: List[Any] = []
        position = 0
        for branch in branches:
            match = None
            while position < len(original):
                candidate = original[position]
                position += 1
                if self._compatible(
                    branch=branch,
                    candidate=candidate,
                    parent_names=parent_names,
                    parent_types=parent_types,
                ):
                    match = candidate
                    break
            aligned.append(match)
        return aligned

    def _compatible(
        self,
        *,
        branch: Any,
        candidate: Any,
        parent_names: set,
        parent_types: Optional[FrozenSet[str]],
    ) -> bool:
        if not isinstance(branch, dict) or not isinstance(candidate, dict):
            return False
        ref = branch.get("$ref")
        if isinstance(ref, str):
            # Walle keeps a $ref only where the original had one.
            return self._reaches(candidate=candidate, canonical_ref=ref)
        declared_types = self._declared_types(candidate)
        if (
            parent_types is not None
            and declared_types is not None
            and not parent_types & declared_types
        ):
            # Walle drops a branch whose type the parent's type excludes.
            return False
        canonical_types = _type_set(branch.get("type"))
        candidate_types = declared_types or parent_types
        if canonical_types is not None and candidate_types is not None:
            if not canonical_types <= candidate_types:
                return False
        names = branch.get("properties")
        if not isinstance(names, dict) or not names:
            return True
        declared = set(parent_names)
        for schema in self._expand(nodes=[candidate]):
            if isinstance(schema.get("properties"), dict):
                declared.update(schema["properties"])
        return all(name in declared for name in names)

    def _ordered_properties(
        self, *, properties: Dict[str, Any], sources: _Sources
    ) -> Dict[str, Any]:
        local_order: Dict[str, int] = {}
        for source in sources:
            declared = source.get("properties")
            if isinstance(declared, dict):
                for name in declared:
                    local_order.setdefault(name, len(local_order))

        def order(name: str) -> Tuple[int, int]:
            if name in local_order:
                return 0, local_order[name]
            return 1, self.first_declared.get(name, len(self.first_declared))

        return {name: properties[name] for name in sorted(properties, key=order)}

    def _expand(self, *, nodes: List[Any]) -> _Sources:
        """The structural schemas among ``nodes`` and their $ref chains, each
        chain's targets first: a ref's target declares before the siblings walle
        merges into it. Must not be mutated: it may be shared."""
        if len(nodes) == 1:
            return self._expansion(nodes[0])
        expanded: _Sources = []
        seen = set()
        for node in nodes:
            for schema in self._expansion(node):
                if id(schema) not in seen:
                    seen.add(id(schema))
                    expanded.append(schema)
        return expanded

    def _expansion(self, node: Any) -> _Sources:
        if not isinstance(node, dict):
            return []
        known = self._expansions.get(id(node))
        if known is not None:
            return known
        chain = []
        on_chain = set()
        current: Any = node
        base: _Sources = []
        while isinstance(current, dict):
            known = self._expansions.get(id(current))
            if known is not None:
                base = known
                break
            if id(current) in on_chain:
                break  # a $ref cycle
            on_chain.add(id(current))
            chain.append(current)
            current = self._ref_target(current)
        for member in reversed(chain):
            # A pure alias (or an annotation-only schema) shares its target's list.
            if not _STRUCTURE_KEYWORDS.isdisjoint(member):
                base = base + [member]
            self._expansions[id(member)] = base
        return self._expansions[id(node)]

    def _declared_types(self, node: Any) -> Optional[FrozenSet[str]]:
        """The ``type`` the schema or its $ref chain declares first, if any."""
        chain = []
        on_chain = set()
        current: Any = node
        found: Any = None
        while isinstance(current, dict):
            known = self._types.get(id(current), _NO_TYPES)
            if known is not _NO_TYPES:
                found = known
                break
            if id(current) in on_chain:
                break
            on_chain.add(id(current))
            chain.append(current)
            if "type" in current:
                found = _type_set(current["type"])
                break
            current = self._ref_target(current)
        for member in chain:
            self._types[id(member)] = found
        return found

    def _reaches(self, *, candidate: dict, canonical_ref: str) -> bool:
        """Whether the candidate's $ref chain leads where the canonical $ref points."""
        target: Any = None
        if canonical_ref.startswith(_DEFS_REF_PREFIX):
            target = self._def_target(canonical_ref[len(_DEFS_REF_PREFIX) :])
        has_ref = False
        on_chain = set()
        current: Any = candidate
        while isinstance(current, dict) and id(current) not in on_chain:
            if target is not None and current is target:
                return True
            on_chain.add(id(current))
            ref = current.get("$ref")
            if not isinstance(ref, str):
                break
            if ref == canonical_ref:
                return True
            has_ref = True
            current = self._ref_target(current)
        # A canonical name traced to no original: any $ref branch may be it.
        return has_ref and target is None

    def _def_target(self, name: str) -> Any:
        """The original $defs schema of the same name, or None."""
        pool = self.root.get("$defs")
        target = pool.get(name) if isinstance(pool, dict) else None
        return target if isinstance(target, dict) else None

    def _ref_target(self, node: dict) -> Any:
        ref = node.get("$ref")
        return (
            _resolve_pointer(ref=ref, root=self.root) if isinstance(ref, str) else None
        )


def _first_declarations(root: Dict[str, Any]) -> Dict[str, int]:
    """Position of each property name where the original first declares it,
    over the keywords walle keeps."""
    order: Dict[str, int] = {}
    stack: List[Any] = [root]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        children: List[Any] = []
        for key, value in node.items():
            if key == "properties" and isinstance(value, dict):
                for name in value:
                    order.setdefault(name, len(order))
                children.extend(value.values())
            elif key in ("$defs", "definitions") and isinstance(value, dict):
                children.extend(value.values())
            elif key in ("items", "additionalProperties"):
                children.append(value)
            elif key == "anyOf" and isinstance(value, list):
                children.extend(value)
        stack.extend(reversed(children))
    return order


def _type_set(value: Any) -> Optional[FrozenSet[str]]:
    if isinstance(value, str):
        types = {value}
    elif isinstance(value, list):
        types = {item for item in value if isinstance(item, str)}
    else:
        return None
    if "number" in types:
        types.add("integer")
    return frozenset(types)


def _constant(*, value: _Sources) -> _Sources:
    return value


def _merged(*, first: _Sources, second: _Sources) -> _Sources:
    seen = {id(source) for source in first}
    return first + [source for source in second if id(source) not in seen]


def _anyof(schema: dict) -> list:
    branches = schema.get("anyOf")
    return branches if isinstance(branches, list) else []


def _resolve_pointer(*, ref: str, root: Dict[str, Any]) -> Any:
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
