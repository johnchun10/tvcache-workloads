"""Independent reference model of the exact TVCache trie.

Written from the documented semantics (the insert/find rules below, as fixed for B2),
not ported from the Rust, so a disagreement between this model and the server
is a finding about one of them. The semantics modelled:

  insert(path, value, sandbox_id):
    - the parent must exist, else the insert is refused (False, False)
    - values are first-writer-wins; an identical value agrees, a different one
      conflicts; a conflict is refused and cannot attach a sandbox
    - a node without a value (placeholder / sandbox-only) accepts one
    - sandboxes are first-writer-wins
  find(path): longest-prefix match. Returns the value at the full path (if any)
    and the deepest sandbox on the walked prefix. A returned sandbox is leased
    (reference count + 1) until unref'd.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# bytes of fixed per-node overhead in the server's size accounting:
# runtime(4) + ttl_present(1) + ttl(4) + inserted_at(8) + cache_hits(8) + lpm_hits(8)
NODE_OVERHEAD_BYTES = 33


@dataclass
class RefNode:
    children: Dict[str, "RefNode"] = field(default_factory=dict)
    value: Optional[str] = None
    sandbox_id: Optional[str] = None
    runtime: int = 0
    cache_hits: int = 0
    lpm_hits: int = 0


@dataclass(frozen=True)
class Resolution:
    value: Optional[str]
    sandbox_path: Optional[List[str]]
    sandbox_id: Optional[str]

    @property
    def resume_depth(self) -> int:
        """Depth of the sandbox in the REQUEST path's own terms."""
        return len(self.sandbox_path) if self.sandbox_path else 0


def node_size(key: str, node: RefNode) -> int:
    return (
        len(key.encode())
        + (len(node.value.encode()) if node.value is not None else 0)
        + (len(node.sandbox_id.encode()) if node.sandbox_id is not None else 0)
        + NODE_OVERHEAD_BYTES
    )


class RefTrie:
    def __init__(self) -> None:
        self.root = RefNode()
        self.references: Dict[str, int] = {}
        self.reclaimable: Set[str] = set()

    # ---- operations -----------------------------------------------------

    def insert(
        self, path: List[str], value: Optional[str], sandbox_id: Optional[str], runtime: int = 0
    ) -> Tuple[bool, bool]:
        if not path:
            return (False, False)
        parent = self._walk(path[:-1])
        if parent is None:
            return (False, False)

        key = path[-1]
        node = parent.children.get(key)
        is_new = node is None
        if is_new:
            node = RefNode()
            parent.children[key] = node

        conflict = (
            not is_new and node.value is not None and value is not None and node.value != value
        )
        if is_new or (value is not None and node.value is None):
            node.value = value
            node.runtime = runtime

        sandbox_attached = False
        if sandbox_id is not None and node.sandbox_id is None and not conflict:
            node.sandbox_id = sandbox_id
            sandbox_attached = True
        return (not conflict, sandbox_attached)

    def find(self, path: List[str], lease: bool = True) -> Resolution:
        node = self.root
        sandbox_depth: Optional[int] = None
        sandbox_id: Optional[str] = None
        value: Optional[str] = None
        full_match = False
        for depth, key in enumerate(path, start=1):
            child = node.children.get(key)
            if child is None:
                break
            node = child
            if node.sandbox_id is not None:
                sandbox_id, sandbox_depth = node.sandbox_id, depth
            if depth == len(path):
                full_match = True
                value = node.value
                node.cache_hits += 1
        if value is None and sandbox_depth is not None:
            self._walk(path[:sandbox_depth]).lpm_hits += 1
        if sandbox_id is not None and lease:
            self.references[sandbox_id] = self.references.get(sandbox_id, 0) + 1
        sandbox_path = path[:sandbox_depth] if sandbox_depth is not None else None
        return Resolution(value, sandbox_path, sandbox_id)

    def unref(self, sandbox_id: str) -> int:
        count = self.references.get(sandbox_id)
        if count is None:
            return 0
        if count <= 1:
            del self.references[sandbox_id]
            return 0
        self.references[sandbox_id] = count - 1
        return count - 1

    def mark_evicted(self, sandbox_path: List[str], sandbox_id: str) -> None:
        self.reclaimable.add(sandbox_id)
        node = self._walk(sandbox_path)
        if node is None or not sandbox_path or node.sandbox_id != sandbox_id:
            return
        node.sandbox_id = None

    def drain(self) -> List[str]:
        ready = sorted(s for s in self.reclaimable if s not in self.references)
        self.reclaimable -= set(ready)
        return ready

    def remove_subtree(self, path: List[str]) -> List[str]:
        """Evicts the node at `path` and its descendants (what TTL/watermark
        eviction does); returns the sandbox ids that become reclaimable."""
        parent = self._walk(path[:-1])
        if parent is None or path[-1] not in parent.children:
            return []
        removed = parent.children.pop(path[-1])
        ids = [n.sandbox_id for _, n in _walk_nodes(removed) if n.sandbox_id]
        self.reclaimable.update(ids)
        return ids

    # ---- inspection -----------------------------------------------------

    def _walk(self, path: List[str]) -> Optional[RefNode]:
        node = self.root
        for key in path:
            node = node.children.get(key)
            if node is None:
                return None
        return node

    def counters(self) -> Dict[str, int]:
        nodes = sandboxes = size = 0
        for key, node in _walk_nodes(self.root, with_root=False):
            nodes += 1
            sandboxes += node.sandbox_id is not None
            size += node_size(key, node)
        return {"number_of_nodes": nodes, "number_of_sandboxes": sandboxes, "size_in_bytes": size}

    def diff_against_dump(self, dump: dict) -> List[str]:
        """Compares this model with one task's DumpAll entry; returns a list of
        human-readable differences (empty means identical). Ignores the TTL
        clock, which depends on wall time."""
        problems: List[str] = []
        for name, want in self.counters().items():
            if dump[name] != want:
                problems.append(f"{name}: server {dump[name]} != model {want}")
        if dump["sandbox_references"] != self.references:
            problems.append(
                f"sandbox_references: server {dump['sandbox_references']} != model {self.references}"
            )
        if set(dump["reclaimable_sandboxes"]) != self.reclaimable:
            problems.append(
                f"reclaimable: server {dump['reclaimable_sandboxes']} != model {sorted(self.reclaimable)}"
            )
        _diff_node(self.root, dump["root"], [], problems)
        return problems


def _walk_nodes(node: RefNode, key: str = "", with_root: bool = True):
    if with_root:
        yield key, node
    for child_key, child in node.children.items():
        yield from _walk_nodes(child, child_key, True)


def _diff_node(model: RefNode, server: dict, path: List[str], problems: List[str]) -> None:
    where = "/".join(path) or "<root>"
    if path:
        for field_name, want in (
            ("value", model.value),
            ("sandbox_id", model.sandbox_id),
            ("runtime", model.runtime),
            ("number_of_cache_hits", model.cache_hits),
            ("number_of_lpm_hits", model.lpm_hits),
        ):
            if server[field_name] != want:
                problems.append(f"{where}: {field_name} server {server[field_name]!r} != model {want!r}")
    if set(server["children"]) != set(model.children):
        problems.append(
            f"{where}: children server {sorted(server['children'])} != model {sorted(model.children)}"
        )
    for key, child in model.children.items():
        if key in server["children"]:
            _diff_node(child, server["children"][key], path + [key], problems)
