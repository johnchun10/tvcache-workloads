"""C4: structural invariants of a task's cache trie.

Pure graph checks: no sandboxes, no replay.
"""

from typing import Callable, Dict, List, Optional

from tvverify.adapters.base import WorkloadAdapter
from tvverify.model.graph import GraphExport
from tvverify.validators.findings import Finding


def _f(code: str, detail: str, node_id: Optional[str] = None) -> Finding:
    return Finding("C4", code, detail, node_id, tier=0)


def recount(g: GraphExport) -> Dict[str, int]:
    # Aliases are further calls of a merged node, not nodes of their own.
    non_root = [n for i, n in g.nodes.items() if i != g.root_id and n.alias_of is None]
    return {
        "number_of_nodes": len(non_root),
        "number_of_sandboxes": sum(1 for n in non_root if n.sandbox_id),
    }


def check_structure(
    g: GraphExport,
    *,
    adapter: Optional[WorkloadAdapter] = None,
    snapshot_exists: Optional[Callable[[str], bool]] = None,
    quiescent: bool = True,
) -> List[Finding]:
    out: List[Finding] = []
    nodes, edges = g.nodes, g.edges

    # edges well-formed
    for src, targets in edges.items():
        if src not in nodes:
            out.append(_f("dangling_edge", f"edges from missing node {src}", src))
        for label, dst in targets.items():
            if dst not in nodes:
                out.append(_f("dangling_edge", f"{src} --{label[:30]}--> missing {dst}", src))

    # a tree: the root has no parent, every other node exactly one, and a
    # node's path is its parent's path plus the edge label
    if g.root_id not in nodes:
        out.append(_f("no_root", "root node missing"))
        return out
    parents: Dict[str, List[str]] = {}
    for src, targets in edges.items():
        for label, dst in targets.items():
            if dst not in nodes:
                continue
            parents.setdefault(dst, []).append(src)
            if src in nodes and list(nodes[dst].path) != list(nodes[src].path) + [label]:
                out.append(_f("path_mismatch", f"node {dst} records path {nodes[dst].path[-2:]} but sits under {src} --{label[:30]}-->", dst))
    if g.root_id in parents:
        out.append(_f("root_has_parent", f"root is a child of {parents[g.root_id]}", g.root_id))
    for n, who in parents.items():
        if len(who) > 1:
            out.append(_f("multiple_parents", f"node {n} has parents {who}", n))

    reachable = {g.root_id}
    stack = [g.root_id]
    while stack:
        for dst in edges.get(stack.pop(), {}).values():
            if dst in nodes and dst not in reachable:
                reachable.add(dst)
                stack.append(dst)
    for n in nodes:
        if n not in reachable:
            out.append(_f("unreachable", f"node {n} cannot be reached from the root", n))

    # stateless nodes are leaves
    if adapter is not None:
        for n, node in nodes.items():
            if n != g.root_id and node.key is not None and not adapter.is_stateful(node.key) and edges.get(n):
                out.append(_f("stateless_not_leaf", f"stateless node {n} has children", n))

    # sandbox ownership
    owners: Dict[str, List[str]] = {}
    for n, node in nodes.items():
        if node.sandbox_id and node.alias_of is None:  # an alias shares its node's snapshot
            owners.setdefault(node.sandbox_id, []).append(n)
    for sandbox_id, who in owners.items():
        if len(who) > 1:
            out.append(_f("shared_sandbox", f"sandbox {sandbox_id} is attached to {who}", who[0]))
        if sandbox_id in g.reclaimable:
            out.append(_f("reclaimable_in_use", f"sandbox {sandbox_id} is reclaimable but still attached", who[0]))
        if snapshot_exists is not None and not snapshot_exists(sandbox_id):
            out.append(_f("missing_snapshot", f"sandbox {sandbox_id} of node {who[0]} does not exist", who[0]))
    for sandbox_id, count in g.sandbox_references.items():
        if count > 0 and sandbox_id not in owners and sandbox_id not in g.reclaimable:
            out.append(_f("referenced_sandbox_orphaned", f"sandbox {sandbox_id} is leased {count}x but attached nowhere and not reclaimable"))
    if quiescent and any(c > 0 for c in g.sandbox_references.values()):
        out.append(_f("leaked_reference", f"leases outstanding while quiescent: {g.sandbox_references}"))

    # counters
    want = recount(g)
    for name in ("number_of_nodes", "number_of_sandboxes"):
        if name in g.counters and g.counters[name] != want[name]:
            out.append(_f("counter_mismatch", f"{name} is {g.counters[name]} but recounts to {want[name]}"))
    return out
