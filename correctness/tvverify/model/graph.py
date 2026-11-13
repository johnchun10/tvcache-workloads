"""Standard export of a task's cache trie (tcg-export/v2) that the validators consume.

Built from the server's DumpAll, live or saved to a file. Each node keeps the
one path of calls from the root that reaches it.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from tvclient import TVCacheClient

from tvverify.model.dump import plain_dump

SCHEMA = "tcg-export/v2"


def path_id(path: List[str]) -> str:
    return "n" + hashlib.sha1(json.dumps(path).encode()).hexdigest()[:12]


@dataclass
class GNode:
    node_id: str
    key: Optional[str]  # incoming edge label; None for the root
    stateful: Optional[bool] = None  # from the adapter; the server does not know
    tool_class: str = "default"
    value: Optional[str] = None
    sandbox_id: Optional[str] = None
    runtime_ms: int = 0
    hits: int = 0
    lpm_hits: int = 0
    path: List[str] = field(default_factory=list)  # the calls from the root to this node
    # On a server with graph contraction, a node can hold several calls. The
    # others appear as childless aliases of the first: this is its node id.
    # An alias has its own value and shares the node's snapshot (model/dump.py).
    alias_of: Optional[str] = None


@dataclass
class GraphExport:
    task_id: str
    root_id: str
    nodes: Dict[str, GNode]
    edges: Dict[str, Dict[str, str]]  # node_id -> edge label -> child node_id
    counters: Dict[str, int] = field(default_factory=dict)
    sandbox_references: Dict[str, int] = field(default_factory=dict)
    reclaimable: List[str] = field(default_factory=list)
    root_image_id: str = ""
    epoch: int = 0
    schema: str = SCHEMA

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @staticmethod
    def from_json(text: str) -> "GraphExport":
        data = json.loads(text)
        return GraphExport(**{**data, "nodes": {k: GNode(**v) for k, v in data["nodes"].items()}})

    def children(self, node_id: str) -> Dict[str, GNode]:
        return {label: self.nodes[child] for label, child in self.edges.get(node_id, {}).items()}


def from_dumpall(dump: dict, adapter=None, root_image_id: str = "") -> Dict[str, GraphExport]:
    """Builds one GraphExport per task from a DumpAll result."""
    exports: Dict[str, GraphExport] = {}
    for task_id, tcg in plain_dump(dump).items():
        nodes: Dict[str, GNode] = {}
        edges: Dict[str, Dict[str, str]] = {}
        root_id = "root"
        nodes[root_id] = GNode(root_id, None)

        def visit(node: dict, node_id: str, path: List[str]) -> None:
            for label, child in node["children"].items():
                child_path = path + [label]
                child_id = path_id(child_path)
                nodes[child_id] = GNode(
                    node_id=child_id,
                    key=label,
                    stateful=None if adapter is None else adapter.is_stateful(label),
                    tool_class="default" if adapter is None else adapter.tool_class(label),
                    value=child["value"],
                    sandbox_id=child["sandbox_id"],
                    runtime_ms=child["runtime"],
                    hits=child["number_of_cache_hits"],
                    lpm_hits=child["number_of_lpm_hits"],
                    path=child_path,
                    alias_of=path_id(path + [child["alias_of"]]) if child.get("alias_of") else None,
                )
                edges.setdefault(node_id, {})[label] = child_id
                visit(child, child_id, child_path)

        visit(tcg["root"], root_id, [])
        exports[task_id] = GraphExport(
            task_id=task_id,
            root_id=root_id,
            nodes=nodes,
            edges=edges,
            counters={
                k: tcg[k] for k in ("number_of_nodes", "number_of_sandboxes", "size_in_bytes")
            },
            sandbox_references=dict(tcg["sandbox_references"]),
            reclaimable=list(tcg["reclaimable_sandboxes"]),
            root_image_id=root_image_id,
        )
    return exports


async def export_task_graph(port: int, task_id: str, adapter=None, root_image_id: str = "") -> GraphExport:
    """DumpAll from a running server, exported as the trie of one task."""
    async with TVCacheClient(port=port) as client:
        dump = await client.dump_all()
    return from_dumpall({task_id: dump[task_id]}, adapter, root_image_id)[task_id]
