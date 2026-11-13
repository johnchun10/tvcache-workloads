"""Cleanup of snapshots a task's cache tree still references.

Snapshots hold Docker images (cold) or running containers (parked), so an
experiment should release them when it is done. This goes through the cache's
own eviction path (mark, then drain) so the server never points at a deleted
snapshot.
"""

from typing import Dict, List, Tuple, Type

from tvclient import SandboxEvictionRecord, SandboxManager, TVCacheClient
from tvclient.sandbox import Sandbox


def node_children(node: dict) -> List[Tuple[Dict[str, dict], dict]]:
    """A dumped node's children, each as (its calls, the child node), where
    calls maps a call's cache key to {"value": ..., "runtime": ...}.

    Reads both dump shapes. A server without graph contraction has one call
    per node (`children` is a dict keyed by it, value and runtime sit on the
    node). With contraction a node can hold several merged calls (`children`
    is a list, each child has a `tool_calls` dict)."""
    children = node.get("children", {})
    if isinstance(children, dict):
        return [
            ({call: {"value": child.get("value"), "runtime": child.get("runtime", 0)}}, child)
            for call, child in children.items()
        ]
    return [(child.get("tool_calls", {}), child) for child in children]


def snapshot_nodes(dump: dict, task_id: str) -> List[Tuple[List[str], str]]:
    """(path, sandbox_id) for every node of the task's tree that holds a sandbox."""
    graph = dump.get(task_id)
    if graph is None:
        return []
    found: List[Tuple[List[str], str]] = []
    stack = [([], graph["root"])]
    while stack:
        path, node = stack.pop()
        if node.get("sandbox_id") is not None and path:
            found.append((path, node["sandbox_id"]))
        for calls, child in node_children(node):
            if calls:  # any of a merged node's calls names it
                stack.append((path + [sorted(calls)[0]], child))
    return found


async def release_task_snapshots(
    task_id: str,
    sandbox_class: Type[Sandbox],
    host: str = "127.0.0.1",
    port: int = 7878,
) -> List[str]:
    """Evicts and deletes every snapshot in task_id's tree; returns their ids.

    Uses SandboxManager.drain_once, which deletes *all* zero-reference victims
    on the server, not only this task's. tvclient requires exactly one drainer
    per server, so call this only when no other SandboxManager is running.
    Snapshots still leased by a running rollout stay until it releases them.
    """
    async with TVCacheClient(host, port) as client:
        nodes = snapshot_nodes(await client.dump_all(), task_id)
        if nodes:
            await client.mark_sandboxes_evicted(
                [
                    SandboxEvictionRecord(task_id=task_id, sandbox_path=path, sandbox_id=sid)
                    for path, sid in nodes
                ]
            )
    manager = SandboxManager(sandbox_class, host=host, port=port)
    try:
        drained = await manager.drain_once()
    finally:
        await manager.close()
    return [sandbox.sandbox_id for sandbox in drained]
