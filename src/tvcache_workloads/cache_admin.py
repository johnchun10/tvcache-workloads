"""Generic cleanup of snapshots still referenced by a task's cache tree."""

from typing import List, Tuple, Type

from tvclient import SandboxEvictionRecord, SandboxManager, TVCacheClient
from tvclient.sandbox import Sandbox


def snapshot_nodes(dump: dict, task_id: str) -> List[Tuple[List[str], str]]:
    """Return (path, sandbox_id) pairs for nodes holding a sandbox snapshot."""
    graph = dump.get(task_id)
    if graph is None:
        return []
    found: List[Tuple[List[str], str]] = []
    stack = [([], graph["root"])]
    while stack:
        path, node = stack.pop()
        if node.get("sandbox_id") is not None and path:
            found.append((path, node["sandbox_id"]))
        for key, child in node.get("children", {}).items():
            stack.append((path + [key], child))
    return found


async def release_task_snapshots(
    task_id: str,
    sandbox_class: Type[Sandbox],
    host: str = "127.0.0.1",
    port: int = 7878,
) -> List[str]:
    """Evict and delete task snapshots; returns their ids.

    SandboxManager drains all zero-reference evicted snapshots on the server,
    not only this task's. Call this only when no other drainer is running.
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