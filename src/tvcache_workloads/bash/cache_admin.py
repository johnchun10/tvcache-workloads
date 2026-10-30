"""Cleanup of snapshots a task's cache tree still references.

Snapshots hold Docker images (cold) or running containers (parked), so an
experiment should release them when it is done. This goes through the cache's
own eviction path (mark, then drain) so the server never points at a deleted
snapshot.
"""

from typing import List, Tuple, Type

from tvclient import SandboxEvictionRecord, SandboxManager, TVCacheClient
from tvclient.sandbox import Sandbox


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
        for key, child in node.get("children", {}).items():
            stack.append((path + [key], child))
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
