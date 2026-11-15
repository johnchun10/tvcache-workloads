"""The training run's control channel to the fork bank.

One process beside the cache server (bash/scripts/perlmutter/cache_sidecar.py)
owns the fork bank (TVCache's prewarming). The trainer tells it where each
step begins and ends and which tasks come next, and rollout workers ask it for
ready-made sandboxes. Requests are one JSON line per connection, answered by
one JSON line. (Graph contraction has its own port: bash/contraction.py.)

    {"op": "take", "task": "<cache task id>", "parent": "<snapshot id or root>"}
    -> {"env_id": "<live id>" | null}

Ops (served by cache_sidecar.py): take, warm, begin, end, stats.
"""

import asyncio
import json
import os
import socket
from typing import Optional, Tuple

from tvclient import AbstractForkGenerator

CONTROL_PORT_ENV = "TVCACHE_CONTROL_PORT"


def control_address() -> Optional[Tuple[str, int]]:
    port = os.environ.get(CONTROL_PORT_ENV)
    if not port:
        return None
    return os.environ.get("TVCACHE_HOST", "127.0.0.1"), int(port)


def call(op: str, timeout: float = 3600.0, **fields) -> dict:
    """Blocking request (the trainer's hooks are synchronous)."""
    address = control_address()
    if address is None:
        return {}
    with socket.create_connection(address, timeout=timeout) as sock:
        sock.sendall((json.dumps({"op": op, **fields}) + "\n").encode())
        reply = sock.makefile("r").readline()
    return json.loads(reply) if reply else {}


async def acall(op: str, timeout: float = 60.0, **fields) -> dict:
    address = control_address()
    if address is None:
        return {}
    reader, writer = await asyncio.wait_for(asyncio.open_connection(*address), timeout)
    try:
        writer.write((json.dumps({"op": op, **fields}) + "\n").encode())
        await writer.drain()
        reply = await asyncio.wait_for(reader.readline(), timeout)
    finally:
        writer.close()
    return json.loads(reply) if reply else {}


class RemoteForkGenerator(AbstractForkGenerator):
    """The executor's fork bank, living in the sidecar process (one shelf for
    every rollout worker)."""

    async def get_forked_env(self, task_name: str, parent_env_id: str) -> Optional[str]:
        return (await acall("take", task=task_name, parent=parent_env_id)).get("env_id")
