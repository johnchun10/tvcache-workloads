"""In-memory sandbox whose state is the list of stateful calls applied.

This is the MockToolCallEnv that used to live in tvcache-rust/tests/
test_tvclient.py, rewritten to follow the current Sandbox contract:

  - Sandbox(task_name)               -> start() boots a fresh, empty sandbox
  - Sandbox(task_name, env_id=live)  -> attaches to a live sandbox
  - Sandbox(task_name, env_id=snap)  -> start() restores the snapshot into a NEW
                                        live sandbox (the snapshot stays intact)
  - fork()                           -> a cold handle to a copy of the state
  - stop()                           -> deletes the live sandbox or snapshot

Unknown ids raise instead of silently starting empty, so a cache that hands out
a wrong handle fails loudly.
"""

import asyncio
import itertools
from typing import Dict, List, Optional, Tuple

from tvclient.sandbox import Sandbox
from tvclient.tool_call import ToolCall


class CommandToolCall(ToolCall):
    def __init__(self, command: str, mutates: bool = True, slow: bool = False):
        self.command = command
        self.mutates = mutates
        self.slow = slow

    def to_dict(self) -> dict:
        return {"command": self.command}

    def will_mutate_state(self) -> bool:
        return self.mutates


class MockListSandbox(Sandbox):
    """State = the list of serialized stateful calls applied so far."""

    live: Dict[str, List[str]] = {}
    snapshots: Dict[str, List[str]] = {}
    execute_log: List[Tuple[str, str]] = []
    stop_log: List[str] = []
    # When True (the default), an unknown sandbox/snapshot id raises. Tests that
    # insert fake sandbox ids straight into the cache turn this off; the unknown
    # handle then restores as an empty sandbox.
    strict_ids = True
    slow_seconds = 0.08
    fast_seconds = 0.001
    _counter = itertools.count()

    @classmethod
    def reset(cls) -> None:
        cls.strict_ids = True
        cls.live.clear()
        cls.snapshots.clear()
        cls.execute_log.clear()
        cls.stop_log.clear()

    @classmethod
    def _new_id(cls) -> str:
        return f"env-{next(cls._counter)}"

    async def start(self) -> None:
        cls = MockListSandbox
        if self.env_id is None:
            self.env_id = cls._new_id()
            cls.live[self.env_id] = []
        elif self.env_id in cls.snapshots:
            restored_id = cls._new_id()
            cls.live[restored_id] = list(cls.snapshots[self.env_id])
            self.env_id = restored_id
        elif self.env_id not in cls.live:
            if cls.strict_ids:
                raise KeyError(f"unknown sandbox or snapshot {self.env_id!r}")
            restored_id = cls._new_id()
            cls.live[restored_id] = []
            self.env_id = restored_id

    async def fork(self) -> "MockListSandbox":
        cls = MockListSandbox
        snapshot_id = cls._new_id()
        cls.snapshots[snapshot_id] = list(cls.live[self.env_id])
        return MockListSandbox(task_name=self.task_name, env_id=snapshot_id)

    async def execute(self, tool_call: ToolCall) -> str:
        cls = MockListSandbox
        await asyncio.sleep(
            cls.slow_seconds if getattr(tool_call, "slow", False) else cls.fast_seconds
        )
        serialized = tool_call.serialize()
        cls.execute_log.append((self.env_id, serialized))
        if tool_call.will_mutate_state():
            cls.live[self.env_id].append(serialized)
        return f"result:{serialized}"

    async def stop(self) -> None:
        cls = MockListSandbox
        cls.stop_log.append(self.env_id)
        cls.live.pop(self.env_id, None)
        cls.snapshots.pop(self.env_id, None)

    @classmethod
    def state_of(cls, env_id: str) -> Optional[List[str]]:
        return cls.live.get(env_id, cls.snapshots.get(env_id))
