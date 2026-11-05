"""Template: plug your own sandbox and workload into the correctness framework.

1. Copy this file somewhere importable and fill in every TODO.
2. python -m tvverify check --adapter my_workload:MyAdapter --suite contract
3. python -m tvverify check --adapter my_workload:MyAdapter

REQUIRED: MyToolCall.to_dict, MySandbox.start/fork/execute/stop,
          MySandbox.equals (or state_digest), MyAdapter.tool_call_from_key,
          MyAdapter.tasks, MyAdapter.sample_rollouts (or scripted_paths).
OPTIONAL: will_mutate_state, comparator_for, tool_class, digest_policy.
"""

import random
from typing import List, Optional, Sequence

from tvclient.sandbox import Sandbox
from tvclient.tool_call import ToolCall

from tvverify.adapters.base import TaskSpec, WorkloadAdapter
from tvverify.compare.base import Comparator, ExactComparator


class MyToolCall(ToolCall):
    """One tool invocation. Its serialize() output is the cache key."""

    def __init__(self, name: str, **args: str):
        self.name = name
        self.args = args

    def to_dict(self) -> dict:
        # TODO: everything that identifies the call, JSON-serializable.
        return {"name": self.name, "args": self.args}

    def will_mutate_state(self) -> bool:
        # OPTIONAL. True (the default) is always safe. Return False only for
        # tools that never change sandbox state (reads, SELECTs).
        return True


class MySandbox(Sandbox):
    """Construction conventions: MySandbox(task_name=...) is a fresh sandbox;
    MySandbox(task_name=..., env_id=handle) attaches to a snapshot handle that
    start() restores into a new live sandbox."""

    async def start(self) -> None:
        # TODO: boot a fresh sandbox, or restore self.env_id (a snapshot handle
        # from fork()). Afterwards self.env_id is the live sandbox's id.
        raise NotImplementedError("MySandbox.start")

    async def fork(self) -> "MySandbox":
        # TODO: snapshot this live sandbox (it keeps running) and return
        # MySandbox(task_name=self.task_name, env_id=<snapshot handle>).
        raise NotImplementedError("MySandbox.fork")

    async def execute(self, tool_call: ToolCall) -> str:
        # TODO: run the call here and return its output as a string.
        raise NotImplementedError("MySandbox.execute")

    async def stop(self) -> None:
        # TODO: delete this sandbox, or the snapshot if env_id is a handle.
        raise NotImplementedError("MySandbox.stop")

    async def equals(self, other: "MySandbox") -> bool:
        # TODO (REQUIRED unless you implement state_digest): do two LIVE
        # sandboxes hold the same state? E.g. `diff -r` of the filesystems, or
        # comparing two database dumps. Ignore noise you do not care about
        # (timestamps, pids), but not state a later call could read.
        #
        # Alternative: implement state_digest() returning
        # {"components": {"fs": {"hash": "...", "strict": True}}} and delete
        # this method; the default equals() compares the strict hashes.
        raise NotImplementedError("MySandbox.equals (or state_digest)")


class MyAdapter(WorkloadAdapter):
    name = "my_workload"
    sandbox_class = MySandbox
    tool_call_class = MyToolCall

    def tool_call_from_key(self, key: str) -> ToolCall:
        # TODO: invert MyToolCall.serialize(): key is its JSON string.
        raise NotImplementedError("MyAdapter.tool_call_from_key")

    def tasks(self) -> Sequence[TaskSpec]:
        # TODO: the tasks to check, e.g. [TaskSpec("task-1")].
        raise NotImplementedError("MyAdapter.tasks")

    def sample_rollouts(self, task_id: str, rng: random.Random, count: int) -> List[List[ToolCall]]:
        # TODO: `count` call lists for this task. Make them share prefixes, or
        # the cache is never exercised. (Or delete this and implement
        # scripted_paths(), a list of known-good call sequences.)
        raise NotImplementedError("MyAdapter.sample_rollouts")

    def comparator_for(self, key: str) -> Comparator:
        # OPTIONAL: outputs that legitimately vary (timestamps, ordering) need
        # a normalizing comparator, see tvverify.compare.
        return ExactComparator()
