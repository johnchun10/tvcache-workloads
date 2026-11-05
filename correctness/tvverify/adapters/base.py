"""Workload adapter interface.

A workload (terminal-bench, SkyRL-SQL, a collaborator's new environment, or a
mock) plugs into the correctness framework by implementing this class. The
framework never assumes anything about a workload beyond it.

REQUIRED: sandbox_class, tool_call_class, tool_call_from_key, tasks, and one of
sample_rollouts / scripted_paths. Everything else has a default.
`correctness/templates/my_workload.py` is a fill-in-the-blanks starting point.
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence, Type

from tvclient.sandbox import Sandbox
from tvclient.tool_call import TestToolCall, ToolCall

from tvverify.compare.base import Comparator, ExactComparator
from tvverify.digest.schema import DigestPolicy, StateDigest


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    root_image_id: str = ""


TEST_KEY = TestToolCall("").serialize()


async def sandbox_equals(a: Sandbox, b: Sandbox) -> bool:
    """Whether two live sandboxes hold the same state: `a.equals(b)` when the
    sandbox has it, else the strict components of both state digests (the
    rule the default `Sandbox.equals` follows in tvclients that have it)."""
    equals = getattr(a, "equals", None)
    if equals is not None:
        return await equals(b)
    digests = [await s.state_digest() if hasattr(s, "state_digest") else None for s in (a, b)]
    if digests[0] is None or digests[1] is None:
        raise NotImplementedError(
            f"{type(a).__name__} must implement equals() or state_digest() "
            "so sandbox states can be compared"
        )
    return StateDigest.from_dict(digests[0]).strict_hash() == StateDigest.from_dict(digests[1]).strict_hash()


class WorkloadAdapter(ABC):
    name: str
    sandbox_class: Type[Sandbox]
    tool_call_class: Type[ToolCall]

    # ---- REQUIRED ---------------------------------------------------------

    @abstractmethod
    def tool_call_from_key(self, key: str) -> ToolCall:
        """Rebuilds the ToolCall a cache key (ToolCall.serialize()) came from."""

    @abstractmethod
    def tasks(self) -> Sequence[TaskSpec]:
        """The tasks the checks run on (one cache trie per task)."""

    def scripted_paths(self, task_id: str) -> List[List[ToolCall]]:
        """Known-good call sequences (e.g. a reference solution split into
        steps). The default source for sample_rollouts."""
        return []

    def sample_rollouts(self, task_id: str, rng: random.Random, count: int) -> List[List[ToolCall]]:
        """`count` rollouts (call lists) to drive through the cache. Rollouts
        should share prefixes, or the cache is never exercised. Override this,
        or provide scripted_paths and keep the default, which cycles through
        the scripts in order (so every script runs once count >= len)."""
        scripts = self.scripted_paths(task_id)
        if not scripts:
            raise NotImplementedError(
                f"{type(self).__name__} must implement sample_rollouts() or scripted_paths()"
            )
        return [list(scripts[i % len(scripts)]) for i in range(count)]

    # ---- OPTIONAL ---------------------------------------------------------

    def is_test_key(self, key: str) -> bool:
        return key == TEST_KEY

    def is_stateful(self, key: str) -> bool:
        if self.is_test_key(key):
            return False
        return self.tool_call_from_key(key).will_mutate_state()

    def tool_class(self, key: str) -> str:
        """Stratum label used to group results, e.g. "bash:pip" or "sql:select"."""
        return "test" if self.is_test_key(key) else "default"

    def comparator_for(self, key: str) -> Comparator:
        """How two outputs of this call are judged equal (default: exact)."""
        return ExactComparator()

    def digest_policy(self, task_id: str) -> DigestPolicy:
        return DigestPolicy()

    async def probe_state(self, sandbox: Sandbox, policy: DigestPolicy) -> Optional[StateDigest]:
        """The sandbox's state digest, or None when the sandbox only implements
        equals(), in which case the framework compares live sandboxes instead."""
        state_digest = getattr(sandbox, "state_digest", None)
        raw = await state_digest() if state_digest is not None else None
        return StateDigest.from_dict(raw) if raw is not None else None
