"""Fresh-root replay: the independent oracle.

Replaying a path means booting a brand-new sandbox from the task's root and
executing the path's stateful calls, with no cache and no snapshots involved.
Checks that restore snapshots (`restore_*`) only prove consistency with the
mechanism under test, so reports keep the two apart.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tvverify.adapters.base import TEST_KEY, WorkloadAdapter, sandbox_equals
from tvverify.digest.schema import StateDigest


@dataclass
class ReplayStats:
    sandboxes_started: int = 0
    executions: int = 0


class ReplayEngine:
    def __init__(self, adapter: WorkloadAdapter, task_id: str = "replay", memoize: bool = True):
        """memoize=True assumes deterministic tools (true for the mock
        environments); real adapters pass False so every replay is fresh."""
        self.adapter = adapter
        self.task_id = task_id
        self.memoize = memoize
        self.stats = ReplayStats()
        self._digests: Dict[Tuple[str, ...], Optional[StateDigest]] = {}
        self._leaves: Dict[Tuple[Tuple[str, ...], str], str] = {}
        # Learned from the first probe: False means the sandbox has no
        # state_digest() (it only implements equals()), so digest replays are
        # skipped instead of booting a sandbox just to get None back.
        self.has_digest: Optional[bool] = None

    async def _boot(self, env_id: Optional[str] = None):
        sandbox = self.adapter.sandbox_class(task_name=self.task_id, env_id=env_id)
        await sandbox.start()
        self.stats.sandboxes_started += 1
        return sandbox

    async def _run(self, sandbox, keys: List[str]) -> None:
        for key in keys:
            self.stats.executions += 1
            await sandbox.execute(self.adapter.tool_call_from_key(key))

    async def state_digest(self, stateful_keys: List[str]) -> Optional[StateDigest]:
        """Digest of Replay(stateful_keys)."""
        cache_key = tuple(stateful_keys)
        if self.memoize and cache_key in self._digests:
            return self._digests[cache_key]
        if self.has_digest is False:
            return None
        sandbox = await self._boot()
        try:
            await self._run(sandbox, stateful_keys)
            digest = await self.adapter.probe_state(sandbox, self.adapter.digest_policy(self.task_id))
        finally:
            await sandbox.stop()
        self.has_digest = digest is not None
        self._digests[cache_key] = digest
        return digest

    async def leaf_output(self, stateful_keys: List[str], leaf_key: str) -> str:
        """Output of `leaf_key` executed after Replay(stateful_keys)."""
        cache_key = (tuple(stateful_keys), leaf_key)
        if self.memoize and cache_key in self._leaves:
            return self._leaves[cache_key]
        sandbox = await self._boot()
        try:
            await self._run(sandbox, stateful_keys)
            self.stats.executions += 1
            out = await sandbox.execute(self.adapter.tool_call_from_key(leaf_key))
        finally:
            await sandbox.stop()
        self._leaves[cache_key] = out
        return out

    async def test_output(self, stateful_keys: List[str]) -> str:
        return await self.leaf_output(stateful_keys, TEST_KEY)

    # ---- live-sandbox comparison, for sandboxes that implement equals() ----

    async def same_state(self, keys_a: List[str], keys_b: List[str]) -> bool:
        """Replay both paths from fresh roots and ask the sandboxes whether
        they ended in the same state (Sandbox.equals)."""
        a = await self._boot()
        try:
            b = await self._boot()
            try:
                await self._run(a, keys_a)
                await self._run(b, keys_b)
                return await sandbox_equals(a, b)
            finally:
                await b.stop()
        finally:
            await a.stop()

    async def matches_snapshot(self, keys: List[str], snapshot_id: str) -> bool:
        """Replay `keys` from a fresh root and compare with the restored snapshot."""
        replayed = await self._boot()
        try:
            restored = await self._boot(snapshot_id)
            try:
                await self._run(replayed, keys)
                return await sandbox_equals(replayed, restored)
            finally:
                await restored.stop()
        finally:
            await replayed.stop()

    # ---- snapshot-based checks (consistency only) ----

    async def restore_digest(self, snapshot_id: str) -> Optional[StateDigest]:
        sandbox = await self._boot(snapshot_id)
        try:
            return await self.adapter.probe_state(sandbox, self.adapter.digest_policy(self.task_id))
        finally:
            await sandbox.stop()

    async def restore_leaf(self, snapshot_id: str, leaf_key: str) -> str:
        """Fork-and-execute: restore the snapshot and run the leaf in it."""
        sandbox = await self._boot(snapshot_id)
        try:
            self.stats.executions += 1
            return await sandbox.execute(self.adapter.tool_call_from_key(leaf_key))
        finally:
            await sandbox.stop()

    async def purity(self, stateful_keys: List[str], leaf_key: str) -> Tuple[Optional[StateDigest], Optional[StateDigest]]:
        """A2: digests before and after a call that is declared stateless."""
        sandbox = await self._boot()
        try:
            await self._run(sandbox, stateful_keys)
            policy = self.adapter.digest_policy(self.task_id)
            before = await self.adapter.probe_state(sandbox, policy)
            self.stats.executions += 1
            await sandbox.execute(self.adapter.tool_call_from_key(leaf_key))
            after = await self.adapter.probe_state(sandbox, policy)
            return before, after
        finally:
            await sandbox.stop()
