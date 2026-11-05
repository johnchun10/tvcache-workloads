"""The replay validator for a task's cache trie.

For each node, replays the node's path in a fresh sandbox booted from the
task's root (no cache, no snapshots) and checks what the cache claims:

  C1   the cached value is what executing the node's call gives
  C3   the same for the task's test call; and a node's snapshot passes or
       fails the test the same way the replayed state does
  C2a  the node's snapshot holds the state the replay reaches
  A2   (opt-in) a call the adapter calls stateless did not change the state

The structural checks (C4) run first; they need no sandboxes.

A replay that could not run (its output starts with "Failed to execute", or
the check raised) gives a finding with code `replay_failed` / `replay_error`:
it proves nothing about the cache, and `check` reports it as inconclusive.
"""

from typing import Callable, Iterable, List, Optional

FAILED_PREFIX = "Failed to execute"  # tvclient's convention for a call the sandbox could not run
INCONCLUSIVE_CODES = ("replay_failed", "replay_error")

from tvverify.adapters.base import TEST_KEY, WorkloadAdapter
from tvverify.digest.schema import diff_digests
from tvverify.model.graph import GraphExport
from tvverify.replay.engine import ReplayEngine
from tvverify.validators.findings import Finding, ValidationReport
from tvverify.validators.structural import check_structure


class OfflineValidator:
    def __init__(
        self,
        adapter: WorkloadAdapter,
        engine: ReplayEngine,
        *,
        check_purity: bool = False,
        check_snapshots: bool = True,
    ):
        self.adapter = adapter
        self.engine = engine
        self.check_purity = check_purity
        # False when the snapshots are gone (e.g. a dump saved after the run)
        self.check_snapshots = check_snapshots

    async def check_node(
        self,
        g: GraphExport,
        node_id: str,
        snapshot_exists: Optional[Callable[[str], bool]] = None,
    ) -> List[Finding]:
        node = g.nodes[node_id]
        path = list(node.path)
        key = path[-1]
        stateful = self.adapter.is_stateful(key)
        out: List[Finding] = []

        if node.value is not None:  # C1 / C3: the cached value vs executing the call after a fresh replay
            fresh = await self.engine.leaf_output(path[:-1], key)
            criterion = "C3" if self.adapter.is_test_key(key) else "C1"
            cmp = self.adapter.comparator_for(key).compare(node.value, fresh, key=key)
            if fresh.startswith(FAILED_PREFIX):
                out.append(Finding(criterion, "replay_failed", f"fresh replay could not run the call: {fresh[:80]!r}", node_id, tuple(path)))
            elif not cmp.equal:
                out.append(Finding(criterion, "output_mismatch", f"cache serves {node.value[:40]!r} but fresh replay gives {fresh[:40]!r}", node_id, tuple(path)))
        if not stateful and self.check_purity:
            before, after = await self.engine.purity(path[:-1], key)
            if before and after and not diff_digests(before, after).equal_strict:
                out.append(Finding("A2", "stateless_call_mutated_state", f"{key[:40]} changed the state", node_id, tuple(path)))

        if not (self.check_snapshots and stateful and node.sandbox_id and (snapshot_exists is None or snapshot_exists(node.sandbox_id))):
            return out
        digest = await self.engine.state_digest(path)
        if digest is None:  # a sandbox without state_digest() is compared live through equals()
            if not await self.engine.matches_snapshot(path, node.sandbox_id):
                out.append(Finding("C2a", "snapshot_state_differs", f"snapshot {node.sandbox_id} differs from replay of {path}", node_id, tuple(path)))
        else:
            restored = await self.engine.restore_digest(node.sandbox_id)
            if restored is not None:
                diff = diff_digests(digest, restored)
                if not diff.equal_strict:
                    what = diff.reason or {c: e[:2] for c, e in diff.differing.items()}
                    out.append(Finding("C2a", "snapshot_state_differs", f"snapshot {node.sandbox_id} differs from replay of {path}: {what}", node_id, tuple(path)))
        # C3: the snapshot must also pass/fail the same tests
        from_snapshot = await self.engine.restore_leaf(node.sandbox_id, TEST_KEY)
        from_replay = await self.engine.test_output(path)
        cmp = self.adapter.comparator_for(TEST_KEY).compare(from_replay, from_snapshot, key=TEST_KEY)
        if from_snapshot.startswith(FAILED_PREFIX) or from_replay.startswith(FAILED_PREFIX):
            out.append(Finding("C3", "replay_failed", f"test could not run: snapshot {from_snapshot[:40]!r}, replay {from_replay[:40]!r}", node_id, tuple(path)))
        elif not cmp.equal:
            out.append(Finding("C3", "snapshot_test_differs", f"test from snapshot {from_snapshot[:30]!r} vs replay {from_replay[:30]!r}", node_id, tuple(path)))
        return out

    async def run(
        self,
        g: GraphExport,
        *,
        nodes: Optional[Iterable[str]] = None,
        snapshot_exists: Optional[Callable[[str], bool]] = None,
        replay: bool = True,
        quiescent: bool = True,
    ) -> ValidationReport:
        """`nodes` limits the replay to those nodes (default: all). Set
        `quiescent=False` for a dump taken while rollouts held leases."""
        report = ValidationReport()
        report.findings += check_structure(g, adapter=self.adapter, snapshot_exists=snapshot_exists, quiescent=quiescent)
        report.count("structural_checks")
        if replay:
            targets = [n for n in (nodes if nodes is not None else g.nodes) if n != g.root_id and n in g.nodes]
            for node_id in targets:
                try:
                    report.findings += await self.check_node(g, node_id, snapshot_exists)
                except Exception as error:  # one broken replay must not end the whole check
                    report.findings.append(Finding("C1", "replay_error", f"{type(error).__name__}: {str(error)[:200]}", node_id, tuple(g.nodes[node_id].path)))
                report.count("nodes_replayed")
        report.count("executions", self.engine.stats.executions)
        return report
