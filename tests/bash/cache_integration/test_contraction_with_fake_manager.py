"""Cache-integration tier for graph contraction: a REAL tvcache-rust server and
tvclient's real Contractor, with the FakeManager TEST DOUBLE in place of
Docker. Skipped on a tvclient without contraction.

Checks the whole chain a contraction run depends on: commands are embedded on
insert, the server pairs the similar ones, the coordinator's job rebuilds both
sandboxes from cache keys and compares their state, the equivalent pair is
merged, and the dump of the merged tree still reads (tvverify, snapshot
cleanup). Says nothing about real container state (see state_check.py, which
runs against real containers at the start of every contraction run).

Run: uv run pytest -m cache_integration   (needs the server from the
contraction-plus-fixes branch of tvcache-rust)
"""

import asyncio
import json

import pytest
from fake_manager import FakeManager

pytest.importorskip("tvclient.contraction")

from tvclient import Contractor, TVCacheClient, TvcacheToolExecutor, set_embedder  # noqa: E402

from tvcache_workloads.bash import BashToolCall, SandboxConfig, TerminalBenchSandbox  # noqa: E402
from tvcache_workloads.bash.cache_admin import node_children, release_task_snapshots  # noqa: E402
from tvcache_workloads.bash.contraction import Coordinator, EventLog, instrument, summarize  # noqa: E402

pytestmark = pytest.mark.cache_integration


class KeywordEmbedder:
    """Commands that write "hi" are similar to each other and to nothing else."""

    def encode(self, text):
        return [1.0, 0.0] if "hi" in text else [0.0, 1.0]


def test_equivalent_commands_are_merged_and_the_merged_tree_still_reads(cache_server, cache_task_id, tmp_path):
    host, port = cache_server
    fake = FakeManager()
    Sandbox = TerminalBenchSandbox.with_config(
        SandboxConfig(manager_url="http://fake", command_timeout_sec=5, transport=fake.transport())
    )
    set_embedder(KeywordEmbedder())
    events = EventLog(tmp_path / "events.jsonl")

    async def rollout(commands):
        executor = TvcacheToolExecutor(
            Sandbox, BashToolCall, cache_task_id, host=host, port=port,
            adaptive=False, fixed_threshold_seconds=1e-6, embed_tool_calls=True,
        )
        history, outputs = [], []
        async with executor:
            for command in commands:
                history.append(BashToolCall(command))
                outputs.append(await executor.lookup_or_execute(history))
        return outputs, executor.total_executions

    async def dump():
        async with TVCacheClient(host, port) as client:
            return (await client.dump_all())[cache_task_id]

    async def go():
        await rollout(["echo hi > f", "cat f"])
        await rollout(["echo hi > /app/f", "cat f"])  # another command, the same state
        await rollout(["echo other > g"])
        before = await dump()

        undo = instrument(events)
        coordinator = Coordinator(
            Contractor(Sandbox, BashToolCall, 0.9, 4, host=host, port=port, timeout=10), events, grace_seconds=0
        )
        try:
            hold = await coordinator.acquire(cache_task_id)  # as a rollout would
            await coordinator.release(cache_task_id, hold)
            for _ in range(200):  # the job ends on its own once no deeper nodes are left
                if '"job_end"' in (events.path.read_text() if events.path.exists() else ""):
                    break
                await asyncio.sleep(0.1)
        finally:
            await coordinator.close()
            undo()
        after = await dump()
        again, executions = await rollout(["echo hi > /app/f", "cat f"])
        released = await release_task_snapshots(cache_task_id, Sandbox, host=host, port=port)
        return before, after, again, executions, released

    before, after, again, executions, released = asyncio.run(go())

    def call_sets(task):
        return sorted(sorted(calls) for calls, _ in node_children(task["root"]))

    hi, hi_abs, other = (BashToolCall(c).serialize() for c in ("echo hi > f", "echo hi > /app/f", "echo other > g"))
    assert call_sets(before) == sorted([[hi], [hi_abs], [other]])
    assert call_sets(after) == sorted([sorted([hi, hi_abs]), [other]])  # the equivalent pair is one node now
    assert after["number_of_nodes"] < before["number_of_nodes"]
    for calls, _ in node_children(after["root"]):
        assert all(entry["embedding"] for entry in calls.values())  # stored on insert

    totals = summarize(events.path)["totals"]
    assert totals["similar_pairs_listed"] >= 1 and totals["comparisons"] >= 1
    assert totals["merged_classes"] == 1 and totals["jobs"] == 1

    assert again == ["", "hi"] and executions == 0  # served through the merged node
    assert fake.terminals == {}  # contraction stopped every sandbox it started
    assert released and fake.snapshots == {}  # and cleanup finds every snapshot in the merged tree

    tvverify = pytest.importorskip("tvverify.model.graph")
    from tvverify.validators.structural import check_structure

    graph = tvverify.from_dumpall({cache_task_id: json.loads(json.dumps(after))})[cache_task_id]
    assert [n.key for n in graph.nodes.values() if n.alias_of] == [max(hi, hi_abs)]
    assert check_structure(graph) == []
