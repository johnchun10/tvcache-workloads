"""Real tvcache-rust + SQLite sandbox; no Docker or database service needed."""

import asyncio
import json
from pathlib import Path

import pytest

from tvcache_workloads.cache_admin import release_task_snapshots
from tvcache_workloads.sql import (
    SQLSandboxConfig,
    SQLiteSandbox,
    load_trace_file,
    run_rollout,
)

pytestmark = pytest.mark.cache_integration

TRACE_PATH = Path(__file__).parents[3] / "sql" / "traces" / "smallbank.json"
TRACE = {
    rollout["name"]: rollout
    for rollout in load_trace_file(TRACE_PATH)["rollouts"]
}


def run(coroutine):
    return asyncio.run(coroutine)


def test_exact_replay_alternate_syntax_and_snapshot_branch(
    cache_server, cache_task_id, tmp_path
):
    host, port = cache_server
    sandbox_class = SQLiteSandbox.with_config(
        SQLSandboxConfig(database_dir=tmp_path, simulated_sql_latency_seconds=0)
    )

    async def scenario():
        try:
            first = await run_rollout(
                "first",
                TRACE["first-transfer"]["calls"],
                sandbox_class=sandbox_class,
                task_id=cache_task_id,
                cache_host=host,
                cache_port=port,
                snapshot_threshold_seconds=1e-6,
            )
            replay = await run_rollout(
                "replay",
                TRACE["exact-replay"]["calls"],
                sandbox_class=sandbox_class,
                task_id=cache_task_id,
                cache_host=host,
                cache_port=port,
                snapshot_threshold_seconds=1e-6,
            )
            equivalent = await run_rollout(
                "equivalent",
                TRACE["equivalent-sql-spelling"]["calls"],
                sandbox_class=sandbox_class,
                task_id=cache_task_id,
                cache_host=host,
                cache_port=port,
                snapshot_threshold_seconds=1e-6,
            )
            branch = await run_rollout(
                "branch",
                TRACE["divergent-transfer"]["calls"],
                sandbox_class=sandbox_class,
                task_id=cache_task_id,
                cache_host=host,
                cache_port=port,
                snapshot_threshold_seconds=1e-6,
            )
            return first, replay, equivalent, branch
        finally:
            await release_task_snapshots(cache_task_id, sandbox_class, host, port)

    first, replay, equivalent, branch = run(scenario())
    assert all(not step.cache_hit for step in first.steps)
    assert first.test_passed is True
    assert all(step.cache_hit for step in replay.steps)
    assert replay.test_cache_hit is True
    assert [step.observation for step in equivalent.steps] == [
        step.observation for step in first.steps
    ]
    assert equivalent.steps[0].cache_hit
    assert all(not step.cache_hit for step in equivalent.steps[1:])
    assert branch.steps[:3] and all(step.cache_hit for step in branch.steps[:3])
    assert [step.executions for step in branch.steps] == [0, 0, 0, 1, 1]
    assert any(event["op"] == "restore" for event in branch.sandbox_events)
    assert json.loads(branch.steps[-1].observation)["rows"] != json.loads(
        first.steps[-1].observation
    )["rows"]