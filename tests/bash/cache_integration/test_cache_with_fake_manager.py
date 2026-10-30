"""Cache-integration tier: REAL tvcache-rust server + tvclient executor, with
the FakeManager TEST DOUBLE in place of Docker.

Checks cache hits, resume-from-snapshot, repeated restores, non-caching of
failures, and snapshot cleanup as tvclient actually drives the adapter. Says
nothing about real container behavior (see tests/bash/e2e).

Run: uv run pytest -m cache_integration   (needs `cargo run --release` in ../tvcache-rust)
"""

import json

import pytest
from fake_manager import FakeManager

from tvcache_workloads.bash import (
    COLD_PREFIX,
    PARKED_PREFIX,
    BashToolCall,
    CommandTimeout,
    SandboxConfig,
    TerminalBenchSandbox,
    load_trace_file,
    run_rollout,
)
from tvcache_workloads.bash.cache_admin import release_task_snapshots

pytestmark = pytest.mark.cache_integration

TRACE = {r["name"]: r for r in load_trace_file("bash/traces/hello-world.json")["rollouts"]}
EVERY_STEP = 1e-6  # fixed snapshot threshold: snapshot after each stateful call


@pytest.fixture
def fake():
    return FakeManager()


@pytest.fixture(params=["cold", "parked"])
def mode(request):
    return request.param


@pytest.fixture
def Sandbox(fake, mode):
    return TerminalBenchSandbox.with_config(
        SandboxConfig(
            manager_url="http://fake", command_timeout_sec=5, snapshot_mode=mode,
            transport=fake.transport(),
        )
    )


def saved_snapshots(fake, mode):
    """Snapshots the fake manager holds: images (cold) or parked terminals."""
    return list(fake.snapshots) if mode == "cold" else list(fake.terminals)


@pytest.fixture
def rollout(Sandbox, cache_server, cache_task_id):
    host, port = cache_server

    async def _run(name, commands=None, **kwargs):
        return await run_rollout(
            name, commands if commands is not None else TRACE[name]["commands"],
            sandbox_class=Sandbox, task_id=cache_task_id,
            cache_host=host, cache_port=port, **kwargs,
        )

    return _run


def test_identical_rollout_is_served_entirely_from_cache(rollout, fake, run):
    first = run(rollout("solve", snapshot_threshold_seconds=EVERY_STEP))
    executes_after_first = len(fake.calls("/execute"))
    second = run(rollout("solve-again", snapshot_threshold_seconds=EVERY_STEP))

    assert [s.cache_hit for s in first.steps] == [False, False, False]
    assert first.test_passed is True and first.test_cache_hit is False
    assert [s.cache_hit for s in second.steps] == [True, True, True]
    assert second.test_cache_hit is True
    assert [s.observation for s in second.steps] == [s.observation for s in first.steps]
    assert second.test_result == first.test_result
    assert len(fake.calls("/execute")) == executes_after_first  # nothing ran
    assert second.sandbox_events == []


def test_divergent_suffixes_restore_the_same_snapshot_independently(rollout, fake, run, mode):
    run(rollout("solve", snapshot_threshold_seconds=EVERY_STEP))
    assert len(saved_snapshots(fake, mode)) == 3  # one snapshot per stateful step
    if mode == "cold":
        assert fake.terminals == {}  # nothing left running between rollouts

    wrong = run(rollout("wrong-content", snapshot_threshold_seconds=EVERY_STEP))
    inspect = run(rollout("inspect-prefix", snapshot_threshold_seconds=EVERY_STEP))

    for record in (wrong, inspect):
        ops = [e["op"] for e in record.sandbox_events]
        assert ops.count("restore") == 1 and ops.count("start") == 0
        assert record.steps[0].cache_hit and not record.steps[1].cache_hit
        # Resumed at depth 1, so only the new command ran (no replay).
        assert record.steps[1].executions == 1

    restored_from = {e["handle"] for r in (wrong, inspect) for e in r.sandbox_events if e["op"] == "restore"}
    assert len(restored_from) == 1  # both resumed from the same step-0 snapshot
    assert wrong.test_passed is False
    assert inspect.steps[1].observation == "/app/work"  # cwd carried by the snapshot
    # The snapshot predates every hello.txt write, so the file is absent.
    assert inspect.steps[2].observation == "work"


def test_failed_extraction_is_not_cached(rollout, fake, run):
    commands = ["echo start", "seq 3000"]
    first = run(rollout("big", commands, run_test=False))
    second = run(rollout("big", commands, run_test=False))
    assert first.steps[1].observation.startswith("Failed to execute")
    assert second.steps[0].cache_hit is True
    assert second.steps[1].cache_hit is False  # re-executed, never served from cache


def test_command_timeout_propagates_and_live_sandbox_is_discarded(rollout, fake, run):
    with pytest.raises(CommandTimeout):
        run(rollout("hang", ["echo before", "sleep 999"], run_test=False))
    # The executor stops the sandbox whose shell may still be busy; with the
    # adaptive policy and instant fake commands no snapshot was taken.
    assert fake.terminals == {}
    assert len(fake.calls("/stop")) == 1


def test_release_task_snapshots_deletes_every_snapshot(
    rollout, fake, run, cache_server, cache_task_id, Sandbox, mode
):
    run(rollout("solve", snapshot_threshold_seconds=EVERY_STEP))
    run(rollout("wrong-content", snapshot_threshold_seconds=EVERY_STEP))
    assert len(saved_snapshots(fake, mode)) == 5  # 3 + 2 snapshots, no live sandboxes

    released = run(release_task_snapshots(cache_task_id, Sandbox, *cache_server))

    prefix = COLD_PREFIX if mode == "cold" else PARKED_PREFIX
    assert len(released) == 5 and all(r.startswith(prefix) for r in released)
    assert fake.terminals == {} and fake.snapshots == {}


def test_test_results_are_json(rollout, run):
    record = run(rollout("solve"))
    assert json.loads(record.test_result) == {"passed": True}


def test_embedding_switch_stores_one_vector_per_executed_command(
    rollout, run, cache_server, cache_task_id
):
    tvclient = pytest.importorskip("tvclient")
    if not hasattr(tvclient, "set_embedder"):
        pytest.skip("installed tvclient has no embedding support (tvclient PR #2 not merged)")
    from tvclient import Embedder, TVCacheClient, set_embedder

    class StubEmbedder(Embedder):
        def encode(self, text):
            return [float(len(text)), 1.0]

    set_embedder(StubEmbedder())
    record = run(rollout("solve", embed_tool_calls=True))
    assert record.test_passed is True

    async def stored():
        async with TVCacheClient(*cache_server) as client:
            stack, found = [(await client.dump_all())[cache_task_id]["root"]], []
            while stack:
                for key, child in stack.pop()["children"].items():
                    found.append((key, child.get("embedding")))
                    stack.append(child)
            return found

    embeddings = dict(run(stored()))
    commands = TRACE["solve"]["commands"]
    for command in commands:
        key = BashToolCall(command).serialize()
        assert embeddings[key] == [float(len(key)), 1.0]
    # The test result is stateless, so tvclient does not embed it.
    assert [v for k, v in embeddings.items() if "TESTING" in k] == [None]
