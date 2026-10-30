"""End-to-end tier: REAL tvcache-rust, REAL container-manager-public, REAL Docker
containers for the existing Terminal-Bench task "hello-world". Nothing mocked.

Run: uv run pytest -m e2e -s
Needs both services running (bash/README.md) with ENABLE_CACHING=1 on the
manager, which must be on the cold-snapshots branch for the "cold" cases.
Takes a few minutes; creates and removes containers and images.
"""

import subprocess
from urllib.parse import urlparse

import pytest
from conftest import MANAGER_URL, _reachable

from tvcache_workloads.bash import (
    BashToolCall,
    SandboxConfig,
    SnapshotContractError,
    TerminalBenchSandbox,
    is_cold_handle,
    load_trace_file,
    run_rollout,
)
from tvcache_workloads.bash.cache_admin import release_task_snapshots

pytestmark = pytest.mark.e2e

TRACE = load_trace_file("bash/traces/hello-world.json")


@pytest.fixture(params=["cold", "parked"])
def manager(request):
    url = urlparse(MANAGER_URL)
    if not _reachable(url.hostname, url.port or 80):
        pytest.fail(f"container-manager-public is not reachable at {MANAGER_URL}")
    events = []
    sandbox = TerminalBenchSandbox.with_config(
        SandboxConfig(
            manager_url=MANAGER_URL, command_timeout_sec=60,
            snapshot_mode=request.param, on_event=events.append,
        )
    )
    return sandbox, events


def _docker_lines(*args):
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=True).stdout.split()


def images_for(handle_ids):
    """Docker images still present for cold snapshot handles (tb-snap:snap_<token>)."""
    tokens = [h.split("snap_", 1)[1] for h in handle_ids if is_cold_handle(h)]
    return [i for i in _docker_lines("images", "--format", "{{.Repository}}") if any(t in i for t in tokens)]


def containers_for(env_ids):
    """Docker containers still present for the given terminal ids (term_<uuid>)."""
    names = []
    for env_id in env_ids:
        uid = env_id.split("term_", 1)[-1]
        out = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={uid}", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        names.extend(out)
    return names


def test_snapshot_lifecycle_and_state_limits(manager, run):
    Sandbox, events = manager
    task = "hello-world::e2e-lifecycle"
    created = []  # underlying terminal ids, for the Docker leak check
    handles = []  # snapshot handles, for the image leak check
    sandboxes = []  # everything to stop, even if an assertion fails midway
    # Exact cmdline match: the task container already runs `sleep infinity`.
    count_bg = (
        "for p in /proc/[0-9]*; do [ \"$(tr '\\0' ' ' < $p/cmdline 2>/dev/null)\" = 'sleep 6123 ' ]"
        " && echo $p; done 2>/dev/null | wc -l"
    )

    async def sh(sandbox, command):
        return await sandbox.execute(BashToolCall(command))

    async def restore(handle_id):
        child = Sandbox(task_name=task, env_id=handle_id)
        await child.start()
        created.append(child.get_id())
        sandboxes.append(child)
        return child

    async def scenario():
        parent = Sandbox(task_name=task)
        await parent.start()
        created.append(parent.get_id())
        sandboxes.append(parent)
        await sh(parent, "mkdir -p /app/work && cd /app/work")
        await sh(parent, "export FOO=exported && BAR=unexported && echo base > /app/a.txt")
        await sh(parent, "(sleep 6123 &) ; true")
        assert await sh(parent, count_bg) == "1"

        running_before = len(_docker_lines("ps", "-q"))
        handle = await parent.fork()
        handles.append(handle.get_id())
        if is_cold_handle(handle.get_id()):
            # A cold snapshot is images only: nothing new is running.
            assert images_for(handles)
            assert len(_docker_lines("ps", "-q")) == running_before
        else:
            created.append(handle.get_id().split(":", 1)[1])
        sandboxes.append(handle)
        with pytest.raises(SnapshotContractError):
            await sh(handle, "ls")
        await sh(parent, "echo parent-only > /app/b.txt")

        first, second = await restore(handle.get_id()), await restore(handle.get_id())
        await sh(first, "echo first-only > /app/c.txt")
        third = await restore(handle.get_id())  # after siblings and parent mutated

        results = {
            "parent_ls": await sh(parent, "ls /app"),
            "parent_bar": await sh(parent, 'echo "[$BAR]"'),
            "first_ls": await sh(first, "ls /app"),
            "second_ls": await sh(second, "ls /app"),
            "third_ls": await sh(third, "ls /app"),
            "third_a": await sh(third, "cat /app/a.txt"),
            "third_pwd": await sh(third, "pwd"),
            "third_foo": await sh(third, 'echo "[$FOO]"'),
            "third_bar": await sh(third, 'echo "[$BAR]"'),
            "parent_bg": await sh(parent, count_bg),
            "third_bg": await sh(third, count_bg),
            "no_newline": await sh(third, "printf hi"),
            "multi_line": await sh(third, "printf 'l1\\nl2\\n'"),
            "blank_lines": await sh(third, "printf 'a\\n\\nb\\n'"),
            "exit_status": await sh(third, "false"),
            "stderr": await sh(third, "ls /nope"),
        }
        return results

    async def stop_all():
        for sandbox in sandboxes:
            try:
                await sandbox.stop()
            except Exception as error:
                print(f"cleanup: failed to stop {sandbox.get_id()}: {error}")

    async def scenario_with_cleanup():
        try:
            return await scenario()
        finally:
            await stop_all()

    images_before = set(_docker_lines("images", "--format", "{{.Repository}}"))
    results = run(scenario_with_cleanup())
    print("\nlifecycle observations:", results)
    print("lifecycle costs:", [(e["op"], round(e["seconds"], 2)) for e in events if e["op"] != "execute"])

    # Parent is unaffected by being forked and keeps its own later writes.
    assert results["parent_ls"].split() == ["a.txt", "b.txt", "work"]
    assert results["parent_bar"] == "[unexported]"
    # Restores are independent of the parent and of each other.
    assert results["first_ls"].split() == ["a.txt", "c.txt", "work"]
    assert results["second_ls"].split() == ["a.txt", "work"]
    # The parked snapshot did not change after siblings and parent mutated.
    assert results["third_ls"].split() == ["a.txt", "work"]
    assert results["third_a"] == "base"
    # Preserved: filesystem, cwd, exported variables (manager tmux-resurrect replay).
    assert results["third_pwd"] == "/app/work"
    assert results["third_foo"] == "[exported]"
    # NOT preserved (documented limits): unexported variables, background processes.
    assert results["third_bar"] == "[]"
    assert results["parent_bg"] == "1"
    assert results["third_bg"] == "0"
    # Observation format.
    assert results["no_newline"] == "hi"
    assert results["multi_line"] == "l1\nl2"
    assert results["blank_lines"] == "a\nb"  # the manager drops blank lines
    assert results["exit_status"] == ""  # exit status is not observable
    assert "No such file or directory" in results["stderr"]  # stderr is captured

    assert containers_for(created) == []
    assert images_for(handles) == []
    # No snapshot images or per-restore tags left behind.
    assert set(_docker_lines("images", "--format", "{{.Repository}}")) - images_before == set()


def test_hello_world_trace_through_tvcache(manager, run, cache_server, cache_task_id):
    Sandbox, _ = manager
    host, port = cache_server
    records = {}

    released = []

    async def scenario():
        try:
            for rollout in TRACE["rollouts"]:
                records[rollout["name"]] = await run_rollout(
                    rollout["name"], rollout["commands"], sandbox_class=Sandbox,
                    task_id=cache_task_id, cache_host=host, cache_port=port,
                    snapshot_threshold_seconds=1e-6,
                )
        finally:
            # Not `return` here: that would swallow a rollout's exception.
            released.extend(await release_task_snapshots(cache_task_id, Sandbox, host, port))

    images_before = set(_docker_lines("images", "--format", "{{.Repository}}"))
    run(scenario())

    for name, record in records.items():
        print(f"\n{name}: {record.seconds:.1f}s test={record.test_result} "
              f"hits={[s.cache_hit for s in record.steps]} "
              f"ops={[e['op'] for e in record.sandbox_events]}")
        for step in record.steps:
            print(f"   {step.seconds:5.2f}s {'HIT ' if step.cache_hit else 'EXEC'} "
                  f"{step.command!r} -> {step.observation!r}")

    for rollout in TRACE["rollouts"]:
        assert records[rollout["name"]].test_passed is rollout["expect_test_passed"], rollout["name"]

    solve, again = records["solve"], records["solve-again"]
    assert not any(s.cache_hit for s in solve.steps)
    assert all(s.cache_hit for s in again.steps) and again.test_cache_hit
    assert [s.observation for s in again.steps] == [s.observation for s in solve.steps]
    assert again.test_result == solve.test_result
    assert solve.steps[2].observation == "Hello, world!"

    handles = set()
    for name in ("wrong-content", "inspect-prefix", "equivalent-rewrite"):
        record = records[name]
        ops = [e["op"] for e in record.sandbox_events]
        assert ops.count("restore") == 1 and "start" not in ops, name
        assert record.steps[0].cache_hit and record.steps[1].executions == 1, name
        handles |= {e["handle"] for e in record.sandbox_events if e["op"] == "restore"}
    assert len(handles) == 1  # all three resumed from the same step-0 snapshot

    inspect = records["inspect-prefix"]
    assert inspect.steps[1].observation == "/app/work"
    assert inspect.steps[2].observation == "work"  # snapshot predates hello.txt

    # 3 snapshots from "solve" + 2 from each of the three divergent rollouts.
    assert len(released) == 9
    every_env = {e.get("env_id") for r in records.values() for e in r.sandbox_events} | {
        r.split(":", 1)[1] for r in released
    }
    assert containers_for({e for e in every_env if e and e.startswith("term_")}) == []
    assert images_for(released) == []
    assert set(_docker_lines("images", "--format", "{{.Repository}}")) - images_before == set()
