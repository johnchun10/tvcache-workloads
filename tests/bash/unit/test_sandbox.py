"""Unit tests (no services): TerminalBenchSandbox against the FakeManager TEST DOUBLE.

These check the adapter's translation and lifecycle rules. They say nothing
about real Docker behavior; see tests/bash/e2e for that.
"""

import asyncio
import json

import pytest
from fake_manager import FakeManager
from tvclient import TestToolCall as SentinelTestCall  # renamed so pytest does not collect it

from tvcache_workloads.bash import (
    COLD_PREFIX,
    PARKED_PREFIX,
    BashToolCall,
    CommandTimeout,
    ManagerError,
    SandboxConfig,
    SnapshotContractError,
    TerminalBenchSandbox,
)


@pytest.fixture
def fake():
    return FakeManager()


@pytest.fixture
def events():
    return []


# Lifecycle tests run once per snapshot mode.
@pytest.fixture(params=["cold", "parked"])
def mode(request):
    return request.param


@pytest.fixture
def Sandbox(fake, events, mode):
    return TerminalBenchSandbox.with_config(
        SandboxConfig(
            manager_url="http://fake", command_timeout_sec=5, snapshot_mode=mode,
            transport=fake.transport(), on_event=events.append,
        )
    )


def run(coro):
    return asyncio.run(coro)


async def started(Sandbox, task="hello-world"):
    sandbox = Sandbox(task_name=task)
    await sandbox.start()
    return sandbox


def test_with_config_does_not_change_the_base_class(Sandbox):
    assert TerminalBenchSandbox.config.transport is None
    assert Sandbox.config.manager_url == "http://fake"
    assert issubclass(Sandbox, TerminalBenchSandbox)


def test_start_uses_terminal_bench_task_without_cache_namespace(Sandbox, fake):
    sandbox = run(started(Sandbox, "hello-world::exp7"))
    assert fake.calls("/start") == [{"task_name": "hello-world"}]
    assert sandbox.get_id() in fake.terminals


def test_execute_returns_only_the_commands_output(Sandbox, fake):
    async def scenario():
        sandbox = await started(Sandbox)
        await sandbox.execute(BashToolCall("echo 'Hello, world!' > /app/hello.txt"))
        return await sandbox.execute(BashToolCall("cat /app/hello.txt"))

    assert run(scenario()) == "Hello, world!"


def test_user_command_is_sent_unchanged_between_marker_commands(Sandbox, fake):
    async def scenario():
        sandbox = await started(Sandbox)
        await sandbox.execute(BashToolCall("pwd"))

    run(scenario())
    commands = [c["command"] for c in fake.calls("/execute")]
    assert len(commands) == 3
    assert commands[0].startswith("printf '__TVB%s_%s__\\n' BEGIN ")
    assert commands[1] == "pwd"
    assert commands[2].startswith("printf '__TVB%s_%s__\\n' END ")
    assert all(c["blocking"] is True and c["timeout"] == 5 for c in fake.calls("/execute"))


def test_output_without_trailing_newline(Sandbox):
    async def scenario():
        sandbox = await started(Sandbox)
        return await sandbox.execute(BashToolCall("echo-n partial"))

    assert run(scenario()) == "partial"


def test_command_timeout_raises_so_tvclient_discards_the_sandbox(Sandbox):
    async def scenario():
        sandbox = await started(Sandbox)
        await sandbox.execute(BashToolCall("sleep 999"))

    with pytest.raises(CommandTimeout):
        run(scenario())


def test_unextractable_output_is_reported_as_failed_execution(Sandbox):
    async def scenario():
        sandbox = await started(Sandbox)
        return await sandbox.execute(BashToolCall("seq 3000"))

    result = run(scenario())
    assert result.startswith("Failed to execute")  # never cached by tvclient


def test_fork_returns_a_handle_of_the_configured_kind(Sandbox, fake, mode):
    async def scenario():
        parent = await started(Sandbox)
        handle = await parent.fork()
        return parent, handle

    parent, handle = run(scenario())
    if mode == "cold":
        assert handle.get_id().startswith(COLD_PREFIX)
        assert handle.get_id()[len(COLD_PREFIX):] in fake.snapshots
        assert len(fake.terminals) == 1  # a cold snapshot runs nothing
        assert fake.calls("/fork") == []
    else:
        assert handle.get_id().startswith(PARKED_PREFIX)
        assert handle.get_id()[len(PARKED_PREFIX):] in fake.terminals
        assert fake.calls("/snapshot") == []
    assert parent.get_id() in fake.terminals


def test_snapshot_handle_refuses_execute_and_fork(Sandbox):
    async def scenario():
        handle = await (await started(Sandbox)).fork()
        with pytest.raises(SnapshotContractError):
            await handle.execute(BashToolCall("ls"))
        with pytest.raises(SnapshotContractError):
            await handle.fork()

    run(scenario())


def test_start_refuses_to_restore_a_live_id(Sandbox):
    async def scenario():
        live = await started(Sandbox)
        with pytest.raises(SnapshotContractError, match="only snapshot handles"):
            await Sandbox(task_name="hello-world", env_id=live.get_id()).start()

    run(scenario())


def test_unstarted_sandbox_refuses_execute(Sandbox):
    with pytest.raises(SnapshotContractError, match="not started"):
        run(Sandbox(task_name="hello-world").execute(BashToolCall("ls")))


def test_restore_is_repeatable_and_children_are_independent(Sandbox, fake):
    async def scenario():
        parent = await started(Sandbox)
        await parent.execute(BashToolCall("echo base > /app/a"))
        handle_id = (await parent.fork()).get_id()
        await parent.execute(BashToolCall("echo later > /app/b"))

        first = Sandbox(task_name="hello-world", env_id=handle_id)
        await first.start()
        second = Sandbox(task_name="hello-world", env_id=handle_id)
        await second.start()
        await first.execute(BashToolCall("echo mine > /app/c"))
        return (
            first.get_id(), second.get_id(),
            await first.execute(BashToolCall("cat /app/b")),
            await second.execute(BashToolCall("cat /app/a")),
            await second.execute(BashToolCall("cat /app/c")),
        )

    first_id, second_id, first_b, second_a, second_c = run(scenario())
    assert first_id != second_id
    assert "No such file" in first_b      # post-fork parent write not in snapshot
    assert second_a == "base"             # snapshot state preserved
    assert "No such file" in second_c     # sibling restores are isolated


def test_stop_deletes_snapshots_and_stops_live_ids(Sandbox, fake, mode):
    async def scenario():
        live = await started(Sandbox)
        handle = await live.fork()
        await handle.stop()
        await live.stop()

    run(scenario())
    assert fake.terminals == {} and fake.snapshots == {}
    assert len(fake.calls("/delete_snapshot")) == (1 if mode == "cold" else 0)
    assert all(not c["id"].startswith(("tb-", "snap")) for c in fake.calls("/stop"))


def test_handles_are_dispatched_by_prefix_not_by_current_mode(fake, events):
    def configured(mode):
        return TerminalBenchSandbox.with_config(
            SandboxConfig(manager_url="http://fake", snapshot_mode=mode, transport=fake.transport())
        )

    async def scenario():
        cold_handle = await (await started(configured("cold"))).fork()
        parked_handle = await (await started(configured("parked"))).fork()
        # A sandbox class in the other mode still restores and deletes each correctly.
        for handle, other in ((cold_handle, "parked"), (parked_handle, "cold")):
            restored = configured(other)(task_name="hello-world", env_id=handle.get_id())
            await restored.start()
            await restored.stop()
            await configured(other)(task_name="hello-world", env_id=handle.get_id()).stop()

    run(scenario())
    assert len(fake.calls("/restore")) == 1 and len(fake.calls("/delete_snapshot")) == 1
    assert fake.snapshots == {}


def test_cold_mode_explains_a_manager_without_snapshot_endpoints(events):
    fake = FakeManager(cold_snapshots=False)
    Sandbox = TerminalBenchSandbox.with_config(
        SandboxConfig(manager_url="http://fake", snapshot_mode="cold", transport=fake.transport())
    )

    async def scenario():
        await (await started(Sandbox)).fork()

    with pytest.raises(ManagerError, match="snapshot_mode='parked'"):
        run(scenario())


def test_cold_snapshot_of_unknown_terminal_is_a_plain_404(fake):
    Sandbox = TerminalBenchSandbox.with_config(
        SandboxConfig(manager_url="http://fake", snapshot_mode="cold", transport=fake.transport())
    )
    with pytest.raises(ManagerError, match="Terminal not found"):
        run(Sandbox(task_name="hello-world", env_id="term_gone").fork())


def test_invalid_snapshot_mode_is_rejected():
    with pytest.raises(ValueError):
        SandboxConfig(snapshot_mode="hot")


def test_stop_without_env_is_a_noop(Sandbox, fake):
    run(Sandbox(task_name="hello-world").stop())
    assert fake.calls("/stop") == []


def test_stop_of_unknown_id_raises(Sandbox):
    with pytest.raises(ManagerError):
        run(Sandbox(task_name="hello-world", env_id="term_gone").stop())


def test_test_tool_call_routes_to_test_endpoint(Sandbox, fake):
    async def scenario():
        sandbox = await started(Sandbox)
        before = await sandbox.execute(SentinelTestCall())
        await sandbox.execute(BashToolCall("echo 'Hello, world!' > /app/hello.txt"))
        after = await sandbox.execute(SentinelTestCall())
        return before, after

    before, after = run(scenario())
    assert json.loads(before) == {"passed": False}
    assert json.loads(after) == {"passed": True}
    assert len(fake.calls("/test")) == 2


def test_broken_test_run_is_not_a_cacheable_result(Sandbox, fake):
    fake.test_error = "Error while setting up environment for tests"

    async def scenario():
        return await (await started(Sandbox)).execute(SentinelTestCall())

    assert run(scenario()).startswith("Failed to execute tests")


def test_rejects_foreign_tool_calls(Sandbox):
    from tvclient import ToolCall

    class Other(ToolCall):
        def to_dict(self):
            return {}

    async def scenario():
        await (await started(Sandbox)).execute(Other())

    with pytest.raises(TypeError):
        run(scenario())


def test_events_report_lifecycle_costs(Sandbox, events):
    async def scenario():
        live = await started(Sandbox)
        handle = await live.fork()
        restored = Sandbox(task_name="hello-world", env_id=handle.get_id())
        await restored.start()
        await restored.execute(BashToolCall("pwd"))
        await restored.stop()

    run(scenario())
    assert [e["op"] for e in events] == ["start", "snapshot", "restore", "execute", "stop"]
    assert all(e["seconds"] >= 0 for e in events)
