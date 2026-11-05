"""Regression tests for bugs B1-B3 (listed in the README).

Each test fails on the code before its fix (TVCache `main`) and is marked
`known_bug`. They run against a real Rust server
(fixture `server`) and the in-memory MockListSandbox.
"""

import asyncio

import pytest

from tvclient import TvcacheToolExecutor, TVCacheClient
from tvverify.adapters.mock_list import CommandToolCall, MockListSandbox


def run(coro):
    return asyncio.run(coro)


def make_executor(server, task_id: str, **kwargs) -> TvcacheToolExecutor:
    return TvcacheToolExecutor(
        MockListSandbox,
        CommandToolCall,
        task_id,
        port=server.port,
        adaptive=False,
        fixed_threshold_seconds=1000.0,  # never snapshot unless a test inserts one
        **kwargs,
    )


class FlakyOnceSandbox(MockListSandbox):
    """Reports an environment failure the first time `flaky` runs."""

    failed_once = False

    async def execute(self, tool_call):
        value = await super().execute(tool_call)
        if tool_call.serialize() == CommandToolCall("flaky").serialize():
            if not FlakyOnceSandbox.failed_once:
                FlakyOnceSandbox.failed_once = True
                return "Failed to execute: boom"
        return value


@pytest.fixture(autouse=True)
def reset_flaky():
    FlakyOnceSandbox.failed_once = False


@pytest.mark.known_bug("B1")
def test_b1_snapshot_at_the_full_path_is_not_reused():
    """A snapshot cached at the node of the current call already contains the
    call. Resuming from it used to run the call a second time on top of it."""
    c1 = CommandToolCall("setup")

    async def scenario(server):
        async with TVCacheClient(port=server.port) as client:
            donor = MockListSandbox(task_name="b1")
            await donor.start()
            await donor.execute(c1)
            snapshot = await donor.fork()
            # Node for [c1] has a snapshot but no value: a miss that finds an
            # LPM sandbox at depth == len(path).
            await client.insert("b1", [c1.serialize()], None, snapshot.get_id(), runtime=1)

        async with make_executor(server, "b1") as executor:
            await executor.lookup_or_execute([c1])
            state = MockListSandbox.state_of(executor.rollout_sandbox.get_id())
            assert state == [c1.serialize()], f"c1 was applied twice: {state}"

    from tvverify.testing.server import RustServer

    with RustServer() as server:
        run(scenario(server))


@pytest.mark.known_bug("B1")
def test_b1_live_sandbox_already_holding_the_current_call_is_not_reused():
    """After a failed (uncached) stateful call the live sandbox holds the call;
    re-issuing it must not execute it again on that same sandbox."""
    flaky = CommandToolCall("flaky")

    async def scenario(server):
        executor = TvcacheToolExecutor(
            FlakyOnceSandbox,
            CommandToolCall,
            "b1-live",
            port=server.port,
            adaptive=False,
            fixed_threshold_seconds=1000.0,
        )
        async with executor:
            first = await executor.lookup_or_execute([flaky])
            assert first.startswith("Failed to execute")
            second = await executor.lookup_or_execute([flaky])
            assert second == f"result:{flaky.serialize()}"
            state = MockListSandbox.state_of(executor.rollout_sandbox.get_id())
            assert state == [flaky.serialize()], f"flaky was applied twice: {state}"

    from tvverify.testing.server import RustServer

    with RustServer() as server:
        run(scenario(server))


@pytest.mark.known_bug("B3")
def test_b3_failed_stateful_call_does_not_orphan_deeper_inserts():
    flaky = CommandToolCall("flaky")
    after = CommandToolCall("after")

    async def scenario(server):
        executor = TvcacheToolExecutor(
            FlakyOnceSandbox,
            CommandToolCall,
            "b3",
            port=server.port,
            adaptive=False,
            fixed_threshold_seconds=1000.0,
        )
        async with executor:
            assert (await executor.lookup_or_execute([flaky])).startswith("Failed to execute")
            await executor.lookup_or_execute([flaky, after])
            assert executor.rejected_inserts == 0, "a deeper insert lost its parent"

        # Another rollout now gets the deeper node from the cache without executing.
        MockListSandbox.execute_log.clear()
        async with make_executor(server, "b3") as other:
            value = await other.lookup_or_execute([flaky, after])
        assert value == f"result:{after.serialize()}"
        assert MockListSandbox.execute_log == [], "deeper node should have been cached"

        # The failed call itself is still never served as a cached value.
        async with make_executor(server, "b3") as other:
            value = await other.lookup_or_execute([flaky])
        assert not value.startswith("Failed to execute")

    from tvverify.testing.server import RustServer

    with RustServer() as server:
        run(scenario(server))


@pytest.mark.known_bug("B2")
def test_b2_conflicting_value_cannot_overwrite_or_attach_a_snapshot():
    async def scenario(server):
        async with TVCacheClient(port=server.port) as client:
            path = ["x"]
            first = await client.insert("b2", path, "n=1", "snap-a", runtime=1)
            assert (first.value_inserted, first.sandbox_inserted) == (True, True)

            second = await client.insert("b2", path, "n=2", "snap-b", runtime=1)
            assert (second.value_inserted, second.sandbox_inserted) == (False, False)

            async with await client.find("b2", path) as found:
                assert found.value == "n=1"
                assert found.sandbox_id == "snap-a"

    from tvverify.testing.server import RustServer

    with RustServer() as server:
        run(scenario(server))
