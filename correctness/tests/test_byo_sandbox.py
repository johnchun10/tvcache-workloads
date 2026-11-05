"""Bring-your-own-sandbox: the contract checks, the equals() fallback, and the
`check` command, tested against deliberately broken variants of the toy."""

import asyncio
import importlib.util
import random
import sys
from pathlib import Path

import pytest

from tvverify.adapters.mock_fs import ToyCall, ToyFsSandbox
from tvverify.adapters.toyfs_adapter import ToyFsAdapter
from tvverify.digest.schema import StateDigest
from tvverify.replay.engine import ReplayEngine
from tvverify.testing.sandbox_contract import run_contract
from tvverify.validators.offline import OfflineValidator

TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "my_workload.py"


@pytest.fixture(autouse=True)
def reset_toyfs():
    with ToyFsSandbox.isolated():
        yield


def adapter_with(sandbox_cls):
    class Adapter(ToyFsAdapter):
        sandbox_class = sandbox_cls

        # Same as WorkloadAdapter.probe_state: undoes ToyFsAdapter's override,
        # which digests the toy state directly and so ignores the subclass.
        async def probe_state(self, sandbox, policy):
            raw = await sandbox.state_digest()
            return StateDigest.from_dict(raw) if raw is not None else None

    return Adapter()


def failed(results):
    return {r.name for r in results if not r.ok}


def run(coro):
    return asyncio.run(coro)


def test_toy_passes_the_contract():
    assert failed(run(run_contract(ToyFsAdapter()))) == set()


def test_equals_that_is_always_true_is_caught():
    class Blind(ToyFsSandbox):
        async def equals(self, other):
            return True

    assert "equals() notices a stateful call" in failed(run(run_contract(adapter_with(Blind))))


def test_fork_that_mutates_the_source_is_caught():
    class Dirty(ToyFsSandbox):
        async def fork(self):
            handle = await super().fork()
            self.state.files["/junk"] = "x"
            return handle

    assert "fork() does not change the source" in failed(run(run_contract(adapter_with(Dirty))))


def test_restore_that_drops_state_is_caught():
    class Lossy(ToyFsSandbox):
        async def start(self):
            restoring = self.env_id in ToyFsSandbox.snapshots
            await super().start()
            if restoring:
                self.state.files.clear()

    assert "a restored snapshot equals the source at fork time" in failed(
        run(run_contract(adapter_with(Lossy)))
    )


def test_unimplemented_sandbox_is_reported_not_crashed():
    spec = importlib.util.spec_from_file_location("my_workload_template", TEMPLATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    results = run(run_contract(module.MyAdapter()))
    assert results and not all(r.ok for r in results)
    assert any("not implemented" in r.hint for r in results)


class EqualsOnly(ToyFsSandbox):
    """No state_digest: only equals(), comparing live states."""

    async def state_digest(self):
        return None

    async def equals(self, other):
        a, b = self.state, other.state
        return (a.files, a.env, a.cwd) == (b.files, b.env, b.cwd)


def test_equals_only_sandbox_passes_contract():
    assert failed(run(run_contract(adapter_with(EqualsOnly)))) == set()


def test_equals_fallback_catches_lossy_snapshot_state():
    """With no digest the offline validator must not skip C2a: a snapshot that
    loses a file is still found by comparing live sandboxes."""

    class DroppingSnapshots(EqualsOnly):
        async def fork(self):
            handle = await super().fork()
            ToyFsSandbox.snapshots[handle.get_id()].files.clear()
            return handle

    from tvclient import TVCacheClient
    from tvverify.model.graph import from_dumpall
    from tvverify.testing.exact_vs_nocache import run_rollouts_vs_nocache
    from tvverify.testing.server import RustServer

    adapter = adapter_with(DroppingSnapshots)

    async def go():
        with RustServer() as server:
            rollouts = adapter.sample_rollouts("t", random.Random(1), 6)
            await run_rollouts_vs_nocache(server.port, adapter, "t", rollouts)
            async with TVCacheClient(port=server.port) as client:
                graph = from_dumpall(await client.dump_all(), adapter)["t"]
        engine = ReplayEngine(adapter, "validator", memoize=False)
        return await OfflineValidator(adapter, engine).run(graph)

    report = run(go())
    assert "C2a" in report.criteria_fired


def test_check_command_passes_on_the_toy(capsys):
    from tvverify.selftest import main

    assert main(["check", "--adapter", "tvverify.adapters.toyfs_adapter:ToyFsAdapter"]) == 0
    out = capsys.readouterr().out
    assert "contract: 11/11" in out and "replay toyfs-task: PASS" in out


def test_check_command_fails_on_the_unfilled_template(capsys):
    from tvverify.selftest import main

    sys.path.insert(0, str(TEMPLATE.parent))
    try:
        assert main(["check", "--adapter", "my_workload:MyAdapter", "--suite", "contract"]) == 1
    finally:
        sys.path.remove(str(TEMPLATE.parent))
    assert "not implemented" in capsys.readouterr().out


def test_nondeterministic_tool_is_noise_not_a_bug(capsys):
    """`stamp` writes a fresh random token each run, so cached and replayed
    outputs differ; two fresh replays also differ, so it is reported as noise."""
    from tvverify.selftest import main

    class NoisyAdapter(ToyFsAdapter):
        def sample_rollouts(self, task_id, rng, count):
            return [[ToyCall("stamp", "f"), ToyCall("read", "f")] for _ in range(count)]

    sys.modules[__name__].NoisyAdapter = NoisyAdapter
    status = main(["check", "--adapter", f"{__name__}:NoisyAdapter", "--suite", "replay", "--replays", "3"])
    out = capsys.readouterr().out
    assert status == 0 and "BUG" not in out
    assert "0 noise" not in out and "noise" in out


# ---- regressions for the fixes after review --------------------------------


def test_contract_boots_sandboxes_for_a_real_task():
    """Workloads like Terminal-Bench pick the environment from the task name."""
    known = {t.task_id for t in ToyFsAdapter().tasks()}

    class StrictTask(ToyFsSandbox):
        async def start(self):
            assert self.task_name in known, self.task_name
            await super().start()

    assert failed(run(run_contract(adapter_with(StrictTask)))) == set()


def test_contract_leaves_no_sandboxes_or_snapshots_behind():
    run(run_contract(ToyFsAdapter()))
    assert ToyFsSandbox.live == {} and ToyFsSandbox.snapshots == {}


def test_check_deletes_the_snapshots_it_cached():
    from tvverify.selftest import main

    assert main(["check", "--adapter", "tvverify.adapters.toyfs_adapter:ToyFsAdapter"]) == 0
    assert ToyFsSandbox.live == {} and ToyFsSandbox.snapshots == {}


def test_engine_stops_the_first_sandbox_when_the_second_fails_to_start():
    class SecondStartFails(ToyFsSandbox):
        starts = 0

        async def start(self):
            SecondStartFails.starts += 1
            if SecondStartFails.starts == 2:
                raise RuntimeError("manager down")
            await super().start()

    engine = ReplayEngine(adapter_with(SecondStartFails), "toyfs-task", memoize=False)
    with pytest.raises(RuntimeError):
        run(engine.same_state([], []))
    assert ToyFsSandbox.live == {}


def test_engine_stops_digest_replays_once_it_learns_there_is_no_digest():
    engine = ReplayEngine(adapter_with(EqualsOnly), "toyfs-task", memoize=False)
    write = ToyCall("write", "a", "x").serialize()
    for _ in range(3):
        assert run(engine.state_digest([write])) is None
    assert engine.stats.sandboxes_started == 1


def test_adapter_without_tasks_is_reported_not_crashed():
    class NoTasks(ToyFsAdapter):
        def tasks(self):
            return []

    results = run(run_contract(NoTasks()))
    assert len(results) == 1 and not results[0].ok and "no tasks" in results[0].hint
