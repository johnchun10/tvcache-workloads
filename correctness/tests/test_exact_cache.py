"""Phase 1 suite: exact TVCache must be correct on deterministic tools."""

import asyncio

import pytest

from tvverify.adapters.mock_fs import ToyCall, ToyFsSandbox
from tvverify.adapters.toyfs_adapter import ToyFsAdapter
from tvverify.testing.exact_vs_nocache import (
    _one_rollout,
    run_exact_vs_nocache,
    run_value_snapshot_coherence,
)
from tvverify.testing.ref_vs_server import run_ref_vs_server


@pytest.fixture(autouse=True)
def reset_toyfs():
    with ToyFsSandbox.isolated():
        yield


@pytest.mark.known_bug("B2")
def test_p1_reference_model_matches_server(server):
    for seed in range(40):
        problems = asyncio.run(run_ref_vs_server(server.port, seed))
        assert not problems, problems[:5]


@pytest.mark.parametrize("concurrent", [False, True])
def test_p2_exact_cache_equals_direct_execution(server, concurrent):
    for seed in range(60):
        mismatches = asyncio.run(run_exact_vs_nocache(server.port, seed, concurrent=concurrent))
        assert not mismatches, [str(m) for m in mismatches[:3]]


@pytest.mark.known_bug("B2")
def test_p2b_value_and_snapshot_come_from_the_same_run(server):
    assert asyncio.run(run_value_snapshot_coherence(server.port, trials=15)) == []


def _two_rollouts(port, first, second):
    async def go():
        a = await _one_rollout(port, ToyFsAdapter(), "hidden", 0, first, 1e-9)
        b = await _one_rollout(port, ToyFsAdapter(), "hidden", 1, second, 1e-9)
        return a + b

    return asyncio.run(go())


def test_hidden_process_state_lost_by_snapshots_is_detected(server):
    """Snapshots do not capture background processes (as with docker commit).
    A rollout that resumes from such a snapshot then sees a different `ps` than
    direct execution does. That is a limit of the paper's assumption A1, not a
    cache bug, and the framework must flag it."""
    spawn, ps = ToyCall("spawn", "p"), ToyCall("ps")
    ToyFsSandbox.lossy_snapshots = True
    mismatches = _two_rollouts(server.port, [spawn], [spawn, ps])
    assert len(mismatches) >= 1
    assert mismatches[0].cached == "" and mismatches[0].expected == "p"


def test_same_scenario_is_clean_when_snapshots_capture_everything(server):
    spawn, ps = ToyCall("spawn", "p"), ToyCall("ps")
    ToyFsSandbox.lossy_snapshots = False
    assert _two_rollouts(server.port, [spawn], [spawn, ps]) == []
