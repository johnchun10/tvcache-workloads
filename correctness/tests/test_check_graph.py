"""check-graph: replay checks on a saved DUMP_ALL, node budgets, the
inconclusive bucket, JSON results and the report."""

import asyncio
import json
from types import SimpleNamespace

from tvclient import TvcacheToolExecutor, TVCacheClient

from tvverify.adapters.mock_fs import ToyCall, ToyFsSandbox
from tvverify.adapters.toyfs_adapter import ToyFsAdapter
from tvverify.check import _classify, main_graph, select_nodes
from tvverify.model.graph import from_dumpall
from tvverify.replay.engine import ReplayEngine
from tvverify.report import render
from tvverify.validators.findings import Finding

ADAPTER = "tvverify.adapters.toyfs_adapter:ToyFsAdapter"
w, r, ls = ToyCall("write", "a", "x"), ToyCall("read", "a"), ToyCall("ls")


def save_dump(port, tmp_path, task="cg") -> dict:
    async def go():
        for calls in ([w, r], [w, ls], [w, r]):  # the third rollout is all hits
            ex = TvcacheToolExecutor(ToyFsSandbox, ToyCall, task, port=port, adaptive=False, fixed_threshold_seconds=1e-9)
            async with ex:
                for k in range(len(calls)):
                    await ex.lookup_or_execute(list(calls[: k + 1]))
                await ex.test(list(calls))
        async with TVCacheClient(port=port) as client:
            return await client.dump_all()

    dump = asyncio.run(go())
    (tmp_path / "dump.json").write_text(json.dumps(dump))
    return dump


def graph_args(tmp_path, **kw):
    base = dict(adapter=ADAPTER, dump=str(tmp_path / "dump.json"), task=["cg"], max_replay_nodes=20,
                no_snapshots=False, seed=0, replays=2, json=str(tmp_path / "result.json"))
    return SimpleNamespace(**{**base, **kw})


def test_clean_dump_passes_and_writes_json(server, tmp_path):
    with ToyFsSandbox.isolated():
        save_dump(server.port, tmp_path)
        assert main_graph(graph_args(tmp_path)) == 0
    result = json.loads((tmp_path / "result.json").read_text())
    task = result["tasks"][0]
    assert result["command"] == "check-graph" and result["status"] == 0
    assert task["bugs"] == [] and task["nodes_replayed"] == task["nodes_total"] > 0


def test_tampered_value_is_a_bug(server, tmp_path):
    with ToyFsSandbox.isolated():
        dump = save_dump(server.port, tmp_path)
        node = dump["cg"]["root"]["children"][w.serialize()]["children"][r.serialize()]
        node["value"] = "tampered"
        (tmp_path / "dump.json").write_text(json.dumps(dump))
        assert main_graph(graph_args(tmp_path)) == 1
    bugs = json.loads((tmp_path / "result.json").read_text())["tasks"][0]["bugs"]
    assert [(b["criterion"], b["code"]) for b in bugs] == [("C1", "output_mismatch")]
    assert "tampered" in render([str(tmp_path / "result.json")])


def test_no_snapshots_skips_the_snapshot_checks(server, tmp_path):
    with ToyFsSandbox.isolated():
        save_dump(server.port, tmp_path)
        ToyFsSandbox.snapshots.clear()  # the run is over and its snapshots are gone
        assert main_graph(graph_args(tmp_path, no_snapshots=True)) == 0
    assert json.loads((tmp_path / "result.json").read_text())["tasks"][0]["snapshots_checked"] is False


def test_budget_takes_the_most_served_nodes_then_a_seeded_sample(server, tmp_path):
    with ToyFsSandbox.isolated():
        dump = save_dump(server.port, tmp_path)
    graph = from_dumpall(dump, ToyFsAdapter())["cg"]
    hottest = max((n for n in graph.nodes if n != graph.root_id), key=lambda n: graph.nodes[n].hits + graph.nodes[n].lpm_hits)
    picked = select_nodes(graph, 2, seed=1)
    assert len(picked) == 2 and picked[0] == hottest
    assert picked == select_nodes(graph, 2, seed=1)  # deterministic
    assert len(select_nodes(graph, None, 0)) == len(graph.nodes) - 1


def test_replays_that_could_not_run_are_inconclusive():
    findings = [
        Finding("C1", "replay_failed", "fresh replay could not run the call", "n1", ("k",)),
        Finding("C1", "replay_error", "TimeoutError", "n2", ("k",)),
        Finding("C4", "counter_mismatch", "off by one"),
    ]
    engine = ReplayEngine(ToyFsAdapter(), "unused", memoize=False)
    bugs, noise, inconclusive = asyncio.run(_classify(ToyFsAdapter(), engine, findings, replays=2))
    assert [f.code for f in inconclusive] == ["replay_failed", "replay_error"]
    assert [f.code for f in bugs] == ["counter_mismatch"] and noise == []
