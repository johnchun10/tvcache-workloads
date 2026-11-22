"""check-graph: replay checks on a saved DUMP_ALL, node budgets, the
inconclusive bucket, JSON results and the report."""

import asyncio
import json
from types import SimpleNamespace

from tvclient import TvcacheToolExecutor, TVCacheClient

from tvverify.adapters.mock_fs import ToyCall, ToyFsSandbox
from tvverify.adapters.toyfs_adapter import ToyFsAdapter
from tvverify.check import _classify, main_graph, select_nodes
from tvverify.model.dump import plain_dump
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
        dump = plain_dump(save_dump(server.port, tmp_path))  # one shape, whichever server wrote it
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


def test_random_file_with_steady_output_is_noise_not_a_bug():
    """`keygen` prints "ok" every time but writes a random file, like
    `openssl genrsa -out f`: two fresh replays reach different states, so a
    snapshot difference there is noise. A deterministic path stays a bug."""
    adapter = ToyFsAdapter()
    keygen = ToyCall("keygen", "k")
    findings = [
        Finding("C2a", "snapshot_state_differs", "snapshot differs", "n1", (w.serialize(), keygen.serialize())),
        Finding("C2a", "snapshot_state_differs", "snapshot differs", "n2", (w.serialize(),)),
    ]
    with ToyFsSandbox.isolated():
        engine = ReplayEngine(adapter, "cg", memoize=False)
        bugs, noise, inconclusive = asyncio.run(_classify(adapter, engine, findings, replays=2))
    assert [f.node_id for f in noise] == ["n1"]
    assert [f.node_id for f in bugs] == ["n2"] and inconclusive == []


def contracted_dump() -> dict:
    """A dump from a server with graph contraction: two merged nodes (one
    right, one wrong) and a child, plus plain nodes to fill a budget."""
    def node(calls, children=(), hits=0):
        return {"children": list(children), "sandbox_id": None, "settled": True,
                "tool_calls": {c.serialize(): {"value": v, "runtime": 1, "embedding": None} for c, v in calls},
                "number_of_cache_hits": hits, "number_of_lpm_hits": 0}

    right = node([(ToyCall("write", "a", "x"), "ok"), (ToyCall("append", "a", "x"), "len=1")],
                 children=[node([(ToyCall("read", "a"), "x")])])
    wrong = node([(ToyCall("write", "b", "1"), "ok"), (ToyCall("write", "b", "2"), "ok")])  # different files, merged anyway
    plain = [node([(ToyCall("touch", f"t{i}"), "ok")], hits=5 - i) for i in range(5)]
    return {"mg": {"root": node([], children=[right, wrong, *plain]), "number_of_nodes": 8, "number_of_sandboxes": 0,
                   "size_in_bytes": 0, "sandbox_references": {}, "reclaimable_sandboxes": []}}


def test_merged_calls_are_checked_first_and_against_their_node(tmp_path):
    """A call contraction merged into a node must leave the state the node's
    first call leaves. Those calls are replayed first, and the result counts
    how many were found equivalent."""
    dump = contracted_dump()
    (tmp_path / "dump.json").write_text(json.dumps(dump))
    graph = from_dumpall(dump, ToyFsAdapter())["mg"]
    by_key = {node.key: node_id for node_id, node in graph.nodes.items() if node.key}
    key = lambda *call: by_key[ToyCall(*call).serialize()]

    # Budget 6 of 10 entries: the two merged calls and the two nodes holding them take the
    # first half; the rest goes to the most-served nodes and a seeded sample, as before.
    picked = select_nodes(graph, 6, seed=0)
    merged_calls = {node_id for node_id, node in graph.nodes.items() if node.alias_of}
    assert len(merged_calls) == 2 and set(picked[:2]) == merged_calls
    assert len(picked) == 6 and len(set(picked)) == 6 and key("touch", "t0") in picked
    assert picked == select_nodes(graph, 6, seed=0)

    with ToyFsSandbox.isolated():
        assert main_graph(graph_args(tmp_path, task=["mg"], no_snapshots=True)) == 1
    task = json.loads((tmp_path / "result.json").read_text())["tasks"][0]
    assert [(b["criterion"], b["code"]) for b in task["bugs"]] == [("C2m", "merged_state_differs")]
    wrong_alias = max(key("write", "b", "1"), key("write", "b", "2"), key=lambda n: bool(graph.nodes[n].alias_of))
    assert task["bugs"][0]["node_id"] == wrong_alias
    assert task["merged"] == {"nodes": 2, "calls": 2, "calls_checked": 2, "calls_equivalent": 1,
                              "nodes_and_children_replayed": 3}
    assert (task["state_checked"], task["state_matched"]) == (2, 1)  # no snapshots here: only the merges
    report = render([str(tmp_path / "result.json")])
    assert "| merged calls equivalent to their node / checked | 1 / 2 |" in report
    assert "| merged nodes in the tries (extra calls they hold) | 2 (2) |" in report


def test_merged_call_with_a_snapshot_is_checked_against_the_snapshot_only(tmp_path, monkeypatch):
    """Where the merged node holds a snapshot, the snapshot check (C2a) already
    compares each merged call's replay with the node's state, so the two extra
    replays of the merge check are not spent."""
    async def snapshot_of(*calls):
        sandbox = ToyFsSandbox(task_name="mg")
        await sandbox.start()
        for call in calls:
            await sandbox.execute(call)
        handle = await sandbox.fork()
        await sandbox.stop()
        return handle.get_id()

    digests = []
    real = ReplayEngine.state_digest

    async def counting(self, keys):
        digests.append(list(keys))
        return await real(self, keys)

    monkeypatch.setattr(ReplayEngine, "state_digest", counting)
    with ToyFsSandbox.isolated():
        dump = contracted_dump()
        right, wrong = dump["mg"]["root"]["children"][:2]
        right["sandbox_id"] = asyncio.run(snapshot_of(ToyCall("write", "a", "x")))
        dump["mg"]["number_of_sandboxes"] = 1
        (tmp_path / "dump.json").write_text(json.dumps(dump))
        assert main_graph(graph_args(tmp_path, task=["mg"])) == 1  # the wrong merge is still found
    task = json.loads((tmp_path / "result.json").read_text())["tasks"][0]
    assert [(b["criterion"], b["code"]) for b in task["bugs"]] == [("C2m", "merged_state_differs")]
    assert task["merged"]["calls_checked"] == 2 and task["merged"]["calls_equivalent"] == 1
    append, write_a = ToyCall("append", "a", "x").serialize(), ToyCall("write", "a", "x").serialize()
    # The right merge: one digest for each of its two calls (C2a), none for the pair.
    assert digests.count([append]) == 1 and digests.count([write_a]) == 1
