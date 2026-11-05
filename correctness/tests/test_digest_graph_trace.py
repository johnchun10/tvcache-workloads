import asyncio

import pytest
from tvclient import TvcacheToolExecutor

from tvverify.adapters.mock_fs import ToyCall, ToyFsSandbox, toy_digest, DEFAULT_POLICY, _State
from tvverify.adapters.toyfs_adapter import ToyFsAdapter
from tvverify.digest.schema import DigestPolicy, StateDigest, diff_digests
from tvverify.model.graph import GraphExport, from_dumpall
from tvclient import TVCacheClient


def digest_after(*calls):
    async def go():
        sandbox = ToyFsSandbox(task_name="d")
        await sandbox.start()
        for call in calls:
            await sandbox.execute(call)
        return StateDigest.from_dict(await sandbox.state_digest())

    return asyncio.run(go())


def test_equal_states_have_equal_digests_regardless_of_path():
    a = digest_after(ToyCall("write", "a", "x"), ToyCall("write", "b", "y"))
    b = digest_after(ToyCall("write", "b", "y"), ToyCall("write", "a", "x"))
    c = digest_after(ToyCall("write", "a", "x"), ToyCall("touch", "a"))
    d = digest_after(ToyCall("write", "a", "x"))
    assert a.strict_hash() == b.strict_hash()  # commuting writes: same state
    assert c.strict_hash() == d.strict_hash()  # touch on an existing file: no-op
    assert diff_digests(a, d).equal_strict is False


def test_digest_diff_explains_which_component_and_entry_differs():
    a = digest_after(ToyCall("write", "a", "x"))
    b = digest_after(ToyCall("write", "a", "y"), ToyCall("export", "K", "1"))
    diff = diff_digests(a, b)
    assert not diff.equal_strict
    assert set(diff.differing) == {"fs", "env"}
    assert ("/a", "x", "y") in diff.differing["fs"]


def test_advisory_components_never_decide_equality():
    a = digest_after(ToyCall("write", "a", "x"), ToyCall("spawn", "p"))
    b = digest_after(ToyCall("write", "a", "x"))
    diff = diff_digests(a, b)
    assert diff.equal_strict and diff.advisory_differing == {"procs"}


def test_digests_under_different_policies_are_not_comparable():
    a = digest_after(ToyCall("write", "a", "x"))
    other = DigestPolicy(strict_components=frozenset({"fs"}), advisory_components=frozenset())
    b = toy_digest(_State(), other)
    diff = diff_digests(a, b)
    assert not diff.comparable and not diff.equal_strict


def test_digest_roundtrips_through_dict():
    a = digest_after(ToyCall("write", "a", "x"))
    assert StateDigest.from_dict(a.to_dict()).strict_hash() == a.strict_hash()


@pytest.fixture(autouse=True)
def isolate_toyfs():
    with ToyFsSandbox.isolated():
        yield


def _write_then_read(port, task, **executor_kw):
    w, r = ToyCall("write", "a", "x"), ToyCall("read", "a")

    async def go():
        ex = TvcacheToolExecutor(
            ToyFsSandbox, ToyCall, task, port=port, adaptive=False, fixed_threshold_seconds=1e-9, **executor_kw
        )
        async with ex:
            await ex.lookup_or_execute([w])
            await ex.lookup_or_execute([w, r])
        async with TVCacheClient(port=port) as client:
            return await client.dump_all()

    return w, r, asyncio.run(go())


def test_graph_export_from_dumpall_and_json_roundtrip(server):
    w, r, dump = _write_then_read(server.port, "g")
    graph = from_dumpall(dump, ToyFsAdapter())["g"]

    assert len(graph.nodes) == 3  # root + 2
    stateful = {n.key: n.stateful for n in graph.nodes.values() if n.key}
    assert stateful == {w.serialize(): True, r.serialize(): False}
    assert graph.nodes[[n for n, node in graph.nodes.items() if node.key == r.serialize()][0]].path == [w.serialize(), r.serialize()]
    assert graph.counters["number_of_nodes"] == 2
    restored = GraphExport.from_json(graph.to_json())
    assert restored.nodes.keys() == graph.nodes.keys() and restored.edges == graph.edges


def test_trace_recorder_logs_lookups_snapshots_and_values(server, tmp_path):
    pytest.importorskip("tvclient.observer", reason="this tvclient has no observer hooks")
    from tvverify.trace.recorder import BlobStore, TraceRecorder, group_by_call, read_trace

    recorder = TraceRecorder(tmp_path / "trace.jsonl", BlobStore(tmp_path / "blobs"), digest_every=1)
    _write_then_read(server.port, "t", observer=recorder)

    # one snapshot (after the stateful write), digests present, values in blobs
    events = read_trace(tmp_path / "trace.jsonl")
    kinds = [e["type"] for e in events]
    assert kinds.count("lookup") == 2 and kinds.count("snapshot") == 1
    snap = next(e for e in events if e["type"] == "snapshot")
    assert snap["digest"] is not None and snap["depth"] == 1
    returned = next(e for e in events if e["type"] == "return" and e["idx"] == 2)
    assert recorder.blobs.get(returned["value_sha"]) == "x"
    assert len(group_by_call(events)) == 2
