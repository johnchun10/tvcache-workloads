"""C4 invariants, checked by corrupting a known-good trie one way at a time."""

import copy

from tvverify.model.graph import GraphExport
from tvverify.testing.fixtures import wa_x
from tvverify.validators.structural import check_structure


def codes(findings):
    return {f.code for f in findings}


def clone(g: GraphExport) -> GraphExport:
    return GraphExport.from_json(g.to_json())


def check(fx, g, **kw):
    return check_structure(g, adapter=fx.adapter, snapshot_exists=fx.snapshot_exists, **kw)


def test_exact_trie_is_clean(exact_fixture):
    assert check(exact_fixture, exact_fixture.graph) == []


def test_each_corruption_is_reported(exact_fixture):
    fx = exact_fixture
    snap = next(n for n, node in fx.graph.nodes.items() if node.sandbox_id)
    other_snap = next(n for n, node in fx.graph.nodes.items() if node.sandbox_id and n != snap)
    leaf = next(n for n, node in fx.graph.nodes.items() if node.stateful is False)
    first = next(iter(fx.graph.edges[fx.graph.root_id].values()))

    g = clone(fx.graph); g.edges[g.root_id]["ghost"] = "nonexistent"
    assert "dangling_edge" in codes(check(fx, g))

    g = clone(fx.graph); g.edges[first] = {"loop": g.root_id}
    assert "root_has_parent" in codes(check(fx, g))

    g = clone(fx.graph); g.edges[other_snap] = {**g.edges.get(other_snap, {}), "again": snap}
    assert "multiple_parents" in codes(check(fx, g))

    g = clone(fx.graph); g.nodes[snap].path = ["somewhere", "else"]
    assert "path_mismatch" in codes(check(fx, g))

    g = clone(fx.graph); g.nodes["orphan"] = copy.deepcopy(g.nodes[snap]); g.nodes["orphan"].node_id = "orphan"
    assert "unreachable" in codes(check(fx, g))

    g = clone(fx.graph); g.edges[leaf] = {wa_x.serialize(): snap}
    assert "stateless_not_leaf" in codes(check(fx, g))

    g = clone(fx.graph); g.nodes[other_snap].sandbox_id = g.nodes[snap].sandbox_id
    assert "shared_sandbox" in codes(check(fx, g))

    g = clone(fx.graph); g.reclaimable.append(g.nodes[snap].sandbox_id)
    assert "reclaimable_in_use" in codes(check(fx, g))

    g = clone(fx.graph); g.sandbox_references = {"nobody-owns-me": 2}
    assert "referenced_sandbox_orphaned" in codes(check(fx, g))

    g = clone(fx.graph); g.counters["number_of_nodes"] += 1
    assert "counter_mismatch" in codes(check(fx, g))

    g = clone(fx.graph); g.nodes[snap].sandbox_id = "does-not-exist"
    assert "missing_snapshot" in codes(check(fx, g))


def test_leaked_leases_are_reported_when_quiescent(exact_fixture):
    g = clone(exact_fixture.graph)
    g.sandbox_references = {next(n.sandbox_id for n in g.nodes.values() if n.sandbox_id): 1}
    assert "leaked_reference" in codes(check_structure(g, adapter=exact_fixture.adapter))
    assert "leaked_reference" not in codes(check_structure(g, adapter=exact_fixture.adapter, quiescent=False))
