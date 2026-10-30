"""Unit tests (no services): trace loading and cache-dump walking."""

import json

import pytest

from tvcache_workloads.bash import load_trace_file
from tvcache_workloads.bash.cache_admin import snapshot_nodes
from tvcache_workloads.bash.runner import RolloutRecord

TRACE = "bash/traces/hello-world.json"


def test_bundled_trace_is_valid_and_targets_an_existing_task():
    trace = load_trace_file(TRACE)
    assert trace["task"] == "hello-world"
    names = [r["name"] for r in trace["rollouts"]]
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    "content", [{"rollouts": []}, {"task": "t"}, {"task": "t", "rollouts": [{"name": "x"}]}]
)
def test_invalid_traces_are_rejected(tmp_path, content):
    path = tmp_path / "t.json"
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        load_trace_file(path)


def test_test_passed_parses_only_real_results():
    record = RolloutRecord(name="r", task_id="t")
    assert record.test_passed is None
    record.test_result = '{"passed": true}'
    assert record.test_passed is True
    record.test_result = "Failed to execute tests: boom"
    assert record.test_passed is None


def test_snapshot_nodes_walks_dump_format():
    # Shape copied from a real tvcache-rust DumpAll response.
    def node(sandbox=None, **children):
        return {"sandbox_id": sandbox, "value": "v", "children": children}

    dump = {
        "t": {"root": node(None, a=node("tb-parked:1", b=node(None, c=node("tb-parked:2"))))},
        "other": {"root": node(None, x=node("tb-parked:9"))},
    }
    assert sorted(snapshot_nodes(dump, "t")) == [
        (["a"], "tb-parked:1"),
        (["a", "b", "c"], "tb-parked:2"),
    ]
    assert snapshot_nodes(dump, "missing") == []
