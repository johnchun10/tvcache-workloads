"""stats.py on a small synthetic run directory."""

import json

from tvcache_workloads.bash.verl import stats

TASK = "hello::run1"
A = '{"command": "a", "tool": "bash"}'
B = '{"command": "b", "tool": "bash"}'


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def ev(t, kind, rollout, idx, **fields):
    return {"v": "trace/v1", "type": kind, "t": t, "task": TASK, "rollout": rollout, "idx": idx, **fields}


def make_run(tmp_path):
    run = tmp_path / "run"
    write_jsonl(run / "control.jsonl", [
        {"t": 100, "event": "begin", "step": 1, "epoch": 0, "tasks": ["hello"]},
        {"t": 110, "event": "take", "task": TASK, "parent": "root", "env_id": "term_1"},
        {"t": 150, "event": "end", "step": 1, "epoch": 0, "tasks": ["hello"], "last": False},
        {"t": 150.5, "event": "graph", "task": TASK, "step": 1, "epoch": 0, "graph_depth": 2, "graph_nodes": 2},
        {"t": 200, "event": "idle", "task": TASK, "step": 2, "epoch": 1, "idle_s": 49.5},
        {"t": 200, "event": "begin", "step": 2, "epoch": 1, "tasks": ["hello"]},
        {"t": 250, "event": "end", "step": 2, "epoch": 1, "tasks": ["hello"], "last": True},
    ])
    write_jsonl(run / "trace" / "trace-1.jsonl", [
        # epoch 0, rollout r1: a, then b; both executed and inserted
        ev(110, "lookup", "r1", 1, path=[A], hit=False, lpm_depth=None, sandbox_id=None),
        ev(110.1, "resume", "r1", 1, source="fresh", depth=0, env_id="term_1", from_bank=True),
        ev(112, "executed", "r1", 1, depth=1, key=A, value_sha="x", runtime_s=2.0, failed=False, digest=None),
        ev(112.1, "insert", "r1", 1, path=[A], value_sha="x", sandbox_id="tb-snap:snap_aa",
           value_inserted=True, sandbox_inserted=True),
        ev(112.2, "return", "r1", 1, source="exec", value_sha="x"),
        ev(113, "lookup", "r1", 2, path=[A, B], hit=False, lpm_depth=1, sandbox_id="tb-snap:snap_aa"),
        ev(113.1, "resume", "r1", 2, source="live", depth=1, env_id="term_1", from_bank=False),
        ev(114, "executed", "r1", 2, depth=2, key=B, value_sha="y", runtime_s=1.0, failed=False, digest=None),
        ev(114.1, "insert", "r1", 2, path=[A, B], value_sha="y", sandbox_id=None,
           value_inserted=True, sandbox_inserted=False),
        ev(114.2, "return", "r1", 2, source="exec", value_sha="y"),
        # epoch 1, rollout r2: a is a hit; then a new call c resumes from a's snapshot
        ev(210, "lookup", "r2", 1, path=[A], hit=True, lpm_depth=None, sandbox_id=None),
        ev(210.1, "return", "r2", 1, source="hit", value_sha="x"),
        ev(211, "lookup", "r2", 2, path=[A, '{"command": "c", "tool": "bash"}'], hit=False, lpm_depth=1,
           sandbox_id="tb-snap:snap_aa"),
        ev(211.1, "resume", "r2", 2, source="fork", depth=1, env_id="term_2", from_bank=True),
        ev(212, "executed", "r2", 2, depth=2, key="c", value_sha="z", runtime_s=1.0, failed=False, digest=None),
        ev(212.2, "return", "r2", 2, source="exec", value_sha="z"),
    ])
    write_jsonl(run / "trace" / "rollouts-1.jsonl", [
        {"t_start": 105, "t_end": 140, "task": "hello", "rollout": "r1", "num_turns": 4, "response_len": 2048,
         "max_response_len": 2048, "max_assistant_turns": 30, "length_truncated": True, "calls": 2,
         "failed_calls": 0, "test": '{"passed": true}'},
        {"t_start": 205, "t_end": 240, "task": "hello", "rollout": "r2", "num_turns": 60, "response_len": 900,
         "max_response_len": 2048, "max_assistant_turns": 30, "length_truncated": False, "calls": 2,
         "failed_calls": 0, "test": '{"passed": false}'},
    ])
    write_jsonl(run / "contraction" / "events.jsonl", [
        {"time": 151, "elapsed": 1, "event": "resume", "task": TASK},
        {"time": 160, "elapsed": 10, "event": "job_end", "task": TASK, "depth": 2, "comparisons": 1, "merges": 1,
         "restores": 2, "replays": 0, "unavailable": 0},
        {"time": 160, "elapsed": 10, "event": "round", "task": TASK, "deepest_depth": 2, "comparisons": 1,
         "merges": 1, "restores": 2, "replays": 0, "unavailable": 0, "groups": 1, "similar_pairs": 1,
         "pairs_decided": 1, "equal_comparisons": 1, "fresh_starts": 0, "nodes_built": 2, "merged_nodes": 1,
         "completed": True, "preempted": False, "walltime_s": 9.04},
    ])
    (run / "logs").mkdir(parents=True)
    (run / "logs" / "memory.csv").write_text(
        "time,cgroup_mem_gb,graphroot_gb,containers,images,sandboxes\n"
        "2026-11-13T09:00:00,3.1,2.0,9,3,4\n"
        "2026-11-13T09:01:00,3.7,2.5,12,4,6\n"
        "2026-11-13T09:02:00,NA,NA,0,0,0\n"
    )
    (run / "logs" / "manager.log").write_text("Snapshot snap_aa of term_1 took 1.2s, 5000 bytes\n")
    (run / "dumps").mkdir()
    graph = {"number_of_nodes": 2, "root": {"children": [
        {"tool_calls": {A: {"value": "x"}}, "sandbox_id": "tb-snap:snap_aa", "children": [
            {"tool_calls": {B: {"value": "y"}, '{"command": "b2", "tool": "bash"}': {"value": "y"}}, "children": []},
        ]},
    ]}}
    (run / "dumps" / "epoch_001.json").write_text(json.dumps({TASK: graph}))
    return run


def test_run_metrics_per_task_epoch(tmp_path):
    metrics = stats.run_metrics(make_run(tmp_path))
    e0, e1 = metrics["tasks"]["hello"][0], metrics["tasks"]["hello"][1]
    assert (e0["lookups"], e0["cache_hit_rate"], e0["lpm_hit_rate"]) == (2, 0.0, 0.0)
    # LPM rate is over misses: the one miss resumed from a's snapshot
    assert (e1["lookups"], e1["cache_hit_rate"], e1["lpm_hit_rate"]) == (2, 0.5, 1.0)
    assert round(e1["lpm_time_saved_pct"], 6) == round(200 / 3, 6)  # a: 2 s skipped, c: 1 s run
    assert e1["new_seq_hit_rate"] == 0.0  # the hit replays a path r1 executed literally
    assert (e0["length_truncated"], e1["turn_capped"], e0["passed"], e1["passed"]) == (1, 1, 1, 0)
    assert e0["from_bank"] == {"fresh": 1} and e1["from_bank"] == {"fork": 1}  # by resume source
    # epoch_001: one merged node, one snapshot of known size, depth 2
    assert (e1["nodes"], e1["merged_nodes_in_graph"], e1["snapshots"], e1["snapshot_bytes"], e1["graph_depth"]) == (
        2, 1, 1, 5000, 2)


def test_rounds_and_memory(tmp_path):
    metrics = stats.run_metrics(make_run(tmp_path))
    (r,) = metrics["rounds"]
    assert (r["task"], r["step"], r["graph_depth"], r["depth_reached"], r["idle_s"]) == ("hello", 1, 2, 2, 49.5)
    assert (r["similar_pairs"], r["comparisons"], r["merged_nodes"], r["preempted"], r["walltime_s"]) == (
        1, 1, 1, False, 9.0)
    live = metrics["live_summary"]
    assert (live["peak_sandboxes"], live["peak_containers"], live["peak_mem_gb"]) == (6, 12, 3.7)
    assert metrics["fork_bank"]["takes"] == 1


def test_rounds_from_an_older_tvclient_use_job_end(tmp_path):
    run = make_run(tmp_path)
    events = run / "contraction" / "events.jsonl"
    events.write_text("".join(line + "\n" for line in events.read_text().splitlines() if '"round"' not in line))
    (r,) = stats.run_metrics(run)["rounds"]
    assert (r["depth_reached"], r["comparisons"], r["merged_classes"], r["similar_pairs"]) == (2, 1, 1, None)


def test_compare_writes_the_ratio_table(tmp_path):
    metrics = stats.run_metrics(make_run(tmp_path))
    table = stats.compare(metrics, metrics)
    assert "| method hit rate / baseline hit rate | 1.000 |" in table
    assert "| comparisons / similar pairs | 1.000 |" in table
