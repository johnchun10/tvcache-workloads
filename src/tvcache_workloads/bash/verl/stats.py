"""The testing plan's metrics for one training run, from what the run wrote:

    $RUN/trace/trace-*.jsonl      every lookup/resume/execution (TraceRecorder)
    $RUN/trace/rollouts-*.jsonl   one line per trajectory (agent_loop.py)
    $RUN/control.jsonl            step boundaries, idle windows, graph sizes, fork bank
    $RUN/contraction/events.jsonl contraction rounds (bash/contraction.py)
    $RUN/dumps/epoch_NNN.json     the trie as each epoch ended
    $RUN/logs/manager.log         snapshot sizes ("Snapshot snap_.. took ..s, N bytes")
    $RUN/logs/memory.csv          job memory, containers and sandboxes once a minute

    python -m tvcache_workloads.bash.verl.stats $RUN           # -> $RUN/metrics.{json,md}
    python -m tvcache_workloads.bash.verl.stats --compare $BASELINE_RUN $METHOD_RUN

Standard library only. A lookup belongs to the epoch of the last step that
began before it. Test calls (the rollout's final test) are left out of the
tool-call metrics.
"""

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

NAMESPACE_SEPARATOR = "::"
TEST_KEY_MARK = "TV_CACHE_TOOL_TYPE"
SNAPSHOT_LINE = re.compile(r"Snapshot (snap_[0-9a-f]+) of \S+ took [\d.]+s, (\d+) bytes")


def read_jsonl(paths: Iterable[Path]) -> List[dict]:
    events = []
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a line cut off by a killed process
    return events


def base_task(task: Optional[str]) -> Optional[str]:
    return task.split(NAMESPACE_SEPARATOR, 1)[0] if task else task


def is_test(path: List[str]) -> bool:
    return bool(path) and TEST_KEY_MARK in path[-1]


class EpochClock:
    """Maps a time to the epoch (and step) of the last step that began before it."""

    def __init__(self, control: List[dict]):
        self.begins = sorted((e["t"], e["epoch"], e["step"]) for e in control if e["event"] == "begin")

    def epoch(self, t: float) -> int:
        current = self.begins[0][1] if self.begins else 0
        for start, epoch, _ in self.begins:
            if start > t:
                break
            current = epoch
        return current


def walk(node: dict):
    children = node.get("children") or []
    if isinstance(children, dict):
        children = list(children.values())
    for child in children:
        yield child
        yield from walk(child)


def depth(node: dict) -> int:
    children = node.get("children") or []
    if isinstance(children, dict):
        children = list(children.values())
    return 1 + max((depth(c) for c in children), default=0)


def ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    return None if numerator is None or not denominator else numerator / denominator


def run_metrics(run: Path) -> dict:
    trace = read_jsonl(sorted((run / "trace").glob("trace-*.jsonl")))
    rollouts = read_jsonl(sorted((run / "trace").glob("rollouts-*.jsonl")))
    control_path = run / "control.jsonl"
    control = read_jsonl([control_path]) if control_path.exists() else []
    clock = EpochClock(control)

    snapshot_bytes: Dict[str, int] = {}
    manager_log = run / "logs" / "manager.log"
    if manager_log.exists():
        for match in SNAPSHOT_LINE.finditer(manager_log.read_text(errors="replace")):
            snapshot_bytes[match.group(1)] = int(match.group(2))

    per: Dict[tuple, dict] = defaultdict(lambda: defaultdict(float))
    call_times: Dict[tuple, List[float]] = defaultdict(list)
    inserted: Dict[str, set] = defaultdict(set)  # task -> literal paths executed and inserted
    runtime_of: Dict[str, Dict[tuple, float]] = defaultdict(dict)  # task -> path -> seconds
    lookups: Dict[tuple, dict] = {}

    for event in sorted(trace, key=lambda e: e["t"]):
        task = base_task(event["task"])
        key = (event["task"], event["rollout"], event["idx"])
        kind = event["type"]
        if kind == "lookup":
            lookups[key] = event
            if is_test(event["path"]):
                continue
            row = per[(task, clock.epoch(event["t"]))]
            row["lookups"] += 1
            if event["hit"]:
                row["hits"] += 1
                if tuple(event["path"]) not in inserted[task]:
                    row["new_seq_hits"] += 1
        elif kind == "resume":
            lookup = lookups.get(key)
            if lookup is None or is_test(lookup["path"]):
                continue
            row = per[(task, clock.epoch(event["t"]))]
            row[f"resume_{event['source']}"] += 1
            if event.get("from_bank"):
                row[f"bank_{event['source']}"] += 1
            if event["source"] == "fork":
                row["lpm_hits"] += 1
                prefix = tuple(lookup["path"][: event["depth"]])
                if prefix not in inserted[task]:
                    row["new_prefix_lpm_hits"] += 1
                saved, unknown = 0.0, 0
                for d in range(1, event["depth"] + 1):
                    seconds = runtime_of[task].get(tuple(lookup["path"][:d]))
                    if seconds is None:
                        unknown += 1
                    else:
                        saved += seconds
                row["lpm_saved_s"] += saved
                row["lpm_saved_unknown_calls"] += unknown
        elif kind == "executed":
            lookup = lookups.get(key)
            if lookup is None:
                continue
            path = tuple(lookup["path"][: event["depth"]])
            runtime_of[task].setdefault(path, event["runtime_s"])
            if not is_test(lookup["path"]):
                row = per[(task, clock.epoch(event["t"]))]
                row["executed"] += 1
                row["executed_s"] += event["runtime_s"]
        elif kind == "insert":
            inserted[task].add(tuple(event["path"]))
        elif kind == "snapshot":
            per[(task, clock.epoch(event["t"]))]["snapshots_taken"] += 1
        elif kind == "return":
            lookup = lookups.get(key)
            if lookup is not None and not is_test(lookup["path"]):
                call_times[(task, clock.epoch(lookup["t"]))].append(event["t"] - lookup["t"])

    for record in rollouts:
        row = per[(base_task(record.get("task")), clock.epoch(record["t_start"]))]
        row["rollouts"] += 1
        row["length_truncated"] += bool(record.get("length_truncated"))
        max_turns = record.get("max_assistant_turns") or 0
        # num_turns counts user, assistant and tool messages; an assistant turn
        # that calls a tool adds two, so ~2*max means the turn cap stopped it.
        row["turn_capped"] += bool(max_turns and (record.get("num_turns") or 0) >= 2 * max_turns)
        row["passed"] += '"passed": true' in str(record.get("test"))

    tasks: Dict[str, dict] = defaultdict(dict)
    for (task, epoch), row in sorted(per.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        times = call_times.get((task, epoch), [])
        lookups_n = row["lookups"]
        misses = lookups_n - row["hits"]
        tasks[task][epoch] = {
            "lookups": int(lookups_n),
            "cache_hit_rate": ratio(row["hits"], lookups_n),
            "new_seq_hit_rate": ratio(row["new_seq_hits"], lookups_n),
            "lpm_hit_rate": ratio(row["lpm_hits"], misses),
            "new_prefix_lpm_hit_rate": ratio(row["new_prefix_lpm_hits"], misses),
            "lpm_time_saved_pct": ratio(100 * row["lpm_saved_s"], row["lpm_saved_s"] + row["executed_s"]),
            "lpm_saved_unknown_calls": int(row["lpm_saved_unknown_calls"]),
            "median_call_s": statistics.median(times) if times else None,
            "executed_calls": int(row["executed"]),
            "executed_s": round(row["executed_s"], 2),
            "resumes": {k[len("resume_"):]: int(v) for k, v in row.items() if k.startswith("resume_")},
            "from_bank": {k[len("bank_"):]: int(v) for k, v in row.items() if k.startswith("bank_")},
            "snapshots_taken": int(row["snapshots_taken"]),
            "rollouts": int(row["rollouts"]),
            "length_truncated": int(row["length_truncated"]),
            "turn_capped": int(row["turn_capped"]),
            "passed": int(row["passed"]),
        }

    # Graph size as each epoch ended.
    for dump_path in sorted((run / "dumps").glob("epoch_*.json")):
        epoch = int(dump_path.stem.split("_")[1])
        for task_id, graph in json.loads(dump_path.read_text()).items():
            nodes = list(walk(graph["root"]))
            sandboxes = [n["sandbox_id"] for n in nodes if n.get("sandbox_id")]
            sizes = [snapshot_bytes.get(s.split(":", 1)[-1]) for s in sandboxes]
            entry = tasks[base_task(task_id)].setdefault(epoch, {})
            entry.update({
                "nodes": graph.get("number_of_nodes", len(nodes)),
                "merged_nodes_in_graph": sum(len(n.get("tool_calls") or {}) > 1 for n in nodes),
                "snapshots": len(sandboxes),
                "snapshot_bytes": sum(s for s in sizes if s is not None),
                "snapshot_sizes_unknown": sum(s is None for s in sizes),
                "graph_depth": depth(graph["root"]) - 1,
            })

    round_rows = contraction_rounds(run, control)

    # Live sandboxes and memory over time, on the node running the containers.
    # (podman stats' per-container memory is one shared value on podman-hpc,
    # so the job's cgroup memory is the number to use.)
    live = []
    memory_csv = run / "logs" / "memory.csv"
    if memory_csv.exists():
        for row in csv.DictReader(memory_csv.open()):
            live.append({
                "time": row.get("time"),
                "sandboxes": _int(row.get("sandboxes")),
                "containers": _int(row.get("containers")),
                "mem_gb": _float(row.get("cgroup_mem_gb")),
                "image_store_gb": _float(row.get("graphroot_gb")),
            })

    takes = sum(e["event"] == "take" for e in control)
    warms = [e for e in control if e["event"] == "warm"]
    return {
        "run": str(run),
        "tasks": tasks,
        "rounds": round_rows,
        "live_sandboxes": live,
        "live_summary": {
            "peak_sandboxes": _peak(live, "sandboxes"),
            "peak_containers": _peak(live, "containers"),
            "peak_mem_gb": _peak(live, "mem_gb"),
            "peak_image_store_gb": _peak(live, "image_store_gb"),
        },
        "fork_bank": {
            "takes": takes,
            "warm_ups": len(warms),
            "last": warms[-1] if warms else None,
            "closed": next((e for e in control if e["event"] == "closed"), None),
        },
        "trace_events": len(trace),
    }


def _int(text) -> Optional[int]:
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _float(text) -> Optional[float]:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _peak(samples: List[dict], key: str):
    values = [s[key] for s in samples if s.get(key) is not None]
    return max(values) if values else None


def contraction_rounds(run: Path, control: List[dict]) -> List[dict]:
    """One row per contraction job, from the coordinator's events: `round`
    (the full JobSummary, from a tvclient with Contractor(on_summary=...)) or,
    from an older tvclient, `job_end` (its log line). The graph it worked on
    and its idle window come from control.jsonl."""
    events_path = run / "contraction" / "events.jsonl"
    events = read_jsonl([events_path]) if events_path.exists() else []
    kind = "round" if any(e["event"] == "round" for e in events) else "job_end"
    graphs = sorted((e for e in control if e["event"] == "graph"), key=lambda e: e["t"])
    idles = sorted((e for e in control if e["event"] == "idle"), key=lambda e: e["t"])
    rows = []
    for r in (e for e in events if e["event"] == kind):
        task, t = base_task(r["task"]), r["time"]
        # The graph the job started on: the last step end of its task before it.
        graph = next((g for g in reversed(graphs) if base_task(g["task"]) == task and g["t"] <= t), {})
        idle = next((i for i in idles if base_task(i["task"]) == task and i["t"] >= t - 1), None)
        rows.append({
            "task": task, "epoch": graph.get("epoch"), "step": graph.get("step"),
            "preempted": r.get("preempted"), "completed": r.get("completed"),
            "depth_reached": r.get("deepest_depth", r.get("depth")), "graph_depth": graph.get("graph_depth"),
            "walltime_s": None if r.get("walltime_s") is None else round(r["walltime_s"], 1),
            "idle_s": None if idle is None else round(idle["idle_s"], 1),
            "similar_pairs": r.get("similar_pairs"), "pairs_decided": r.get("pairs_decided"),
            "comparisons": r.get("comparisons"), "equal_comparisons": r.get("equal_comparisons"),
            "sandboxes_built": None if r.get("fresh_starts") is None else r["restores"] + r["fresh_starts"],
            "restores": r.get("restores"), "fresh_starts": r.get("fresh_starts"), "nodes_built": r.get("nodes_built"),
            "replayed_calls": r.get("replays"), "merged_classes": r.get("merges"),
            "merged_nodes": r.get("merged_nodes"), "unavailable": r.get("unavailable"),
        })
    return rows


def fmt(value) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
    return str(value)


def to_markdown(metrics: dict) -> str:
    out = [f"# Metrics: {metrics['run']}", ""]
    cols = ["lookups", "cache_hit_rate", "new_seq_hit_rate", "lpm_hit_rate", "new_prefix_lpm_hit_rate",
            "lpm_time_saved_pct", "median_call_s", "nodes", "snapshots", "snapshot_bytes",
            "graph_depth", "merged_nodes_in_graph", "rollouts", "length_truncated", "turn_capped", "passed"]
    out += ["## Per task, per epoch", "", "| task | epoch | " + " | ".join(cols) + " |",
            "|---|---|" + "---|" * len(cols)]
    for task, epochs in metrics["tasks"].items():
        for epoch, row in sorted(epochs.items(), key=lambda kv: int(kv[0])):
            out.append(f"| {task} | {epoch} | " + " | ".join(fmt(row.get(c)) for c in cols) + " |")
    out += ["", "Resume sources and fork-bank use per task/epoch:", ""]
    for task, epochs in metrics["tasks"].items():
        for epoch, row in sorted(epochs.items(), key=lambda kv: int(kv[0])):
            out.append(f"- {task} e{epoch}: resumes {row.get('resumes')}, from bank {row.get('from_bank')}, "
                       f"executed {row.get('executed_calls')} calls / {row.get('executed_s')} s, "
                       f"LPM saved-time calls without a known runtime: {row.get('lpm_saved_unknown_calls')}")
    if metrics["rounds"]:
        rcols = list(metrics["rounds"][0].keys())
        out += ["", "## [M] Contraction rounds", "", "| " + " | ".join(rcols) + " |", "|" + "---|" * len(rcols)]
        for r in metrics["rounds"]:
            out.append("| " + " | ".join(fmt(r[c]) for c in rcols) + " |")
    out += ["", "## Live sandboxes", "", f"{metrics['live_summary']}", ""]
    out += ["Fork bank: " + json.dumps(metrics["fork_bank"]), ""]
    return "\n".join(out)


def totals(metrics: dict) -> dict:
    """Run-wide numbers for the ratio table (sums over tasks and epochs)."""
    rows = [row for epochs in metrics["tasks"].values() for row in epochs.values()]
    lookups = sum(r.get("lookups", 0) for r in rows)

    def weighted(name):
        return ratio(sum((r.get(name) or 0) * r.get("lookups", 0) for r in rows), lookups)

    last_epoch = {}
    for epochs in metrics["tasks"].values():
        if epochs:
            last = epochs[max(epochs, key=int)]
            for k in ("nodes", "snapshots", "snapshot_bytes"):
                last_epoch[k] = last_epoch.get(k, 0) + (last.get(k) or 0)
    medians = [r["median_call_s"] for r in rows if r.get("median_call_s") is not None]
    return {"cache_hit_rate": weighted("cache_hit_rate"), "new_seq_hit_rate": weighted("new_seq_hit_rate"),
            "lpm_hit_rate": weighted("lpm_hit_rate"), "new_prefix_lpm_hit_rate": weighted("new_prefix_lpm_hit_rate"),
            "lpm_time_saved_pct": weighted("lpm_time_saved_pct"),
            "median_call_s": statistics.median(medians) if medians else None, **last_epoch}


def compare(baseline: dict, method: dict) -> str:
    b, m = totals(baseline), totals(method)
    rounds = method["rounds"]

    def total(key):
        values = [r[key] for r in rounds if r.get(key) is not None]
        return sum(values) if values else None

    lines = ["# Method vs baseline", "", "| ratio | value |", "|---|---|"]
    pairs = [
        ("method hit rate / baseline hit rate", ratio(m["cache_hit_rate"] or 0, b["cache_hit_rate"] or 0)),
        ("method new-seq hit rate / method hit rate", ratio(m["new_seq_hit_rate"] or 0, m["cache_hit_rate"] or 0)),
        ("method LPM hit rate / baseline LPM hit rate", ratio(m["lpm_hit_rate"] or 0, b["lpm_hit_rate"] or 0)),
        ("method new-prefix LPM rate / method LPM rate", ratio(m["new_prefix_lpm_hit_rate"] or 0, m["lpm_hit_rate"] or 0)),
        ("method % time saved by LPM / baseline", ratio(m["lpm_time_saved_pct"] or 0, b["lpm_time_saved_pct"] or 0)),
        ("baseline median call time / method", ratio(b["median_call_s"] or 0, m["median_call_s"] or 0)),
        ("method nodes / baseline nodes", ratio(m.get("nodes", 0), b.get("nodes", 0))),
        ("method snapshots / baseline snapshots", ratio(m.get("snapshots", 0), b.get("snapshots", 0))),
        ("method snapshot bytes / baseline", ratio(m.get("snapshot_bytes", 0), b.get("snapshot_bytes", 0))),
        ("preempted rounds / rounds", ratio(total("preempted"), len(rounds))),
        ("depth reached / graph depth", ratio(total("depth_reached"), total("graph_depth"))),
        ("contraction walltime / idle time", ratio(total("walltime_s"), total("idle_s"))),
        ("equal comparisons / comparisons", ratio(total("equal_comparisons"), total("comparisons"))),
        ("comparisons / similar pairs", ratio(total("comparisons"), total("similar_pairs"))),
        ("sandboxes built / nodes built", ratio(total("sandboxes_built"), total("nodes_built"))),
        ("sandboxes built / comparisons", ratio(total("sandboxes_built"), total("comparisons"))),
        ("pairs decided / similar pairs", ratio(total("pairs_decided"), total("similar_pairs"))),
    ]
    lines += [f"| {name} | {fmt(value)} |" for name, value in pairs]
    lines += ["", f"baseline totals: {json.dumps(b)}", "", f"method totals: {json.dumps(m)}", ""]
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--compare", action="store_true", help="two runs: baseline then method")
    args = parser.parse_args(argv)
    if args.compare:
        if len(args.runs) != 2:
            parser.error("--compare takes the baseline run, then the method run")
        print(compare(*(run_metrics(run) for run in args.runs)))
        return
    for run in args.runs:
        metrics = run_metrics(run)
        (run / "metrics.json").write_text(json.dumps(metrics, indent=1, default=list))
        markdown = to_markdown(metrics)
        (run / "metrics.md").write_text(markdown)
        print(markdown)


if __name__ == "__main__":
    main(sys.argv[1:])
