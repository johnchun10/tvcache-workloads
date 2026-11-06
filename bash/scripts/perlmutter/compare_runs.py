"""Compares pilot runs (with and without the cache) from their run directories.

    python compare_runs.py $TVROOT/runs/<...>-train-nocache $TVROOT/runs/<...>-train-main

Reads logs/verl.log (per-step metrics), logs/manager.log (container requests)
and, when present, dumps/dump_*.json (the cache tree with hit counters). Prints
one summary table and a per-epoch reward table, and writes <run>/steps.csv
with every metric veRL logged per step. Standard library only.
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

STEP_RE = re.compile(r"\bstep:(\d+) - (.*)$")
REQUEST_RE = re.compile(r'"(?:POST|GET) /(\w+)')
KEY_METRICS = {
    "timing_s/step": "step time (s)",
    "timing_s/gen": "rollout time (s)",
    "critic/score/mean": "reward",
    "response_length/mean": "response len",
}
EXTRA_KEYWORDS = ("passed", "tool_calls", "failed_calls")


def parse_steps(verl_log: Path) -> list:
    """One dict per training step: {"step": n, <metric>: value, ...}."""
    steps = {}
    for line in verl_log.read_text(errors="replace").splitlines():
        match = STEP_RE.search(line)
        if not match:
            continue
        row = {"step": int(match.group(1))}
        for item in match.group(2).split(" - "):
            key, sep, value = item.rpartition(":")
            if not sep:
                continue
            try:
                row[key.strip()] = float(value)
            except ValueError:
                row[key.strip()] = value.strip()
        steps[row["step"]] = row  # the last line for a step wins
    return [steps[n] for n in sorted(steps)]


def count_requests(manager_log: Path) -> Counter:
    counts = Counter()
    if manager_log.exists():
        for line in manager_log.read_text(errors="replace").splitlines():
            match = REQUEST_RE.search(line)
            if match:
                counts[match.group(1)] += 1
    return counts


def cache_stats(dump_path: Path) -> dict:
    """Totals over every task's tree in a DumpAll file."""
    dump = json.loads(dump_path.read_text())
    totals = Counter()

    def visit(node: dict) -> None:
        for child in node.get("children", {}).values():
            totals["nodes"] += 1
            totals["hits"] += child.get("number_of_cache_hits", 0)
            totals["lpm_hits"] += child.get("number_of_lpm_hits", 0)
            totals["runtime_ms"] += child.get("runtime", 0)
            visit(child)

    for tcg in dump.values():
        totals["tasks"] += 1
        totals["sandboxes"] += tcg.get("number_of_sandboxes", 0)
        visit(tcg["root"])
    return dict(totals)


def summarize(run: Path, epochs: int) -> dict:
    steps = parse_steps(run / "logs" / "verl.log")
    if not steps:
        sys.exit(f"{run}: no step lines in logs/verl.log")
    write_csv(run / "steps.csv", steps)
    requests = count_requests(run / "logs" / "manager.log")
    dumps = sorted((run / "dumps").glob("dump_*.json")) if (run / "dumps").is_dir() else []
    summary = {
        "run": run.name,
        "steps": len(steps),
        "train time (min)": sum(s.get("timing_s/step", 0) for s in steps) / 60,
    }
    for key, label in KEY_METRICS.items():
        values = [s[key] for s in steps if isinstance(s.get(key), float)]
        if values:
            summary[f"mean {label}"] = mean(values)
    for key in sorted({k for s in steps for k in s if any(w in k for w in EXTRA_KEYWORDS)}):
        values = [s[key] for s in steps if isinstance(s.get(key), float)]
        if values:
            summary[f"mean {key}"] = mean(values)
    for endpoint in ("start", "execute", "test", "snapshot", "restore"):
        summary[f"manager /{endpoint}"] = requests.get(endpoint, 0)
    if dumps:
        stats = cache_stats(dumps[-1])
        summary["cache nodes"] = stats.get("nodes", 0)
        summary["cache hits"] = stats.get("hits", 0)
        summary["cache lpm hits"] = stats.get("lpm_hits", 0)
        lookups = stats.get("hits", 0) + stats.get("nodes", 0)  # every node was one miss
        summary["cache hit rate"] = stats.get("hits", 0) / lookups if lookups else 0.0
        summary["cache snapshots"] = stats.get("sandboxes", 0)
    summary["_per_epoch"] = per_epoch(steps, epochs)
    return summary


def per_epoch(steps: list, epochs: int) -> list:
    """Mean reward and step time per epoch (steps split evenly over epochs)."""
    per = max(1, len(steps) // epochs)
    rows = []
    for e in range(epochs):
        chunk = steps[e * per : (e + 1) * per] if e < epochs - 1 else steps[e * per :]
        if not chunk:
            break
        rewards = [s["critic/score/mean"] for s in chunk if isinstance(s.get("critic/score/mean"), float)]
        times = [s["timing_s/step"] for s in chunk if isinstance(s.get("timing_s/step"), float)]
        rows.append({
            "epoch": e + 1,
            "reward": mean(rewards) if rewards else float("nan"),
            "step time (s)": mean(times) if times else float("nan"),
        })
    return rows


def write_csv(path: Path, steps: list) -> None:
    keys = ["step"] + sorted({k for s in steps for k in s if k != "step"})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(steps)


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 100 else f"{value:.1f}"
    return str(value)


def print_table(rows: list, first: str) -> None:
    keys = [first] + [k for k in rows[0] if k != first and not k.startswith("_")]
    for row in rows[1:]:
        keys += [k for k in row if k not in keys and not k.startswith("_")]
    width = max(len(k) for k in keys)
    cols = [max(len(fmt(r.get(k, ""))) for r in rows) for k in keys]
    for k in keys:
        cells = [fmt(r.get(k, "")).rjust(w) for r, w in zip(rows, cols)]
        print(f"{k.ljust(width)}  " + "  ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="run directories")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    summaries = [summarize(run, args.epochs) for run in args.runs]
    print("== summary ==")
    print_table(summaries, "run")
    print()
    print("== per epoch (reward / step time) ==")
    header = "epoch".ljust(6) + "".join(s["run"].rjust(36) for s in summaries)
    print(header)
    n_epochs = max(len(s["_per_epoch"]) for s in summaries)
    for i in range(n_epochs):
        cells = []
        for s in summaries:
            if i < len(s["_per_epoch"]):
                row = s["_per_epoch"][i]
                cells.append(f"{row['reward']:.3f} / {row['step time (s)']:.1f}s".rjust(36))
            else:
                cells.append("".rjust(36))
        print(str(i + 1).ljust(6) + "".join(cells))
    print()
    for run in args.runs:
        print(f"per-step metrics: {run / 'steps.csv'}")


if __name__ == "__main__":
    main()
