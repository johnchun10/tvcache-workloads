"""Compares pilot runs (with and without the cache) from their run directories.

    python compare_runs.py $TVROOT/runs/<...>-none $TVROOT/runs/<...>-tvcache $TVROOT/runs/<...>-contraction-t0.9-minilm

Reads logs/verl.log (per-step metrics), logs/manager.log (container requests)
and, when present, dumps/dump_*.json (the size of the cache tree). Prints one
summary table and a per-epoch reward table, and writes <run>/steps.csv with
every metric veRL logged per step. Needs tvcache-workloads installed; --plot
<dir> also writes reward_per_epoch.png, step_time_per_epoch.png and, for runs
with metrics, cache_hit_rate_per_epoch.png and cache_hits_per_epoch.png (needs
matplotlib).

Each run's own report is <run>/report.md. This script puts several runs side
by side. Hit rates are not computed here: they come from the run's
metrics.json, which verl/stats.py writes next to report.md (it is the one
place the testing plan's metrics are computed). The cache server's own hit
counters are not used: the fork bank's lookups count as hits there. Given a
run with the cache and a run with contraction, it also prints the testing
plan's ratio table (the same one as stats.py --compare).
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from tvcache_workloads.bash.cache_admin import node_children
from tvcache_workloads.bash.verl import stats as run_stats

REQUEST_RE = re.compile(r'"(?:POST|GET) /(\w+)')
KEY_METRICS = {
    "timing_s/step": "step time (s)",
    "timing_s/gen": "rollout time (s)",
    "critic/score/mean": "reward",
    "response_length/mean": "response len",
}
EXTRA_KEYWORDS = ("passed", "tool_calls", "failed_calls")
# Series colors, same as the SmallBank (SQL) chart: blue no-cache, red cache.
# Validated with the dataviz palette checker (CVD dE 27.6, normal 31.8).
SERIES_COLORS = {"nocache": "#4a86e8", "cache": "#cc4125"}
WORKLOAD_TITLE = "Terminal-Bench Workload (Bash)"


parse_steps = run_stats.parse_steps  # veRL's per-step log lines


def count_requests(manager_log: Path) -> Counter:
    counts = Counter()
    if manager_log.exists():
        for line in manager_log.read_text(errors="replace").splitlines():
            match = REQUEST_RE.search(line)
            if match:
                counts[match.group(1)] += 1
    return counts


def cache_stats(dump_path: Path) -> dict:
    """Size of every task's tree in a DumpAll file."""
    dump = json.loads(dump_path.read_text())
    totals = Counter()

    def visit(node: dict) -> None:
        for calls, child in node_children(node):
            totals["nodes"] += 1
            totals["calls"] += len(calls)  # more than one on a node that contraction merged
            totals["runtime_ms"] += sum(entry.get("runtime", 0) for entry in calls.values())
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
        summary["cache snapshots"] = stats.get("sandboxes", 0)
        summary["calls merged away"] = stats.get("calls", 0) - stats.get("nodes", 0)
    metrics = load_metrics(run)
    if metrics and metrics.get("tasks"):
        totals = run_stats.totals(metrics)
        summary["lookups"] = sum(r.get("lookups", 0) for e in metrics["tasks"].values() for r in e.values())
        for key, label in (("cache_hit_rate", "cache hit rate"), ("new_seq_hit_rate", "new-sequence hit rate"),
                           ("lpm_hit_rate", "lpm hit rate")):
            if totals.get(key) is not None:
                summary[label] = totals[key]
    contraction = run / "contraction" / "summary.json"
    if contraction.exists():
        totals = json.loads(contraction.read_text())["totals"]
        summary["similar pairs listed"] = totals["similar_pairs_listed"]
        summary["contraction comparisons"] = totals["comparisons"]
        summary["contraction merges"] = totals["merged_classes"]
    peak = peak_memory(run / "logs" / "memory.csv")
    if peak:
        summary["peak memory (GB)"], summary["peak containers"] = peak
    summary["_label"] = series_label(run)
    summary["_per_epoch"] = per_epoch(steps, epochs)
    summary["_cached"] = bool(dumps)
    summary["_cache_mode"] = run_config(run).get("CACHE", "")
    summary["_metrics"] = metrics
    summary["_hits_per_epoch"] = hits_per_epoch(metrics)
    return summary


def load_metrics(run: Path):
    """The run's metrics from verl/stats.py: metrics.json, or computed from the
    trace when the file is not there yet. None for a run without a trace."""
    path = run / "metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    if (run / "trace").is_dir() and any((run / "trace").glob("trace-*.jsonl")):
        return json.loads(json.dumps(run_stats.run_metrics(run), default=list))  # same shape as the file
    return None


def hits_per_epoch(metrics) -> list:
    """Lookups, hits and hit rate per epoch over all tasks, from metrics.json.
    Epochs are numbered from 1 here, as in the reward table."""
    if not metrics:
        return []
    per = defaultdict(lambda: [0, 0.0])  # epoch -> [lookups, hits]
    for epochs in metrics.get("tasks", {}).values():
        for epoch, row in epochs.items():
            lookups = row.get("lookups", 0)
            per[int(epoch)][0] += lookups
            per[int(epoch)][1] += (row.get("cache_hit_rate") or 0.0) * lookups
    return [
        {"epoch": epoch + 1, "hits": round(hits), "lookups": lookups, "hit rate": hits / lookups if lookups else 0.0}
        for epoch, (lookups, hits) in sorted(per.items())
        if lookups
    ]


def plot_hits_per_epoch(summaries: list, out_dir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for summary in summaries:
        rows = summary["_hits_per_epoch"]
        if not rows:
            continue
        suffix = "" if len(summaries) == 1 else "-" + summary["run"]
        xs = [r["epoch"] for r in rows]
        for key, ylabel, title, name in (
            ("hit rate", "Hit rate (%)", "Cache Hit Rate per Epoch", "cache_hit_rate_per_epoch"),
            ("hits", "Cache hits", "Cache Hits per Epoch", "cache_hits_per_epoch"),
        ):
            ys = [100 * r[key] if key == "hit rate" else r[key] for r in rows]
            fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
            ax.plot(xs, ys, color=SERIES_COLORS["cache"], linewidth=2, marker="o", markersize=5)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.1f}" if key == "hit rate" else f"{y:.0f}", (x, y), xytext=(0, 7),
                            textcoords="offset points", ha="center", fontsize=9, color="#444444")
            ax.set_title(f"{title} on {WORKLOAD_TITLE}", fontsize=13, pad=12)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_xticks(xs)
            ax.set_ylim(0, max(ys) * 1.25 if max(ys) > 0 else 1)
            ax.grid(True, color="#e5e5e5", linewidth=0.8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            fig.tight_layout()
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{name}{suffix}.png"
            fig.savefig(path)
            plt.close(fig)
            paths.append(path)
    return paths


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


run_config = run_stats.run_config  # the settings in <run>/config.env


def series_label(run: Path) -> str:
    config = run_config(run)
    cache = config.get("CACHE") or ("none" if run.name.endswith("nocache") else "tvcache")
    if cache == "none":
        return "No-cache"
    if cache == "contraction":
        return f"Contraction ({config.get('THRESHOLD', '?')}, {config.get('EMBEDDER', '?')})"
    return "Cache"


def peak_memory(memory_csv: Path):
    """(peak job memory in GB, most containers at once) from logs/memory.csv."""
    if not memory_csv.exists():
        return None
    rows = list(csv.DictReader(memory_csv.open()))
    memory = [float(r["cgroup_mem_gb"]) for r in rows if r.get("cgroup_mem_gb") not in (None, "", "NA")]
    containers = [int(r["containers"]) for r in rows if (r.get("containers") or "").strip().isdigit()]
    return (max(memory, default=0.0), max(containers, default=0)) if rows else None


def plot_per_epoch(summaries: list, out_dir: Path, key: str, ylabel: str, title: str, filename: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    # The two validated colors stay: blue for no cache, red for every cached
    # run. Several cached runs (contraction settings) differ by line style.
    styles, cached = ["-", "--", ":", "-."], 0
    for summary in summaries:
        label = summary["_label"]
        color = SERIES_COLORS["nocache" if label == "No-cache" else "cache"]
        style = "-"
        if label != "No-cache":
            style, cached = styles[cached % len(styles)], cached + 1
        xs = [r["epoch"] for r in summary["_per_epoch"]]
        ys = [r[key] for r in summary["_per_epoch"]]
        ax.plot(xs, ys, color=color, linewidth=2, linestyle=style, marker="o", markersize=5, label=label)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color="#444444")
    ax.set_title(f"{title} on {WORKLOAD_TITLE}", fontsize=13, pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted({r["epoch"] for s in summaries for r in s["_per_epoch"]}))
    if key == "reward":
        ax.set_ylim(0, max(0.8, max(r[key] for s in summaries for r in s["_per_epoch"]) * 1.15))
    else:
        ax.set_ylim(bottom=0)
    ax.grid(True, color="#e5e5e5", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return path


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
    parser.add_argument("--plot", type=Path, metavar="DIR", help="write per-epoch charts (PNG) into DIR")
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
    for summary in summaries:
        rows = summary["_hits_per_epoch"]
        if rows:
            print()
            print(f"== cache hits per epoch: {summary['run']} (same numbers as its report.md) ==")
            print("epoch   hits  lookups  hit-rate%")
            for r in rows:
                print(f"{r['epoch']:5d}  {r['hits']:5d}  {r['lookups']:7d}  {100 * r['hit rate']:9.1f}")
        elif summary["_cached"]:
            print()
            print(f"{summary['run']}: no hit rates (no trace; they need tvverify on, see its report.md)")
    baseline = next((s for s in summaries if s["_cache_mode"] == "tvcache" and s["_metrics"]), None)
    for summary in summaries:
        if baseline and summary["_cache_mode"] == "contraction" and summary["_metrics"]:
            print()
            print(f"== {summary['run']} (method) against {baseline['run']} (baseline) ==")
            print(run_stats.compare(baseline["_metrics"], summary["_metrics"]))
    print()
    for run in args.runs:
        print(f"per-step metrics: {run / 'steps.csv'}")
    if args.plot:
        try:
            print("chart:", plot_per_epoch(summaries, args.plot, "reward", "Mean Reward", "Mean Reward", "reward_per_epoch.png"))
            print("chart:", plot_per_epoch(summaries, args.plot, "step time (s)", "Mean Step Time (s)", "Mean Step Time", "step_time_per_epoch.png"))
            for path in plot_hits_per_epoch(summaries, args.plot):
                print("chart:", path)
        except ImportError:
            print("charts skipped: matplotlib is not installed (pip install matplotlib)", file=sys.stderr)


if __name__ == "__main__":
    main()
