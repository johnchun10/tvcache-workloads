"""Compares pilot runs (with and without the cache) from their run directories.

    python compare_runs.py $TVROOT/runs/<...>-train-nocache $TVROOT/runs/<...>-train-main

Reads logs/verl.log (per-step metrics), logs/manager.log (container requests)
and, when present, dumps/dump_*.json (the cache tree with hit counters). Prints
one summary table and a per-epoch reward table, and writes <run>/steps.csv
with every metric veRL logged per step. Standard library only; --plot <dir>
also writes reward_per_epoch.png, step_time_per_epoch.png and, for runs with
dumps, cache_hit_rate_over_time.png (needs matplotlib).
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from collections import Counter
from pathlib import Path
from statistics import mean

STEP_RE = re.compile(r"\bstep:(\d+) - (.*)$")
TRAIN_START_RE = re.compile(r"INFO:(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+:Total training steps")
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
    summary["_hit_rate_over_time"] = hit_rate_over_time(run, dumps)
    summary["_hits_per_epoch"] = hits_per_epoch(run, steps, summary["_hit_rate_over_time"], epochs)
    return summary


def training_start(run: Path) -> float:
    """Unix time of step 1: veRL's 'Total training steps' line, else tasks.parquet."""
    for line in (run / "logs" / "verl.log").read_text(errors="replace").splitlines():
        match = TRAIN_START_RE.search(line)
        if match:
            return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
    return (run / "tasks.parquet").stat().st_mtime


def interpolate(points: list, x: float) -> float:
    """Piecewise-linear value at x over [(x, y), ...]; 0 before the first point."""
    if not points or x <= points[0][0]:
        return 0.0 if not points or x < points[0][0] else points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y1
    return points[-1][1]


def hits_per_epoch(run: Path, steps: list, over_time: list, epochs: int) -> list:
    """Hits and hit rate within each epoch, from the dumps nearest its end.

    Epoch k ends at training start + the sum of timing_s/step through its last
    step; counters at that moment are interpolated between the two dumps
    around it (dumps are 10 min apart, an epoch is about 50 min)."""
    if not over_time:
        return []
    t0 = training_start(run)
    dump_t0 = (run / "tasks.parquet").stat().st_mtime if (run / "tasks.parquet").exists() else t0
    hits_pts = [(dump_t0 + r["minutes"] * 60, r["hits"]) for r in over_time]
    nodes_pts = [(dump_t0 + r["minutes"] * 60, r["nodes"]) for r in over_time]
    per = max(1, len(steps) // epochs)
    rows, elapsed, prev_hits, prev_nodes = [], 0.0, 0.0, 0.0
    for e in range(epochs):
        chunk = steps[e * per : (e + 1) * per] if e < epochs - 1 else steps[e * per :]
        if not chunk:
            break
        elapsed += sum(s.get("timing_s/step", 0) for s in chunk)
        hits = interpolate(hits_pts, t0 + elapsed)
        nodes = interpolate(nodes_pts, t0 + elapsed)
        d_hits, d_nodes = hits - prev_hits, nodes - prev_nodes
        rows.append({
            "epoch": e + 1,
            "hits": round(d_hits),
            "lookups": round(d_hits + d_nodes),
            "hit rate": d_hits / (d_hits + d_nodes) if d_hits + d_nodes > 0 else 0.0,
            "end (min)": elapsed / 60,
        })
        prev_hits, prev_nodes = hits, nodes
    return rows


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


def hit_rate_over_time(run: Path, dumps: list) -> list:
    """Cumulative and per-window hit rate at each dump; minutes since the run started."""
    if not dumps:
        return []
    marker = run / "tasks.parquet"  # written just before veRL starts
    t0 = marker.stat().st_mtime if marker.exists() else dumps[0].stat().st_mtime
    rows, prev_hits, prev_nodes = [], 0, 0
    for dump in dumps:
        stats = cache_stats(dump)
        hits, nodes = stats.get("hits", 0), stats.get("nodes", 0)
        lookups = hits + nodes
        d_hits, d_nodes = hits - prev_hits, nodes - prev_nodes
        rows.append({
            "minutes": (dump.stat().st_mtime - t0) / 60,
            "cumulative": hits / lookups if lookups else 0.0,
            "window": d_hits / (d_hits + d_nodes) if d_hits + d_nodes > 0 else float("nan"),
            "hits": hits,
            "nodes": nodes,
        })
        prev_hits, prev_nodes = hits, nodes
    return rows


def plot_hit_rate(summaries: list, out_dir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for summary in summaries:
        rows = summary["_hit_rate_over_time"]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
        xs = [r["minutes"] for r in rows]
        ax.plot(xs, [100 * r["cumulative"] for r in rows], color=SERIES_COLORS["cache"],
                linewidth=2, marker="o", markersize=5, label="Cumulative")
        ax.plot(xs, [100 * r["window"] for r in rows], color=SERIES_COLORS["nocache"],
                linewidth=2, linestyle="--", marker="o", markersize=5, label="Last 10 min")
        ax.set_title(f"Cache Hit Rate on {WORKLOAD_TITLE}", fontsize=13, pad=12)
        ax.set_xlabel("Minutes of training")
        ax.set_ylabel("Hit rate (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"cache_hit_rate_over_time{'' if len(summaries) == 1 else '-' + summary['run']}.png"
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


def series_label(run_name: str) -> str:
    return "No-cache" if run_name.endswith("nocache") else "Cache"


def plot_per_epoch(summaries: list, out_dir: Path, key: str, ylabel: str, title: str, filename: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for summary in summaries:
        label = series_label(summary["run"])
        color = SERIES_COLORS["nocache" if label == "No-cache" else "cache"]
        xs = [r["epoch"] for r in summary["_per_epoch"]]
        ys = [r[key] for r in summary["_per_epoch"]]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5, label=label)
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
        rows = summary["_hit_rate_over_time"]
        if rows:
            print()
            print(f"== cache hit rate over time: {summary['run']} ==")
            print("minutes  cumulative%  last-window%  hits  nodes")
            for r in rows:
                print(f"{r['minutes']:7.1f}  {100 * r['cumulative']:10.1f}  {100 * r['window']:12.1f}  {r['hits']:4d}  {r['nodes']:5d}")
    for summary in summaries:
        rows = summary["_hits_per_epoch"]
        if rows:
            print()
            print(f"== cache hits per epoch: {summary['run']} ==")
            print("epoch  hits  lookups  hit-rate%  ends-at-min")
            for r in rows:
                print(f"{r['epoch']:5d}  {r['hits']:4d}  {r['lookups']:7d}  {100 * r['hit rate']:9.1f}  {r['end (min)']:11.1f}")
    print()
    for run in args.runs:
        print(f"per-step metrics: {run / 'steps.csv'}")
    if args.plot:
        try:
            print("chart:", plot_per_epoch(summaries, args.plot, "reward", "Mean Reward", "Mean Reward", "reward_per_epoch.png"))
            print("chart:", plot_per_epoch(summaries, args.plot, "step time (s)", "Mean Step Time (s)", "Mean Step Time", "step_time_per_epoch.png"))
            for path in plot_hit_rate(summaries, args.plot) + plot_hits_per_epoch(summaries, args.plot):
                print("chart:", path)
        except ImportError:
            print("charts skipped: matplotlib is not installed (pip install matplotlib)", file=sys.stderr)


if __name__ == "__main__":
    main()
