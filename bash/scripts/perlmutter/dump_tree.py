"""Prints the commands the model ran for a task, from a run's cache dumps.

    python dump_tree.py <run dir>                        # lists the tasks, with their size as each epoch ended
    python dump_tree.py <run dir> hello-world            # that task's tool-call tree at the end of the run
    python dump_tree.py <run dir> hello-world --epoch 2  # the tree as epoch 2 ended
    python dump_tree.py <run dir> --csv graphs.csv       # every node of every epoch's trees, for analysis

Each line is one cached command (indented by depth in the tree) with its hit
count and the first line of its output. Commands that graph contraction merged
into one node are listed together, the later ones marked `=`. The tree only exists for runs that
used the cache; the no-cache variant keeps no record of commands.

A training run saves every task's tree as each epoch ends
(dumps/epoch_NNN.json, epochs counted from 0; cache_sidecar.py writes them when
the fork bank is on). With --epoch, commands first run in that epoch are marked
`+` instead of `$`.

--csv writes one row per node per epoch:
  epoch, task      which tree, and as which epoch ended
  node, parent     ids made from the node's path of commands (parent is empty
                   under the root). The same node has the same id in every
                   epoch, unless contraction merged one of its ancestors.
  depth            1 for a rollout's first command
  command          the node's command (`<run tests>` for the test run)
  merged_calls     commands held by the node: above 1 after a contraction merge
  hits, lpm_hits   the server's counters; the fork bank's own lookups count too
  snapshot         the sandbox snapshot stored at the node, if any
  inserted_at      when the node was first cached (Unix seconds)
  first_epoch      the epoch that first ran it
  runtime_ms, output_bytes
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from tvcache_workloads.bash.cache_admin import node_children

COLUMNS = ["epoch", "task", "node", "parent", "depth", "command", "merged_calls", "hits", "lpm_hits",
           "snapshot", "inserted_at", "first_epoch", "runtime_ms", "output_bytes"]


def label(edge: str) -> str:
    try:
        call = json.loads(edge)
    except ValueError:
        return edge
    if call.get("TV_CACHE_TOOL_TYPE") == "TESTING TOOL":
        return "<run tests>"
    return call.get("command", edge)


def epoch_dumps(run: Path) -> Dict[int, Path]:
    return {int(path.stem.split("_")[1]): path for path in sorted((run / "dumps").glob("epoch_*.json"))}


def nodes_of(node: dict, path: Tuple[str, ...] = ()) -> Iterator[Tuple[Tuple[str, ...], Dict[str, dict], dict]]:
    """Every node under `node` as (its path of commands, its calls, the node).
    A merged node is named by the first of its calls in sorted order."""
    for calls, child in node_children(node):
        child_path = path + (min(calls) if calls else "",)
        yield child_path, calls, child
        yield from nodes_of(child, child_path)


def epoch_ends(dumps: Dict[int, dict]) -> Dict[int, int]:
    """For each epoch dump, when its newest node was cached: a node cached
    after epoch e's dump and by epoch e+1's was first run in epoch e+1."""
    return {
        epoch: max((node.get("inserted_at_seconds", 0) for graph in dump.values()
                    for _, _, node in nodes_of(graph["root"])), default=0)
        for epoch, dump in dumps.items()
    }


def first_epoch(inserted_at: int, ends: Dict[int, int]) -> Optional[int]:
    return next((epoch for epoch in sorted(ends) if inserted_at <= ends[epoch]), None)


def node_id(task: str, path: Tuple[str, ...]) -> str:
    return hashlib.sha1(json.dumps([task, *path]).encode()).hexdigest()[:12] if path else ""


def rows(dumps: Dict[int, dict]) -> Iterator[dict]:
    ends = epoch_ends(dumps)
    for epoch in sorted(dumps):
        for task_id, graph in sorted(dumps[epoch].items()):
            task = task_id.split("::", 1)[0]
            for path, calls, node in nodes_of(graph["root"]):
                call = calls.get(path[-1], {})
                inserted_at = node.get("inserted_at_seconds", 0)
                yield {
                    "epoch": epoch, "task": task, "node": node_id(task, path), "parent": node_id(task, path[:-1]),
                    "depth": len(path), "command": label(path[-1]), "merged_calls": len(calls),
                    "hits": node.get("number_of_cache_hits", 0), "lpm_hits": node.get("number_of_lpm_hits", 0),
                    "snapshot": node.get("sandbox_id") or "", "inserted_at": inserted_at,
                    "first_epoch": first_epoch(inserted_at, ends), "runtime_ms": call.get("runtime", 0),
                    "output_bytes": len(str(call.get("value") or "").encode()),
                }


def show(node: dict, depth: int, width: int, new_after: Optional[int] = None) -> None:
    for calls, child in node_children(node):
        hits = child.get("number_of_cache_hits", 0)
        new = new_after is not None and child.get("inserted_at_seconds", 0) > new_after
        for index, edge in enumerate(sorted(calls)):
            command = label(edge).replace("\n", "\\n")
            value = str(calls[edge].get("value") or "").strip().splitlines()
            first = value[0] if value else ""
            print(f"{'  ' * depth}{'=' if index else '+' if new else '$'} {command[:width]}")
            print(f"{'  ' * depth}    [hits {hits}] {first[:width]}")
        show(child, depth + 1, width, new_after)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("task", nargs="?", help="task name (prefix match, e.g. hello-world)")
    parser.add_argument("--epoch", type=int, help="the tree as this epoch ended (0 = the first)")
    parser.add_argument("--csv", type=Path, help="write every epoch's nodes to this file and exit")
    parser.add_argument("--width", type=int, default=120, help="max characters per line")
    args = parser.parse_args()

    epochs = epoch_dumps(args.run)
    if args.csv is not None or args.epoch is not None:
        if not epochs:
            sys.exit(f"no epoch dumps under {args.run}/dumps (they need the fork bank, and a run that finished an epoch)")
        if args.epoch is not None and args.epoch not in epochs:
            sys.exit(f"no dump for epoch {args.epoch}; this run has epochs {', '.join(map(str, sorted(epochs)))}")
    if args.csv is not None:
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            count = 0
            for row in rows({epoch: json.loads(path.read_text()) for epoch, path in epochs.items()}):
                writer.writerow(row)
                count += 1
        print(f"{args.csv}: {count} nodes over {len(epochs)} epochs")
        return

    new_after = None
    if args.epoch is not None:
        source = epochs[args.epoch]
        if args.epoch - 1 in epochs:
            previous = json.loads(epochs[args.epoch - 1].read_text())
            new_after = epoch_ends({args.epoch - 1: previous})[args.epoch - 1]
        elif args.epoch == min(epochs):
            new_after = 0  # the first epoch: everything is new
    else:
        dumps = sorted((args.run / "dumps").glob("dump_*.json"))
        if not dumps:
            sys.exit(f"no dumps under {args.run}/dumps (the no-cache variant writes none)")
        source = dumps[-1]
    dump = json.loads(source.read_text())
    if args.task is None:
        sizes = {epoch: json.loads(path.read_text()) for epoch, path in epochs.items()} if args.epoch is None else {}
        if sizes:
            print(f"{'nodes':>5}  as each epoch ended ({', '.join(map(str, sorted(sizes)))})")
        for task_id, tcg in sorted(dump.items()):
            per_epoch = "  ".join(f"{sizes[e].get(task_id, {}).get('number_of_nodes', 0):>4}" for e in sorted(sizes))
            print(f"{tcg.get('number_of_nodes', '?'):>5} nodes  {per_epoch}{'  ' if sizes else ''}{task_id.split('::', 1)[0]}")
        return
    matches = [t for t in dump if t.split("::", 1)[0] == args.task or t.startswith(args.task)]
    if not matches:
        sys.exit(f"no task matching {args.task!r}; run without a task name to list them")
    for task_id in sorted(matches):
        print(f"== {task_id} ({source.name}) ==")
        show(dump[task_id]["root"], 0, args.width, new_after)


if __name__ == "__main__":
    main()
