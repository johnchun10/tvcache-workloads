"""Prints the commands the model ran for a task, from a run's last cache dump.

    python dump_tree.py <run dir>              # lists the tasks in the dump
    python dump_tree.py <run dir> hello-world  # that task's tool-call tree

Each line is one cached command (indented by depth in the tree) with its hit
count and the first line of its output. Commands that graph contraction merged
into one node are listed together, the later ones marked `=`. The tree only exists for runs that
used the cache; the no-cache variant keeps no record of commands.
"""

import argparse
import json
import sys
from pathlib import Path

from tvcache_workloads.bash.cache_admin import node_children


def label(edge: str) -> str:
    try:
        call = json.loads(edge)
    except ValueError:
        return edge
    if call.get("TV_CACHE_TOOL_TYPE") == "TESTING TOOL":
        return "<run tests>"
    return call.get("command", edge)


def show(node: dict, depth: int, width: int) -> None:
    for calls, child in node_children(node):
        hits = child.get("number_of_cache_hits", 0)
        for index, edge in enumerate(sorted(calls)):
            command = label(edge).replace("\n", "\\n")
            value = str(calls[edge].get("value") or "").strip().splitlines()
            first = value[0] if value else ""
            print(f"{'  ' * depth}{'=' if index else '$'} {command[:width]}")
            print(f"{'  ' * depth}    [hits {hits}] {first[:width]}")
        show(child, depth + 1, width)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("task", nargs="?", help="task name (prefix match, e.g. hello-world)")
    parser.add_argument("--width", type=int, default=120, help="max characters per line")
    args = parser.parse_args()

    dumps = sorted((args.run / "dumps").glob("dump_*.json"))
    if not dumps:
        sys.exit(f"no dumps under {args.run}/dumps (the no-cache variant writes none)")
    dump = json.loads(dumps[-1].read_text())
    if args.task is None:
        for task_id, tcg in sorted(dump.items()):
            print(f"{tcg.get('number_of_nodes', '?'):>5} nodes  {task_id.split('::', 1)[0]}")
        return
    matches = [t for t in dump if t.split("::", 1)[0] == args.task or t.startswith(args.task)]
    if not matches:
        sys.exit(f"no task matching {args.task!r}; run without a task name to list them")
    for task_id in sorted(matches):
        print(f"== {task_id} ({dumps[-1].name}) ==")
        show(dump[task_id]["root"], 0, args.width)


if __name__ == "__main__":
    main()
