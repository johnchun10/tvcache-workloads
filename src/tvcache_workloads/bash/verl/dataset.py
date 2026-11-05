"""Builds the veRL training parquet from Terminal-Bench task directories.

    python -m tvcache_workloads.bash.verl.dataset \\
        ../container-manager-public/tasks_easy_network_skip train.parquet

One row per task: the system prompt plus the task's instruction (task.yaml),
the task id in extra_info (and as the bash tool's create_kwargs), and no
ground truth: the reward comes from running the task's tests.
"""

import argparse
from pathlib import Path
from typing import List

import yaml

DATA_SOURCE = "terminal_bench"
TOOL_NAME = "bash"
SYSTEM_PROMPT = (
    "You are an agent working in a Linux terminal inside a container. Complete the user's task "
    "by calling the `bash` tool, one shell command per call; each call returns the command's "
    "terminal output. Files, the working directory and exported variables persist between calls. "
    "When the task is done, reply without calling a tool; the task's tests then run on the final state."
)


def task_rows(tasks_dir: Path, exclude: frozenset = frozenset()) -> List[dict]:
    rows = []
    for task_yaml in sorted(Path(tasks_dir).glob("*/task.yaml")):
        task = task_yaml.parent.name
        if task in exclude:
            continue
        instruction = yaml.safe_load(task_yaml.read_text())["instruction"].strip()
        rows.append({
            "data_source": DATA_SOURCE,
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
            ],
            "ability": "terminal",
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {
                "task_id": task,
                "need_tools_kwargs": True,
                "tools_kwargs": {TOOL_NAME: {"create_kwargs": {"task_id": task}}},
            },
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tasks_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--exclude", nargs="*", default=[], help="task names to leave out")
    args = parser.parse_args()
    rows = task_rows(args.tasks_dir, frozenset(args.exclude))
    if not rows:
        raise SystemExit(f"no */task.yaml under {args.tasks_dir}")
    import pandas as pd  # in veRL's environment

    pd.DataFrame(rows).to_parquet(args.output)
    print(f"wrote {len(rows)} tasks to {args.output}")


if __name__ == "__main__":
    main()
