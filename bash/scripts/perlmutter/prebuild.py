"""Builds every task's image once, through the manager's own /start and /stop,
so training never waits on a build. Tasks whose build fails are listed (and
written to --out) so they can be fixed or left out of the dataset.

    python prebuild.py $TASKS_ROOT --manager-url http://127.0.0.1:9901 --out prebuild.json
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx


def build(manager_url: str, task: str, timeout: float) -> dict:
    started = time.perf_counter()
    try:
        with httpx.Client(base_url=manager_url, timeout=timeout) as client:
            response = client.post("/start", json={"task_name": task})
            response.raise_for_status()
            terminal = response.json()["id"]
            client.post("/stop", json={"id": terminal}).raise_for_status()
        return {"task": task, "ok": True, "seconds": round(time.perf_counter() - started, 1)}
    except Exception as error:
        return {"task": task, "ok": False, "seconds": round(time.perf_counter() - started, 1), "error": f"{type(error).__name__}: {error}"[:500]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tasks_root", type=Path)
    parser.add_argument("--manager-url", default="http://127.0.0.1:9901")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds per task (builds can be slow)")
    parser.add_argument("--out", type=Path, default=Path("prebuild.json"))
    args = parser.parse_args()

    tasks = sorted(p.parent.name for p in args.tasks_root.glob("*/task.yaml"))
    with ThreadPoolExecutor(args.parallel) as pool:
        results = []
        for result in pool.map(lambda t: build(args.manager_url, t, args.timeout), tasks):
            print(("ok    " if result["ok"] else "FAIL  ") + f"{result['task']} ({result['seconds']} s)" + ("" if result["ok"] else f": {result['error'][:200]}"), flush=True)
            results.append(result)
    args.out.write_text(json.dumps(results, indent=2))
    failed = [r["task"] for r in results if not r["ok"]]
    print(f"{len(tasks) - len(failed)}/{len(tasks)} built; failures: {failed or 'none'}")


if __name__ == "__main__":
    main()
