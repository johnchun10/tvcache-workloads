"""Calibrates the state comparison for a run's tasks, before contraction relies on it.

    python -m tvcache_workloads.bash.state_check --tasks hello-world regex-log --out $RUN/state.json

Contraction merges two cached commands when their sandboxes compare equal
(TerminalBenchSandbox.equals, state.py). That only works if two sandboxes in
the *same* state compare equal, and files the container runtime writes on its
own (a hosts file, a socket directory) can break it. So, per task, this

  1. starts two fresh sandboxes and restores a third from a snapshot of the
     first: three sandboxes in the same state;
  2. lists every watched file in each and records the paths and exported
     variables that differ anyway. Those are noise, and the run's policy
     skips them for this task (TVCACHE_STATE_FILE);
  3. checks that the three now hash the same, and that writing one new file
     changes the hash.

A task that fails step 3, or has more noise than the probe can skip, is marked
not comparable: the run still trains on it, but it is never contracted.
Needs the container manager. Exits 1 if no task is comparable.
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from tvcache_workloads.bash.sandbox import DEFAULT_MANAGER_URL, SandboxConfig, TerminalBenchSandbox
from tvcache_workloads.bash.state import (
    MANIFEST_LIMIT,
    MAX_LEARNED_PATHS,
    ProbeError,
    WatchPolicy,
    listing_command,
    parse_probe,
    probe_command,
)
from tvcache_workloads.bash.tool_call import BashToolCall

MAX_ENTRIES = 30_000  # 100 pages; a bigger watched tree is not worth listing
MARKER_FILE = ".tvcache-state-check"


async def _run(sandbox: TerminalBenchSandbox, command: str) -> str:
    output = await sandbox.execute(BashToolCall(command))
    if output.startswith("Failed to execute"):
        raise ProbeError(output[:300])
    return output


async def full_listing(sandbox: TerminalBenchSandbox, policy: WatchPolicy) -> Dict[str, str]:
    """Every watched entry -> its content hash or type and mode, read a page
    at a time."""
    entries: Dict[str, str] = {}
    start = 1
    while start <= MAX_ENTRIES:
        output = await _run(sandbox, listing_command(policy, start))
        _, manifests = parse_probe(output, required=("page",))
        entries.update(manifests.get("fs", {}))
        lines = sum(1 for row in output.splitlines() if row.startswith("tvv-m fs "))
        if lines < MANIFEST_LIMIT:
            return entries
        start += MANIFEST_LIMIT
    raise ProbeError(f"more than {MAX_ENTRIES} watched entries")


async def strict_hashes_under(sandbox: TerminalBenchSandbox, policy: WatchPolicy) -> Dict[str, str]:
    hashes, _ = parse_probe(await _run(sandbox, probe_command(policy, manifest=False)), required=policy.strict)
    return {name: hashes[name] for name in policy.strict}


async def env_manifest(sandbox: TerminalBenchSandbox, policy: WatchPolicy) -> Dict[str, str]:
    _, manifests = parse_probe(await _run(sandbox, probe_command(policy, manifest=True)), required=policy.strict)
    return manifests.get("env", {})


def differing(maps: Sequence[Dict[str, str]]) -> List[str]:
    """Keys that are not present with the same value in every map."""
    keys = set().union(*maps)
    return sorted(k for k in keys if len({m.get(k) for m in maps}) > 1)


def noisy_paths(listings: Sequence[Dict[str, str]]) -> List[str]:
    """Paths that differ between the listings, without the ones under another
    noisy path (skipping a directory skips what is in it)."""
    paths = sorted({key.split(" -> ")[0] for key in differing(listings)})
    kept: List[str] = []
    for path in paths:
        if not any(path.startswith(parent.rstrip("/") + "/") for parent in kept):
            kept.append(path)
    return kept


async def calibrate_task(task: str, sandbox_class: type, policy: WatchPolicy, restore: bool = True) -> dict:
    """What state_check learned about one task (see the module docstring)."""
    started = time.perf_counter()
    result = {"paths": [], "env": [], "comparable": False, "reason": ""}
    live: List[TerminalBenchSandbox] = []
    snapshot: Optional[TerminalBenchSandbox] = None
    try:
        first, second = sandbox_class(task_name=task), sandbox_class(task_name=task)
        await asyncio.gather(first.start(), second.start())
        live += [first, second]
        if restore:
            snapshot = await first.fork()
            third = sandbox_class(task_name=task, env_id=snapshot.get_id())
            await third.start()
            live.append(third)

        paths = noisy_paths([await full_listing(s, policy) for s in live])
        variables = differing([await env_manifest(s, policy) for s in live])
        result.update(paths=paths, env=variables)
        if len(paths) > MAX_LEARNED_PATHS:
            result["reason"] = f"{len(paths)} paths differ between sandboxes in the same state, e.g. {paths[:5]}"
            return result

        learned = policy.with_learned(paths, variables)
        hashes = [await strict_hashes_under(s, learned) for s in live]
        unequal = sorted({name for h in hashes[1:] for name in h if h[name] != hashes[0][name]})
        if unequal:
            result["reason"] = f"sandboxes in the same state still differ in {unequal} after skipping the noise"
            return result

        marker = f"{learned.roots[0].rstrip('/')}/{MARKER_FILE}"
        await _run(first, f"mkdir -p '{learned.roots[0]}' && echo changed > '{marker}'")
        if await strict_hashes_under(first, learned) == hashes[0]:
            result["reason"] = f"writing {marker} did not change the state hash"
            return result
        result["comparable"] = True
        return result
    except Exception as error:  # a task that cannot be probed is simply not contracted
        result["reason"] = f"{type(error).__name__}: {error}"[:500]
        return result
    finally:
        result["seconds"] = round(time.perf_counter() - started, 1)
        for sandbox in live + ([snapshot] if snapshot is not None else []):
            try:
                await sandbox.stop()
            except Exception as error:
                print(f"{task}: could not stop {sandbox.get_id()}: {error}", flush=True)


async def calibrate(
    tasks: Sequence[str], sandbox_class: type, policy: WatchPolicy, parallel: int = 4, restore: bool = True
) -> Dict[str, dict]:
    limit = asyncio.Semaphore(parallel)

    async def one(task: str) -> Tuple[str, dict]:
        async with limit:
            result = await calibrate_task(task, sandbox_class, policy, restore)
        verdict = "comparable" if result["comparable"] else f"NOT comparable ({result['reason']})"
        print(
            f"{task}: {verdict}; skipping {len(result['paths'])} paths, {len(result['env'])} variables "
            f"[{result['seconds']} s]",
            flush=True,
        )
        return task, result

    return dict(await asyncio.gather(*(one(task) for task in tasks)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manager-url", default=os.environ.get("TVCACHE_MANAGER_URL", DEFAULT_MANAGER_URL))
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--no-restore", action="store_true", help="skip the snapshot/restore sandbox")
    args = parser.parse_args()

    policy = WatchPolicy.base_from_env()
    sandbox_class = TerminalBenchSandbox.with_config(SandboxConfig(manager_url=args.manager_url))
    tasks = asyncio.run(calibrate(args.tasks, sandbox_class, policy, args.parallel, not args.no_restore))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "policy": {"roots": policy.roots, "excludes": policy.excludes, "packages": policy.packages},
                "tasks": tasks,
            },
            indent=2,
        )
    )
    comparable = [t for t, r in tasks.items() if r["comparable"]]
    print(f"{len(comparable)}/{len(tasks)} tasks comparable; wrote {args.out}")
    if not comparable:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
