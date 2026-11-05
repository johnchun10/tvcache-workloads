"""`tvverify check` and `tvverify check-graph`: correctness checks on your own workload.

    python -m tvverify check --adapter my_pkg.my_workload:MyAdapter
    python -m tvverify check-graph --adapter my_pkg.my_workload:MyAdapter --dump dump.json

check suites:
  contract  your Sandbox behaves as the cache assumes (needs no server)
  cache     rollouts through the cache return what direct execution returns
  replay    every cached node matches a fresh-root replay (C1, C2a, C3)

check-graph runs the replay checks on a cache trie saved with DUMP_ALL (for
example after each epoch of a training run), on a budget of nodes per task.

Findings are bugs, noise (fresh replays of the same path disagree with each
other, so the tool is nondeterministic) or inconclusive (the replay itself
could not run). Only bugs and contract failures set a non-zero exit code.
"""

import asyncio
import importlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tvverify.adapters.base import WorkloadAdapter
from tvverify.model.graph import GraphExport, export_task_graph, from_dumpall
from tvverify.replay.engine import ReplayEngine
from tvverify.testing.exact_vs_nocache import run_rollouts_vs_nocache
from tvverify.testing.sandbox_contract import run_contract
from tvverify.testing.server import RustServer
from tvverify.validators.findings import Finding
from tvverify.validators.offline import INCONCLUSIVE_CODES, OfflineValidator


def load_adapter(spec: str) -> WorkloadAdapter:
    """`package.module:ClassName` -> an adapter instance."""
    module_name, _, class_name = spec.partition(":")
    if not class_name:
        raise SystemExit(f"--adapter must look like package.module:ClassName, got {spec!r}")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)()


async def _contract(adapter: WorkloadAdapter, seed: int) -> int:
    results = await run_contract(adapter, seed)
    for result in results:
        print(result)
    failed = [r for r in results if not r.ok]
    print(f"contract: {len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


async def _classify(
    adapter, engine: ReplayEngine, findings: List[Finding], replays: int
) -> Tuple[List[Finding], List[Finding], List[Finding]]:
    """Splits findings into (bugs, noise, inconclusive).

    Inconclusive: the replay could not run, so it says nothing about the
    cache. Noise: a C1/C3 output mismatch where fresh replays of the same path
    disagree with each other (the tool is nondeterministic, so the cache
    cannot be blamed); the same for state and snapshot differences along a
    path that replays inconsistently. Anything else is a bug."""
    bugs: List[Finding] = []
    noise: List[Finding] = []
    inconclusive: List[Finding] = []

    async def disagrees(path: List[str], key: str) -> bool:
        outputs = [await engine.leaf_output(path, key) for _ in range(replays)]
        compare = adapter.comparator_for(key).compare
        return not all(compare(outputs[0], o, key=key).equal for o in outputs[1:])

    for finding in findings:
        if finding.code in INCONCLUSIVE_CODES:
            inconclusive.append(finding)
            continue
        if not finding.path or replays < 2 or finding.criterion not in ("C1", "C2a", "C3"):
            bugs.append(finding)
            continue
        path = list(finding.path)
        if finding.code == "output_mismatch":
            # the call whose cached value was wrong
            is_noise = await disagrees(path[:-1], path[-1])
        else:
            # a state or snapshot difference: noise if any call on the path
            # gives different outputs on different fresh replays
            is_noise = False
            for i in range(len(path)):
                if await disagrees(path[:i], path[i]):
                    is_noise = True
                    break
        (noise if is_noise else bugs).append(finding)
    return bugs, noise, inconclusive


def select_nodes(graph: GraphExport, budget: Optional[int], seed: int) -> List[str]:
    """The nodes to replay: all of them, or with a budget, the most-served half
    (cache hits plus resumes from their snapshot) and a seeded random sample
    of the rest."""
    nodes = [n for n in graph.nodes if n != graph.root_id]
    if budget is None or budget >= len(nodes):
        return nodes
    by_use = sorted(nodes, key=lambda n: (-(graph.nodes[n].hits + graph.nodes[n].lpm_hits), n))
    top = [n for n in by_use[: (budget + 1) // 2] if graph.nodes[n].hits + graph.nodes[n].lpm_hits > 0]
    chosen = set(top)
    rest = [n for n in nodes if n not in chosen]
    return top + random.Random(seed).sample(rest, budget - len(top))


async def _replay(
    adapter: WorkloadAdapter,
    task_id: str,
    graph: GraphExport,
    *,
    replays: int,
    nodes: Optional[List[str]] = None,
    check_snapshots: bool = True,
    quiescent: bool = True,
) -> dict:
    # memoize=False: real sandboxes are not assumed deterministic.
    engine = ReplayEngine(adapter, task_id, memoize=False)
    validator = OfflineValidator(adapter, engine, check_snapshots=check_snapshots)
    report = await validator.run(graph, nodes=nodes, quiescent=quiescent)
    bugs, noise, inconclusive = await _classify(adapter, engine, report.findings, replays)
    replayed = report.checked.get("nodes_replayed", 0)
    print(f"replay {task_id}: {'PASS' if not bugs else f'FAIL ({len(bugs)} bugs)'}, {len(noise)} noise, "
          f"{len(inconclusive)} inconclusive, {replayed}/{len(graph.nodes) - 1} nodes replayed")
    for f in bugs[:10]:
        print("   BUG  ", str(f)[:300])
    for f in noise[:5]:
        print("   noise", str(f)[:300])
    for f in inconclusive[:5]:
        print("   inconclusive", str(f)[:300])
    return {
        "task": task_id,
        "nodes_total": len(graph.nodes) - 1,
        "nodes_replayed": replayed,
        "snapshots_checked": check_snapshots,
        "executions": engine.stats.executions,
        "bugs": [asdict(f) for f in bugs],
        "noise": [asdict(f) for f in noise],
        "inconclusive": [asdict(f) for f in inconclusive],
    }


async def _delete_snapshots(adapter: WorkloadAdapter, task_id: str, graph: GraphExport) -> int:
    """Stops every snapshot the cache still references. The test server goes
    away after the run, so for real sandboxes these would otherwise leak
    (committed container images, for example). Returns how many failed."""
    handles = {n.sandbox_id for n in graph.nodes.values() if n.sandbox_id} | set(graph.reclaimable)
    failures = 0
    for handle in sorted(handles):
        try:
            await adapter.sandbox_class(task_name=task_id, env_id=handle).stop()
        except Exception:
            failures += 1
    return failures


async def _cache_and_replay(adapter: WorkloadAdapter, args, results: List[dict]) -> int:
    status = 0
    rng = random.Random(args.seed)
    with RustServer() as server:
        for task in adapter.tasks():
            # The replay suite checks the trie these rollouts build, so they
            # run for either suite.
            rollouts = adapter.sample_rollouts(task.task_id, rng, args.rollouts)
            mismatches = await run_rollouts_vs_nocache(server.port, adapter, task.task_id, rollouts)
            graph = await export_task_graph(server.port, task.task_id, adapter, task.root_image_id)
            result: Dict[str, object] = {"task": task.task_id}
            try:
                if args.suite in ("cache", "all"):
                    print(f"cache  {task.task_id}: {'PASS' if not mismatches else f'FAIL ({len(mismatches)} mismatches)'}")
                    for m in mismatches[:10]:
                        print("   ", str(m)[:300])
                    result["cache_mismatches"] = [str(m) for m in mismatches]
                    status |= bool(mismatches)
                if args.suite in ("replay", "all"):
                    nodes = select_nodes(graph, args.max_replay_nodes, args.seed)
                    replay = await _replay(adapter, task.task_id, graph, replays=args.replays, nodes=nodes)
                    result.update(replay)
                    status |= bool(replay["bugs"])
            finally:
                failed = await _delete_snapshots(adapter, task.task_id, graph)
                if failed:
                    print(f"warning: {failed} cached snapshots of {task.task_id} could not be deleted")
                result["snapshot_delete_failures"] = failed
                results.append(result)
    return status


def _write_json(path: Optional[str], command: str, args, results: List[dict], status: int) -> None:
    if not path:
        return
    out = {"schema": "tvverify-result/v1", "command": command, "adapter": args.adapter, "status": status, "tasks": results}
    for key in ("dump", "max_replay_nodes", "seed", "replays"):
        if hasattr(args, key):
            out[key] = getattr(args, key)
    Path(path).write_text(json.dumps(out, indent=2))


def main(args) -> int:
    adapter = load_adapter(args.adapter)
    results: List[dict] = []

    async def go() -> int:
        status = 0
        if args.suite in ("contract", "all"):
            status |= await _contract(adapter, args.seed)
            if status and args.suite == "all":
                print("skipping cache and replay suites: fix the contract failures first")
                return status
        if args.suite in ("cache", "replay", "all"):
            status |= await _cache_and_replay(adapter, args, results)
        return status

    status = asyncio.run(go())
    _write_json(args.json, "check", args, results, status)
    return status


def main_graph(args) -> int:
    """check-graph: replay checks on a saved DUMP_ALL (JSON: task id -> trie)."""
    adapter = load_adapter(args.adapter)
    dump = json.loads(Path(args.dump).read_text())
    tasks = args.task or sorted(dump)
    missing = [t for t in tasks if t not in dump]
    if missing:
        raise SystemExit(f"tasks not in the dump: {missing}")
    results: List[dict] = []

    async def go() -> int:
        status = 0
        for task_id in tasks:
            graph = from_dumpall({task_id: dump[task_id]}, adapter)[task_id]
            nodes = select_nodes(graph, args.max_replay_nodes, args.seed)
            # A dump taken during a run can hold leases (not quiescent), and
            # its snapshots may be gone by now (--no-snapshots).
            result = await _replay(
                adapter, task_id, graph, replays=args.replays, nodes=nodes,
                check_snapshots=not args.no_snapshots, quiescent=False,
            )
            results.append(result)
            status |= bool(result["bugs"])
        return status

    status = asyncio.run(go())
    _write_json(args.json, "check-graph", args, results, status)
    return status
