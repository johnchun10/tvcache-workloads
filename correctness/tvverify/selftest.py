"""`tvverify selftest`: run the framework's own checks against a fresh server."""

import asyncio
import sys
from typing import Callable, List

from tvverify.adapters.mock_fs import ToyFsSandbox
from tvverify.testing.exact_vs_nocache import run_exact_vs_nocache, run_value_snapshot_coherence
from tvverify.testing.ref_vs_server import run_ref_vs_server
from tvverify.testing.server import RustServer


def run_exact_suite(seeds: int, verbose: bool = True) -> int:
    failures: List[str] = []
    with RustServer() as server:
        port = server.port

        def stage(name: str, fn: Callable[[], List[str]]) -> None:
            found = [str(f) for f in fn()]
            failures.extend(f"[{name}] {f}" for f in found)
            if verbose:
                print(f"{name:<34} {'PASS' if not found else f'FAIL ({len(found)} problems)'}")

        stage("P1 reference model vs server",
              lambda: [p for s in range(seeds) for p in asyncio.run(run_ref_vs_server(port, s))])
        ToyFsSandbox.reset()
        stage("P2 exact cache vs no cache",
              lambda: [m for s in range(seeds) for m in asyncio.run(run_exact_vs_nocache(port, s))])
        stage("P2 ... concurrent rollouts",
              lambda: [m for s in range(seeds) for m in asyncio.run(run_exact_vs_nocache(port, s + 10_000, concurrent=True))])
        stage("P2b value/snapshot coherence",
              lambda: asyncio.run(run_value_snapshot_coherence(port, trials=max(5, seeds // 10))))

    for failure in failures[:20]:
        print("  ", failure[:300])
    if verbose:
        print("selftest", "PASSED" if not failures else f"FAILED ({len(failures)} problems)")
    return 1 if failures else 0


def main(argv: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="tvverify")
    sub = parser.add_subparsers(dest="command", required=True)
    selftest = sub.add_parser("selftest", help="check the cache and the framework on mock environments")
    selftest.add_argument("--seeds", type=int, default=100)
    check = sub.add_parser("check", help="run the correctness checks against your own workload")
    check.add_argument("--adapter", required=True, help="package.module:ClassName of your WorkloadAdapter")
    check.add_argument("--suite", choices=["contract", "cache", "replay", "all"], default="all")
    check.add_argument("--rollouts", type=int, default=6)
    check.add_argument("--seed", type=int, default=0)
    check.add_argument("--replays", type=int, default=2, help="fresh replays used to tell noise from bugs (1 disables)")
    check.add_argument("--max-replay-nodes", type=int, default=None, help="replay at most this many nodes per task (default: all)")
    check.add_argument("--json", default=None, help="also write the results to this JSON file")
    graph = sub.add_parser("check-graph", help="replay-check a cache trie saved with DUMP_ALL")
    graph.add_argument("--adapter", required=True, help="package.module:ClassName of your WorkloadAdapter")
    graph.add_argument("--dump", required=True, help="DUMP_ALL saved as JSON (task id -> trie)")
    graph.add_argument("--task", action="append", help="check only this task (repeatable; default: every task in the dump)")
    graph.add_argument("--max-replay-nodes", type=int, default=20, help="nodes replayed per task: the most-served half plus a random sample")
    graph.add_argument("--no-snapshots", action="store_true", help="the dump's snapshots no longer exist: skip the snapshot checks (C2a)")
    graph.add_argument("--seed", type=int, default=0)
    graph.add_argument("--replays", type=int, default=2, help="fresh replays used to tell noise from bugs (1 disables)")
    graph.add_argument("--json", default=None, help="also write the results to this JSON file")
    report = sub.add_parser("report", help="combine --json results into a markdown report")
    report.add_argument("results", nargs="+")
    args = parser.parse_args(argv)
    if args.command == "check":
        from tvverify.check import main as check_main

        return check_main(args)
    if args.command == "check-graph":
        from tvverify.check import main_graph

        return main_graph(args)
    if args.command == "report":
        from tvverify.report import render

        print(render(args.results), end="")
        return 0
    if args.command == "selftest":
        return run_exact_suite(args.seeds)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
