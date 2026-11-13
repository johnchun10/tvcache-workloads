"""Graph contraction beside a training run.

tvclient's Contractor merges cached commands that left the sandbox in the same
state. It may only work on a task while none of that task's rollouts is
running, and veRL has no "this task's rollouts are done" hook: rollouts run in
several worker processes. So one coordinator process owns the Contractor, and
rollouts tell it when they touch a task:

    rollout's first command   -> acquire(task): stop the task's contraction job
                                 and wait until it has applied its last merge
    rollout's tests finished  -> release(task): once no rollout holds the task
                                 for --grace seconds, start contracting it

    python -m tvcache_workloads.bash.contraction serve --threshold 0.9 --max-sandboxes 32 \\
        --events $RUN/contraction/events.jsonl
    python -m tvcache_workloads.bash.contraction summary $RUN/contraction/events.jsonl

A rollout that dies without releasing would pause its task forever, so a hold
expires after --lease seconds. Tasks the state check found not comparable
(TVCACHE_STATE_FILE, state_check.py) are never contracted.

Every decision is appended to the events file with a timestamp:
  pause / resume   a task's contraction stopped for rollouts / started
  groups           one sibling listing: how many similar pairs the server found
  apply            one group applied: how many classes were merged
  job_end          a job's totals (comparisons, merges, restores, replays)
`summary` adds them up, in total and over time.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import signal
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Optional

from tvcache_workloads.bash.sandbox import DEFAULT_MANAGER_URL, SandboxConfig, TerminalBenchSandbox, tb_task_name
from tvcache_workloads.bash.state import load_calibration
from tvcache_workloads.bash.tool_call import BashToolCall

DEFAULT_PORT = 7879
JOB_END = re.compile(
    r"Job for (?P<task>.+) ended at depth (?P<depth>\d+) with (?P<comparisons>\d+) comparisons, "
    r"(?P<merges>\d+) merges, (?P<restores>\d+) restores, (?P<replays>\d+) replayed calls and "
    r"(?P<unavailable>\d+) unavailable children"
)
logger = logging.getLogger(__name__)


class EventLog:
    """Appends one JSON object per line; safe to tail while the run goes on."""

    def __init__(self, path: Optional[Path], clock: Callable[[], float] = time.time):
        self.path, self.clock, self.started = path, clock, clock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, task: str = "", **fields: object) -> None:
        if self.path is None:
            return
        now = self.clock()
        record = {"time": round(now, 3), "elapsed": round(now - self.started, 3), "event": event, "task": task}
        with self.path.open("a") as handle:
            handle.write(json.dumps({**record, **fields}) + "\n")


class Coordinator:
    """Pauses a task's contraction while rollouts hold it (see the module docstring).

    `contractor` needs tvclient's Contractor interface: start(task_id),
    async stop(task_id), async close()."""

    def __init__(
        self,
        contractor: object,
        events: EventLog,
        grace_seconds: float = 2.0,
        lease_seconds: float = 5400.0,
        skip: FrozenSet[str] = frozenset(),
        clock: Callable[[], float] = time.monotonic,
    ):
        self.contractor, self.events = contractor, events
        self.grace_seconds, self.lease_seconds = grace_seconds, lease_seconds
        self.skip, self.clock = skip, clock
        self.holds: Dict[str, Dict[str, float]] = {}  # task -> hold id -> expiry
        self.pending: Dict[str, asyncio.Task] = {}  # task -> its delayed start
        self.running: set = set()  # tasks whose job we started and have not paused
        self.closing = False

    def held(self, task: str) -> bool:
        holds = self.holds.get(task, {})
        for hold in [h for h, expiry in holds.items() if expiry <= self.clock()]:
            del holds[hold]
        return bool(holds)

    async def acquire(self, task: str) -> str:
        hold = uuid.uuid4().hex
        self.holds.setdefault(task, {})[hold] = self.clock() + self.lease_seconds
        self._cancel_pending(task)
        if task in self.running:
            self.running.discard(task)
            self.events.write("pause", task)
        await self.contractor.stop(task)
        return hold

    async def release(self, task: str, hold: str) -> None:
        self.holds.get(task, {}).pop(hold, None)
        self.schedule(task)

    def schedule(self, task: str) -> None:
        """Starts task's contraction after the grace period, unless it is held,
        skipped, or already waiting to start."""
        if self.closing or self.held(task) or task in self.pending or tb_task_name(task) in self.skip:
            return
        self.pending[task] = asyncio.create_task(self._start_later(task))

    async def _start_later(self, task: str) -> None:
        try:
            await asyncio.sleep(self.grace_seconds)
            await self.contractor.stop(task)  # a job still stopping would make start() a no-op
            if not self.closing and not self.held(task):
                self.contractor.start(task)
                self.running.add(task)
                self.events.write("resume", task)
        finally:
            if self.pending.get(task) is asyncio.current_task():
                del self.pending[task]

    def _cancel_pending(self, task: str) -> None:
        pending = self.pending.pop(task, None)
        if pending is not None:
            pending.cancel()

    def sweep(self) -> None:
        """Schedules tasks whose holds have all expired (their rollouts died)."""
        for task in list(self.holds):
            if not self.held(task) and task not in self.running:
                del self.holds[task]
                self.schedule(task)

    async def close(self, final_wait: float = 0.0) -> None:
        """Lets running jobs finish for up to final_wait seconds, then stops everything."""
        self.closing = True
        for task in list(self.pending):
            self._cancel_pending(task)
        jobs = [job for job, _ in getattr(self.contractor, "_jobs", {}).values()]
        if final_wait > 0 and jobs:
            await asyncio.wait(jobs, timeout=final_wait)
        await self.contractor.close()
        self.running.clear()

    # --- the control port: one JSON object per line, each answered with one ---
    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while line := await reader.readline():
                try:
                    request = json.loads(line)
                    op, task = request["op"], request.get("task", "")
                    if op == "acquire":
                        reply = {"ok": True, "hold": await self.acquire(task)}
                    elif op == "release":
                        await self.release(task, request.get("hold", ""))
                        reply = {"ok": True}
                    elif op == "status":
                        reply = {"ok": True, "held": sorted(t for t in self.holds if self.held(t)), "running": sorted(self.running)}
                    else:
                        reply = {"ok": False, "error": f"unknown op {op!r}"}
                except Exception as error:  # one bad request must not take the port down
                    logger.exception("control request failed")
                    reply = {"ok": False, "error": f"{type(error).__name__}: {error}"}
                writer.write((json.dumps(reply) + "\n").encode())
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()


class ContractionGate:
    """A rollout's side of the control port. If nothing listens there, no
    Contractor is running either (it lives in the coordinator), so the rollout
    just goes on."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, timeout: float = 900.0):
        self.host, self.port, self.timeout = host, port, timeout

    async def _call(self, request: dict) -> Optional[dict]:
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(self.host, self.port), 10)
        except (ConnectionRefusedError, FileNotFoundError):
            return None
        try:
            writer.write((json.dumps(request) + "\n").encode())
            await writer.drain()
            line = await asyncio.wait_for(reader.readline(), self.timeout)
        finally:
            writer.close()
        if not line:
            raise ConnectionError("the contraction coordinator closed the connection")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(f"contraction coordinator: {reply.get('error')}")
        return reply

    async def acquire(self, task: str) -> Optional[str]:
        """Returns once task's contraction is stopped. The hold id, or None
        when no coordinator is running."""
        reply = await self._call({"op": "acquire", "task": task})
        return None if reply is None else reply["hold"]

    async def release(self, task: str, hold: str) -> None:
        await self._call({"op": "release", "task": task, "hold": hold})


def instrument(events: EventLog) -> Callable[[], None]:
    """Records what the Contractor's jobs ask the server and what they end with.

    tvclient reports a job's totals only in a log line and the pair counts not
    at all, so this wraps the two client calls a job makes and reads that line.
    Returns a function that removes the wrapping again."""
    import tvclient.contraction
    from tvclient import TVCacheClient

    list_groups, apply = TVCacheClient.list_sibling_groups, TVCacheClient.apply_contraction

    async def counted_list(self, task_id, depth, threshold):
        listing = await list_groups(self, task_id, depth, threshold)
        events.write(
            "groups", task_id, depth=depth, groups=len(listing.groups),
            children=sum(len(g.children) for g in listing.groups),
            pairs=sum(len(g.pairs) for g in listing.groups),
        )
        return listing

    async def counted_apply(self, task_id, parent_path, classes, settled):
        result = await apply(self, task_id, parent_path, classes, settled)
        events.write(
            "apply", task_id, depth=len(parent_path) + 1, classes=len(classes), settled=len(settled),
            merged_nodes=sum(len(c) - 1 for c in classes if len(c) > 1),
            merged_classes=result.merged_classes, skipped_calls=result.skipped_calls,
        )
        return result

    TVCacheClient.list_sibling_groups, TVCacheClient.apply_contraction = counted_list, counted_apply

    class JobEnd(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            match = JOB_END.search(record.getMessage())
            if match:
                numbers = {k: int(v) for k, v in match.groupdict().items() if k != "task"}
                events.write("job_end", match["task"], **numbers)

    job_logger, handler = logging.getLogger(tvclient.contraction.__name__), JobEnd()
    job_logger.setLevel(logging.INFO)
    job_logger.addHandler(handler)

    def undo() -> None:
        TVCacheClient.list_sibling_groups, TVCacheClient.apply_contraction = list_groups, apply
        job_logger.removeHandler(handler)

    return undo


def summarize(events_path: Path, buckets: int = 20) -> dict:
    """Totals of an events file, plus cumulative counts over time."""
    records = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    totals = {"similar_pairs_listed": 0, "comparisons": 0, "merged_classes": 0, "merged_nodes": 0,
              "restores": 0, "replays": 0, "unavailable": 0, "jobs": 0, "pauses": 0}
    per_task: Dict[str, Dict[str, int]] = {}
    span = max((r["elapsed"] for r in records), default=0.0)
    width = span / buckets if span else 1.0
    timeline = [{"until_seconds": round(width * (i + 1), 1), "similar_pairs_listed": 0, "merged_classes": 0,
                 "merged_nodes": 0, "comparisons": 0} for i in range(buckets)]

    def add(record: dict, key: str, amount: int) -> None:
        totals[key] += amount
        task = per_task.setdefault(tb_task_name(record["task"]), {})
        task[key] = task.get(key, 0) + amount
        if key in timeline[0]:
            for bucket in timeline[min(int(record["elapsed"] / width), buckets - 1):]:
                bucket[key] += amount

    for record in records:
        if record["event"] == "groups":
            add(record, "similar_pairs_listed", record["pairs"])
        elif record["event"] == "apply":
            add(record, "merged_classes", record["merged_classes"])
            add(record, "merged_nodes", record["merged_nodes"])
        elif record["event"] == "job_end":
            add(record, "jobs", 1)
            for key in ("comparisons", "restores", "replays", "unavailable"):
                add(record, key, record[key])
        elif record["event"] == "pause":
            add(record, "pauses", 1)
    return {"seconds": round(span, 1), "totals": totals, "per_task": per_task, "over_time": timeline}


async def serve(args: argparse.Namespace) -> None:
    from tvclient import Contractor

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    events = EventLog(args.events)
    instrument(events)
    sandbox = TerminalBenchSandbox.with_config(
        SandboxConfig(manager_url=args.manager_url, snapshot_mode=args.snapshot_mode)
    )
    contractor = Contractor(
        sandbox, BashToolCall, args.threshold, args.max_sandboxes,
        host=args.cache_host, port=args.cache_port, timeout=args.cache_timeout,
    )
    skip = frozenset(task for task, learned in load_calibration().items() if not learned.get("comparable", True))
    coordinator = Coordinator(contractor, events, args.grace, args.lease, skip)
    events.write("serve", threshold=args.threshold, max_sandboxes=args.max_sandboxes, skip=sorted(skip))

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    server = await asyncio.start_server(coordinator.handle, args.host, args.port)
    logger.info("contraction coordinator on %s:%s; never contracting %s", args.host, args.port, sorted(skip) or "nothing")
    try:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=60)
            except asyncio.TimeoutError:
                coordinator.sweep()
    finally:
        server.close()
        await coordinator.close(args.final_wait)
        events.write("closed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("serve", help="run the coordinator until SIGTERM")
    run.add_argument("--threshold", type=float, required=True, help="similarity threshold, -1 to 1")
    run.add_argument("--max-sandboxes", type=int, default=32, help="live sandboxes all contraction jobs share")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=int(os.environ.get("TVCACHE_CONTRACTION_PORT") or DEFAULT_PORT))
    run.add_argument("--cache-host", default=os.environ.get("TVCACHE_HOST", "127.0.0.1"))
    run.add_argument("--cache-port", type=int, default=int(os.environ.get("TVCACHE_PORT") or 7878))
    run.add_argument("--cache-timeout", type=float, default=30.0)
    run.add_argument("--manager-url", default=os.environ.get("TVCACHE_MANAGER_URL", DEFAULT_MANAGER_URL))
    run.add_argument("--snapshot-mode", default=os.environ.get("TVCACHE_SNAPSHOT_MODE") or "cold")
    run.add_argument("--grace", type=float, default=2.0, help="seconds a task must be free before contracting")
    run.add_argument("--lease", type=float, default=5400.0, help="seconds after which a rollout's hold expires")
    run.add_argument("--final-wait", type=float, default=0.0, help="seconds to let jobs finish on shutdown")
    run.add_argument("--events", type=Path, default=None)
    report = commands.add_parser("summary", help="add up an events file")
    report.add_argument("events", type=Path)
    report.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if args.command == "serve":
        asyncio.run(serve(args))
        return
    summary = summarize(args.events)
    if args.json is not None:
        args.json.write_text(json.dumps(summary, indent=2))
    totals = summary["totals"]
    print(
        f"contraction over {summary['seconds']} s: {totals['jobs']} jobs, {totals['similar_pairs_listed']} similar "
        f"pairs listed, {totals['comparisons']} comparisons, {totals['merged_classes']} merges "
        f"({totals['merged_nodes']} nodes removed), {totals['restores']} restores, {totals['replays']} replayed "
        f"calls, {totals['unavailable']} unavailable children, {totals['pauses']} pauses for rollouts"
    )


if __name__ == "__main__":
    main()
