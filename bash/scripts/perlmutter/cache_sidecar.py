"""Runs beside a training job's cache server, on the server node:

  - saves DUMP_ALL to <dump_dir>/dump_<n>.json every --interval seconds, at
    each epoch boundary (epoch_<e>.json, the trie as epoch e ended) and once
    more on SIGTERM/SIGINT (the final trie), for stats.py and tvverify
  - is the server's one sandbox drainer (tvclient SandboxManager): snapshots
    the eviction loop marks reclaimable are deleted, so they do not pile up
  - with --control-port: serves the trainer and rollout workers
    (bash/control.py) from one fork bank (TVCache's prewarming), and logs
    every step boundary, warm-up and hand-out to <dump_dir>/../control.jsonl.
    Graph contraction is not here: its coordinator (bash/contraction.py)
    pauses and resumes it per rollout.

    python cache_sidecar.py --dump-dir $RUN/dumps --port 7878 --manager-url http://127.0.0.1:9901 \\
        --control-port 7880 --top-k 10 --roots-per-task 4
"""

import argparse
import asyncio
import json
import os
import signal
import time
from pathlib import Path
from typing import Dict, Optional

from tvclient import ForkBank, SandboxManager, TVCacheClient

from tvcache_workloads.bash.sandbox import NAMESPACE_SEPARATOR, SandboxConfig, TerminalBenchSandbox


async def dump_to(port: int, host: str, path: Path) -> dict:
    async with TVCacheClient(host, port, timeout=120) as client:
        data = await client.dump_all()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.rename(path)  # never leave a half-written dump behind
    nodes = sum(t.get("number_of_nodes", 0) for t in data.values())
    print(f"{time.strftime('%H:%M:%S')} {path.name}: {len(data)} tasks, {nodes} nodes", flush=True)
    return data


def graph_depth(task_dump: Optional[dict]) -> int:
    """Depth of a task's trie in a DUMP_ALL entry (root = 0)."""
    def depth(node: dict) -> int:
        children = node.get("children") or []
        if isinstance(children, dict):
            children = list(children.values())
        return 1 + max((depth(child) for child in children), default=0)

    if not task_dump or "root" not in task_dump:
        return 0
    return depth(task_dump["root"]) - 1


class Control:
    """The control server's state: fork bank and step log."""

    def __init__(self, args: argparse.Namespace, log_path: Path):
        self.args = args
        self.namespace = os.environ.get("TVCACHE_NAMESPACE", "")
        self.log_path = log_path
        config = SandboxConfig(manager_url=args.manager_url)
        self.bank = ForkBank(
            TerminalBenchSandbox.with_config(config),
            warm_roots_per_task=args.roots_per_task,
            host=args.host,
            port=args.port,
        )
        self.warm_lock = asyncio.Lock()
        self.warm_tasks: set = set()
        self.last_end: Dict[str, float] = {}  # task -> when its rollouts last ended
        self.epoch: Optional[int] = None
        self.dump_dir = Path(args.dump_dir)

    def cache_id(self, task: str) -> str:
        if not self.namespace or NAMESPACE_SEPARATOR in task:
            return task
        return f"{task}{NAMESPACE_SEPARATOR}{self.namespace}"

    def log(self, event: str, **fields) -> None:
        record = {"t": time.time(), "event": event, **fields}
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    async def _warm(self, tasks, keep) -> None:
        async with self.warm_lock:
            started = time.time()
            await self.bank.set_tasks([self.cache_id(t) for t in tasks], keep=[self.cache_id(t) for t in keep])
            try:
                await self.bank.warm_up(top_k=self.args.top_k)
            except Exception as error:
                print(f"warm_up failed: {error!r}", flush=True)
            self.log("warm", tasks=tasks, seconds=time.time() - started,
                     shelf=self.bank.live_count(), **self.bank.stats)

    async def handle(self, request: dict) -> dict:
        op = request.get("op")
        if op == "take":
            env_id = self.bank.get_forked_env(request["task"], request["parent"])
            if env_id is not None:
                self.log("take", task=request["task"], parent=request["parent"], env_id=env_id)
            return {"env_id": env_id}
        if op == "warm":
            job = self._warm(request.get("tasks", []), request.get("keep", []))
            if request.get("wait"):
                await job
            else:
                task = asyncio.create_task(job)
                self.warm_tasks.add(task)
                task.add_done_callback(self.warm_tasks.discard)
            return {"ok": True}
        if op == "begin":
            step, epoch, tasks = request["step"], request["epoch"], request["tasks"]
            if self.epoch is not None and epoch != self.epoch:
                await self._epoch_dump(self.epoch)
            self.epoch = epoch
            now = time.time()
            for task in dict.fromkeys(tasks):
                cache_id = self.cache_id(task)
                if cache_id in self.last_end:  # the window contraction had for this task
                    self.log("idle", task=cache_id, step=step, epoch=epoch, idle_s=now - self.last_end[cache_id])
            self.log("begin", step=step, epoch=epoch, tasks=tasks)
            return {"ok": True}
        if op == "end":
            step, epoch, tasks = request["step"], request["epoch"], request["tasks"]
            now = time.time()
            self.log("end", step=step, epoch=epoch, tasks=tasks, last=request.get("last", False))
            # The graph each task's contraction now starts on (stats.py: depth
            # reached / graph depth).
            async with TVCacheClient(self.args.host, self.args.port, timeout=120) as client:
                dump = await client.dump_all()
            for task in dict.fromkeys(tasks):
                cache_id = self.cache_id(task)
                self.last_end[cache_id] = now
                self.log("graph", task=cache_id, step=step, epoch=epoch,
                         graph_depth=graph_depth(dump.get(cache_id)),
                         graph_nodes=(dump.get(cache_id) or {}).get("number_of_nodes"))
            if request.get("last"):
                await self._epoch_dump(epoch)
            return {"ok": True}
        if op == "stats":
            return {"shelf": self.bank.live_count(), **self.bank.stats}
        return {"error": f"unknown op {op!r}"}

    async def _epoch_dump(self, epoch: int) -> None:
        try:
            await dump_to(self.args.port, self.args.host, self.dump_dir / f"epoch_{epoch:03d}.json")
        except OSError as error:
            print(f"epoch dump skipped: {error}", flush=True)

    async def serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            try:
                reply = await self.handle(json.loads(line))
            except Exception as error:
                print(f"control request {line[:200]!r} failed: {error!r}", flush=True)
                reply = {"error": repr(error)}
            writer.write((json.dumps(reply) + "\n").encode())
            await writer.drain()
        finally:
            writer.close()

    async def close(self) -> None:
        for task in list(self.warm_tasks):
            await asyncio.gather(task, return_exceptions=True)
        await self.bank.stop()
        self.log("closed", **self.bank.stats)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7878)
    parser.add_argument("--manager-url", default="http://127.0.0.1:9901")
    parser.add_argument("--interval", type=float, default=600.0, help="seconds between dumps")
    parser.add_argument("--drain-interval", type=float, default=10.0)
    parser.add_argument("--control-port", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10, help="hot snapshots pre-forked per warm-up")
    parser.add_argument("--roots-per-task", type=int, default=4, help="warm roots per task (= rollouts)")
    args = parser.parse_args()
    args.dump_dir.mkdir(parents=True, exist_ok=True)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    sandbox = TerminalBenchSandbox.with_config(SandboxConfig(manager_url=args.manager_url))
    manager = SandboxManager(sandbox, host=args.host, port=args.port)
    drainer = asyncio.create_task(manager.run_drain_loop(args.drain_interval, stop))
    control = server = None
    if args.control_port is not None:
        control = Control(args, args.dump_dir.parent / "control.jsonl")
        server = await asyncio.start_server(control.serve, "0.0.0.0", args.control_port)
        control.log("start", top_k=args.top_k, roots_per_task=args.roots_per_task)
        print(f"control server on :{args.control_port}", flush=True)
    n = 0
    try:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=args.interval)
            except asyncio.TimeoutError:
                pass
            try:
                await dump_to(args.port, args.host, args.dump_dir / f"dump_{n:03d}.json")
                n += 1
            except OSError as error:  # server not up yet, or already gone
                print(f"dump skipped: {error}", flush=True)
    finally:
        if server is not None:
            server.close()
        if control is not None:
            await control.close()
        await drainer
        await manager.close()


if __name__ == "__main__":
    asyncio.run(main())
