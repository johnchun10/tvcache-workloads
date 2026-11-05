"""Runs beside a training job's cache server:

  - saves DUMP_ALL to <dump_dir>/dump_<n>.json every --interval seconds and
    once more on SIGTERM/SIGINT (the final trie), for `tvverify check-graph`
  - is the server's one sandbox drainer (tvclient SandboxManager): snapshots
    the eviction loop marks reclaimable are deleted, so they do not pile up

    python cache_sidecar.py --dump-dir $RUN/dumps --port 7878 --manager-url http://127.0.0.1:9901
"""

import argparse
import asyncio
import json
import signal
import time
from pathlib import Path

from tvclient import SandboxManager, TVCacheClient

from tvcache_workloads.bash.sandbox import SandboxConfig, TerminalBenchSandbox


async def dump(port: int, host: str, directory: Path, n: int) -> Path:
    async with TVCacheClient(host, port) as client:
        data = await client.dump_all()
    path = directory / f"dump_{n:03d}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.rename(path)  # never leave a half-written dump behind
    nodes = sum(t.get("number_of_nodes", 0) for t in data.values())
    print(f"{time.strftime('%H:%M:%S')} {path.name}: {len(data)} tasks, {nodes} nodes", flush=True)
    return path


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7878)
    parser.add_argument("--manager-url", default="http://127.0.0.1:9901")
    parser.add_argument("--interval", type=float, default=600.0, help="seconds between dumps")
    parser.add_argument("--drain-interval", type=float, default=10.0)
    args = parser.parse_args()
    args.dump_dir.mkdir(parents=True, exist_ok=True)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    sandbox = TerminalBenchSandbox.with_config(SandboxConfig(manager_url=args.manager_url))
    manager = SandboxManager(sandbox, host=args.host, port=args.port)
    drainer = asyncio.create_task(manager.run_drain_loop(args.drain_interval, stop))
    n = 0
    try:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=args.interval)
            except asyncio.TimeoutError:
                pass
            try:
                await dump(args.port, args.host, args.dump_dir, n)
                n += 1
            except OSError as error:  # server not up yet, or already gone
                print(f"dump skipped: {error}", flush=True)
    finally:
        await drainer
        await manager.close()


if __name__ == "__main__":
    asyncio.run(main())
