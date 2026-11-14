"""Records what the cache did, as JSONL events (schema trace/v1).

Attach a TraceRecorder as the `observer` of a TvcacheToolExecutor (needs a
tvclient with observer hooks; the rest of tvverify does not). Values are
kept as sha256 hashes plus (optionally) the full text in a content-addressed
blob directory, so traces stay small and values stay recoverable.

Event types (all carry task, rollout, idx = lookup index within the rollout):
  lookup    path, hit, lpm_depth, sandbox_id
  resume    source (fork|live|fresh), depth, env_id, from_bank
  executed  depth, key, value_sha, runtime_s, failed, digest?
  snapshot  depth, sandbox_id, digest?
  insert    path, value_sha, sandbox_id, value_inserted, sandbox_inserted
  return    source (hit|exec), value_sha
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

try:
    from tvclient.observer import ExecutionObserver, LookupContext
except ImportError as error:  # an unmodified tvclient has no observer hooks
    raise ImportError(
        "TraceRecorder needs a tvclient whose executor takes observer= (tvclient.observer)"
    ) from error

SCHEMA = "trace/v1"


def value_sha(value: Optional[str]) -> Optional[str]:
    return None if value is None else hashlib.sha256(value.encode()).hexdigest()


class BlobStore:
    """Content-addressed store for full values."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def put(self, value: str) -> str:
        sha = value_sha(value)
        target = self.directory / sha[:2] / sha
        if not target.exists():
            target.parent.mkdir(exist_ok=True)
            target.write_text(value)
        return sha

    def get(self, sha: str) -> str:
        return (self.directory / sha[:2] / sha).read_text()


class TraceRecorder(ExecutionObserver):
    def __init__(
        self,
        jsonl_path: Optional[Path] = None,
        blobs: Optional[BlobStore] = None,
        digest_every: int = 0,
        clock: Callable[[], float] = time.time,
    ):
        """digest_every: take a state digest after every Nth executed call
        (0 = never, 1 = always); snapshots are always digested when > 0."""
        self.events: List[dict] = []
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.blobs = blobs
        self.digest_every = digest_every
        self.clock = clock
        self._executed = 0
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self.jsonl_path.write_text("")

    def _emit(self, ctx: LookupContext, type_: str, **fields: object) -> None:
        event = {
            "v": SCHEMA,
            "type": type_,
            "t": self.clock(),
            "task": ctx.task_id,
            "rollout": ctx.rollout_id,
            "idx": ctx.call_index,
            **fields,
        }
        self.events.append(event)
        if self.jsonl_path:
            with self.jsonl_path.open("a") as handle:
                handle.write(json.dumps(event) + "\n")

    def _sha(self, value: Optional[str]) -> Optional[str]:
        if value is not None and self.blobs is not None:
            return self.blobs.put(value)
        return value_sha(value)

    def wants_digest(self, ctx: LookupContext, depth: int) -> bool:
        if self.digest_every <= 0:
            return False
        self._executed += 1
        return self._executed % self.digest_every == 0

    async def on_lookup(self, ctx, found) -> None:
        self._emit(
            ctx,
            "lookup",
            path=ctx.path,
            hit=found.value is not None,
            lpm_depth=len(found.sandbox_path) if found.sandbox_path else None,
            sandbox_id=found.sandbox_id,
        )

    async def on_resume(self, ctx, source, depth, env_id, from_bank=False) -> None:
        self._emit(ctx, "resume", source=source, depth=depth, env_id=env_id, from_bank=from_bank)

    async def on_executed(self, ctx, depth, call, value, runtime_seconds, failed, digest) -> None:
        self._emit(
            ctx,
            "executed",
            depth=depth,
            key=call.serialize(),
            value_sha=self._sha(value),
            runtime_s=runtime_seconds,
            failed=failed,
            digest=digest,
        )

    async def on_snapshot(self, ctx, depth, sandbox_id, digest) -> None:
        self._emit(ctx, "snapshot", depth=depth, sandbox_id=sandbox_id, digest=digest)

    async def on_insert(self, ctx, path, value, sandbox_id, result) -> None:
        self._emit(
            ctx,
            "insert",
            path=path,
            value_sha=self._sha(value),
            sandbox_id=sandbox_id,
            value_inserted=result.value_inserted,
            sandbox_inserted=result.sandbox_inserted,
        )

    async def on_return(self, ctx, value, source) -> None:
        self._emit(ctx, "return", source=source, value_sha=self._sha(value))


def read_trace(path: Path) -> List[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def group_by_call(events: List[dict]) -> Dict[tuple, List[dict]]:
    """(task, rollout, idx) -> that lookup's events, in order."""
    grouped: Dict[tuple, List[dict]] = {}
    for event in events:
        grouped.setdefault((event["task"], event["rollout"], event["idx"]), []).append(event)
    return grouped
