"""One trajectory's link to TVCache: an executor, the command history, and a
lock so a turn's parallel tool calls run one at a time on the same sandbox.

veRL's ToolAgentLoop creates and releases a tool instance around every single
call, so a trajectory's state cannot live in the tool. It lives here instead,
in a registry keyed by an id the tool stores in the trajectory's
`extra_fields`; TvcacheToolAgentLoop finishes the session (runs the task's
tests, closes the executor) once the trajectory ends.

Nothing here imports veRL, so it can be tested without it.
"""

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from tvclient import TvcacheToolExecutor

from tvcache_workloads.bash.sandbox import (
    NAMESPACE_SEPARATOR,
    SandboxConfig,
    TerminalBenchSandbox,
)
from tvcache_workloads.bash.tool_call import BashToolCall

SESSION_KEY = "tvcache_rollout"  # extra_fields key holding the session id
FAILED_PREFIX = "Failed to execute"


@dataclass(frozen=True)
class CacheSettings:
    """Where the cache server and container manager are, and how to run.

    Every field can come from the tool's YAML config or, if absent there,
    from the TVCACHE_* environment variable named in `from_config`."""

    cache_host: str = "127.0.0.1"
    cache_port: int = 7878
    manager_url: str = "http://127.0.0.1:9901"
    namespace: str = ""  # cache task id = "<task><NAMESPACE_SEPARATOR><namespace>", so runs never share a trie
    command_timeout_sec: float = 120.0
    snapshot_mode: str = "cold"
    trace_dir: str = ""  # TraceRecorder JSONL per worker process; needs tvclient's observer hooks
    max_output_chars: int = 4000
    # Seconds of un-snapshotted runtime before a snapshot. None keeps tvclient's
    # adaptive policy (a multiple of the measured fork cost).
    snapshot_threshold: Optional[float] = None

    @staticmethod
    def from_config(config: dict) -> "CacheSettings":
        def get(name: str, env: str, cast=str):
            value = config.get(name)
            if value is None or value == "":
                value = os.environ.get(env)
            return None if value is None or value == "" else cast(value)

        values = {
            "cache_host": get("cache_host", "TVCACHE_HOST"),
            "cache_port": get("cache_port", "TVCACHE_PORT", int),
            "manager_url": get("manager_url", "TVCACHE_MANAGER_URL"),
            "namespace": get("namespace", "TVCACHE_NAMESPACE"),
            "command_timeout_sec": get("command_timeout_sec", "TVCACHE_COMMAND_TIMEOUT", float),
            "snapshot_mode": get("snapshot_mode", "TVCACHE_SNAPSHOT_MODE"),
            "trace_dir": get("trace_dir", "TVCACHE_TRACE_DIR"),
            "max_output_chars": get("max_output_chars", "TVCACHE_MAX_OUTPUT_CHARS", int),
            "snapshot_threshold": get("snapshot_threshold", "TVCACHE_SNAPSHOT_THRESHOLD", float),
        }
        return CacheSettings(**{k: v for k, v in values.items() if v is not None})

    def cache_task_id(self, task: str) -> str:
        return f"{task}{NAMESPACE_SEPARATOR}{self.namespace}" if self.namespace else task


_recorder = None


def _trace_recorder(trace_dir: str):
    """One TraceRecorder per worker process (all its rollouts, one file)."""
    global _recorder
    if _recorder is None:
        from tvverify.trace.recorder import (  # needs tvclient observer hooks
            BlobStore,
            TraceRecorder,
        )

        directory = Path(trace_dir)
        directory.mkdir(parents=True, exist_ok=True)
        _recorder = TraceRecorder(directory / f"trace-{os.getpid()}.jsonl", BlobStore(directory / "blobs"))
    return _recorder


@dataclass
class RolloutSession:
    task: str
    executor: TvcacheToolExecutor
    history: List[BashToolCall] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    failed_calls: int = 0

    async def run(self, command: str) -> str:
        """Runs one command after the trajectory's earlier ones, through the cache."""
        async with self.lock:
            self.history.append(BashToolCall(command))
            try:
                value = await self.executor.lookup_or_execute(list(self.history))
            except Exception as error:  # the executor already discarded the sandbox
                value = f"{FAILED_PREFIX}: {type(error).__name__}: {error}"
            if value.startswith(FAILED_PREFIX):
                self.failed_calls += 1
            return value

    async def finish(self) -> str:
        """Runs the task's tests on the final state (through the cache, like any
        call) and closes the executor. Returns the test output: JSON
        {"passed": bool}, or a "Failed to execute" message."""
        async with self.lock:
            try:
                return await self.executor.test(list(self.history))
            except Exception as error:
                return f"{FAILED_PREFIX}: {type(error).__name__}: {error}"
            finally:
                await self.executor.close()


def make_session(task: str, settings: CacheSettings) -> RolloutSession:
    sandbox = TerminalBenchSandbox.with_config(
        SandboxConfig(
            manager_url=settings.manager_url,
            command_timeout_sec=settings.command_timeout_sec,
            snapshot_mode=settings.snapshot_mode,
        )
    )
    extra = {"observer": _trace_recorder(settings.trace_dir)} if settings.trace_dir else {}
    if settings.snapshot_threshold is not None:
        extra.update(adaptive=False, fixed_threshold_seconds=settings.snapshot_threshold)
    executor = TvcacheToolExecutor(
        sandbox, BashToolCall, settings.cache_task_id(task),
        host=settings.cache_host, port=settings.cache_port, **extra,
    )
    # A unique id per trajectory (the default is a timestamp, which collides
    # across concurrent rollouts); set directly, since set_rollout_id() opens
    # a log file per rollout.
    executor.rollout_id = uuid.uuid4().hex
    return RolloutSession(task, executor)


class SessionRegistry:
    """Live sessions of this worker process, by id."""

    def __init__(self, factory: Callable[[str, CacheSettings], RolloutSession] = make_session):
        self.factory = factory
        self.sessions: Dict[str, RolloutSession] = {}

    def for_trajectory(self, extra_fields: dict, task: str, settings: CacheSettings) -> RolloutSession:
        key = extra_fields.get(SESSION_KEY)
        if key is None or key not in self.sessions:
            key = key or uuid.uuid4().hex
            extra_fields[SESSION_KEY] = key
            self.sessions[key] = self.factory(task, settings)
        return self.sessions[key]

    def pop(self, extra_fields: dict) -> Optional[RolloutSession]:
        key = extra_fields.get(SESSION_KEY)
        return self.sessions.pop(key, None) if key is not None else None


SESSIONS = SessionRegistry()
