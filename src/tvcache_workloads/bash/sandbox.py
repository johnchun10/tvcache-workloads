"""TerminalBenchSandbox: tvclient's Sandbox over container-manager-public.

Snapshot handling
-----------------
tvclient stores only *cold, immutable, reusable* snapshot handles in the cache
and restores them with env_class(task, env_id=handle).start(). Two ways to
provide that, chosen with SandboxConfig.snapshot_mode:

"cold" (default; needs the manager's /snapshot, /restore, /delete_snapshot,
added on the cold-snapshots branch of our container-manager-public fork):

  fork()            /snapshot(live) -> committed images, nothing running;
                    handle "tb-snap:<snapshot id>"
  start() on handle /restore(snapshot) -> a new, independent live terminal
  stop() on handle  /delete_snapshot

"parked" (works with the stock manager, which only has /fork):

  fork()            /fork(live) -> a running child that is *parked*: nothing
                    ever executes in it. Handle "tb-parked:<child id>"
  start() on handle /fork(parked) -> a new, independent live child
  stop() on handle  /stop

The handle prefix, not the current mode, decides how a handle is restored or
deleted, so a cache holding either kind is handled correctly. execute()
refuses handles, and start() refuses to "restore" a plain live id.

Both modes preserve the same state (see bash/docs/snapshot-design.md): each
container's filesystem plus, with the manager's ENABLE_CACHING=1, cwd and
*exported* variables. Processes, unexported variables, functions, aliases,
and shell options are lost. "parked" additionally costs a running container
per snapshot and a commit on every restore.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Optional

from tvclient import Sandbox, TestToolCall, ToolCall

from tvcache_workloads.bash.manager_client import (
    ManagerClient,
    ManagerError,
    ManagerTimeouts,
)
from tvcache_workloads.bash.observation import Markers, ObservationError, extract_output
from tvcache_workloads.bash.tool_call import BashToolCall

COLD_PREFIX = "tb-snap:"
PARKED_PREFIX = "tb-parked:"
SNAPSHOT_MODES = ("cold", "parked")
# Separates a Terminal-Bench task name from an optional cache namespace, e.g.
# "hello-world::exp1" runs task hello-world under its own cache tree.
NAMESPACE_SEPARATOR = "::"
DEFAULT_MANAGER_URL = "http://127.0.0.1:9901"


class SnapshotContractError(Exception):
    """A live id was used where a snapshot handle is required, or vice versa."""


@dataclass(frozen=True)
class SandboxConfig:
    manager_url: str = DEFAULT_MANAGER_URL
    command_timeout_sec: float = 120.0
    snapshot_mode: str = "cold"
    timeouts: ManagerTimeouts = ManagerTimeouts()
    # Receives {"op", "seconds", ...} for start/restore/snapshot/stop/execute/
    # test, so runners can report snapshot and restore costs.
    on_event: Optional[Callable[[dict], None]] = None
    transport: object = field(default=None, repr=False)  # httpx transport for tests

    def __post_init__(self):
        if self.snapshot_mode not in SNAPSHOT_MODES:
            raise ValueError(f"snapshot_mode must be one of {SNAPSHOT_MODES}, got {self.snapshot_mode!r}")


def tb_task_name(task_name: str) -> str:
    """The Terminal-Bench task for a (possibly namespaced) cache task id."""
    return task_name.split(NAMESPACE_SEPARATOR, 1)[0]


def is_parked_handle(env_id: Optional[str]) -> bool:
    return env_id is not None and env_id.startswith(PARKED_PREFIX)


def is_cold_handle(env_id: Optional[str]) -> bool:
    return env_id is not None and env_id.startswith(COLD_PREFIX)


def is_snapshot_handle(env_id: Optional[str]) -> bool:
    return is_cold_handle(env_id) or is_parked_handle(env_id)


class TerminalBenchSandbox(Sandbox):
    """One Terminal-Bench environment managed by container-manager-public.

    tvclient constructs sandboxes as env_class(task_name=..., env_id=...), so
    configuration is a class attribute; bind it with with_config().
    """

    config: ClassVar[SandboxConfig] = SandboxConfig()

    @classmethod
    def with_config(cls, config: SandboxConfig) -> type:
        return type(cls.__name__, (cls,), {"config": config})

    def _client(self) -> ManagerClient:
        return ManagerClient(
            self.config.manager_url, self.config.timeouts, self.config.transport
        )

    def _emit(self, op: str, started: float, **details: object) -> None:
        if self.config.on_event is not None:
            self.config.on_event(
                {"op": op, "seconds": time.perf_counter() - started, **details}
            )

    def _require_live(self, action: str) -> str:
        if self.env_id is None:
            raise SnapshotContractError(f"cannot {action}: sandbox not started")
        if is_snapshot_handle(self.env_id):
            raise SnapshotContractError(
                f"cannot {action} in snapshot {self.env_id}; restore it with start()"
            )
        return self.env_id

    async def start(self) -> None:
        started = time.perf_counter()
        client = self._client()
        handle = self.env_id
        if handle is None:
            self.env_id = await client.start(tb_task_name(self.task_name))
            self._emit("start", started, env_id=self.env_id)
        elif is_cold_handle(handle):
            self.env_id = await client.restore(handle[len(COLD_PREFIX) :])
            self._emit("restore", started, handle=handle, env_id=self.env_id, mode="cold")
        elif is_parked_handle(handle):
            self.env_id = await client.fork(handle[len(PARKED_PREFIX) :])
            self._emit("restore", started, handle=handle, env_id=self.env_id, mode="parked")
        else:
            raise SnapshotContractError(
                f"start() with env_id={handle!r}: only snapshot handles ({COLD_PREFIX}... "
                f"or {PARKED_PREFIX}...) can be restored; a live id is already running"
            )

    async def fork(self) -> "TerminalBenchSandbox":
        live_id = self._require_live("fork")
        started = time.perf_counter()
        if self.config.snapshot_mode == "cold":
            try:
                handle_id = COLD_PREFIX + await self._client().snapshot(live_id)
            except ManagerError as error:
                if error.status == 404 and "Terminal not found" not in error.body:
                    raise ManagerError(
                        error.endpoint, error.status,
                        "manager has no /snapshot endpoint; run the cold-snapshots branch "
                        "of container-manager-public or use snapshot_mode='parked'",
                    ) from error
                raise
        else:
            handle_id = PARKED_PREFIX + await self._client().fork(live_id)
        handle = type(self)(task_name=self.task_name, env_id=handle_id)
        self._emit("snapshot", started, parent=live_id, handle=handle_id, mode=self.config.snapshot_mode)
        return handle

    async def execute(self, tool_call: ToolCall) -> str:
        live_id = self._require_live("execute")
        if isinstance(tool_call, TestToolCall):
            return await self._run_test(live_id)
        if not isinstance(tool_call, BashToolCall):
            raise TypeError(f"TerminalBenchSandbox cannot execute {type(tool_call).__name__}")

        started = time.perf_counter()
        client = self._client()
        markers = Markers.new()
        timeout = self.config.command_timeout_sec
        # A CommandTimeout or transport error propagates: the shell may still
        # be running the command, so tvclient must discard this sandbox.
        await client.execute(live_id, markers.begin_command, timeout)
        await client.execute(live_id, tool_call.command, timeout)
        await client.execute(live_id, markers.end_command, timeout)
        pane_log = await client.get_pane_logs(live_id)
        self._emit("execute", started, env_id=live_id)
        try:
            return extract_output(pane_log, markers)
        except ObservationError as error:
            # The command ran, so the shell is usable, but this output must
            # never be cached ("Failed to execute" results are not inserted).
            return f"Failed to execute: could not extract command output ({error})"

    async def _run_test(self, live_id: str) -> str:
        started = time.perf_counter()
        try:
            passed = await self._client().test(live_id)
        except ManagerError as error:
            # The manager reports a broken test run (as opposed to a failing
            # one) with an error status. Not the agent's result; don't cache it.
            return f"Failed to execute tests: {error}"
        self._emit("test", started, env_id=live_id, passed=passed)
        return json.dumps({"passed": passed}, sort_keys=True)

    async def stop(self) -> None:
        if self.env_id is None:
            return
        started = time.perf_counter()
        client = self._client()
        if is_cold_handle(self.env_id):
            await client.delete_snapshot(self.env_id[len(COLD_PREFIX) :])
        elif is_parked_handle(self.env_id):
            await client.stop(self.env_id[len(PARKED_PREFIX) :])
        else:
            await client.stop(self.env_id)
        self._emit("stop", started, env_id=self.env_id)
