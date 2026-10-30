"""Async HTTP client for container-manager-public (rlhf_server/server.py).

Endpoint shapes verified against bhaskar792/container-manager-public @ 51a90bb
by reading the source and calling a live server:

    POST /start          {"task_name"}                           -> 200 {"id"}
    POST /execute        {"id","command","blocking","timeout"}   -> 200 {"status": "executed"}
                                                                    504 on command timeout
    POST /get_pane_logs  {"id"}                                  -> 200 {"logs"}
    POST /fork           {"id"}                                  -> 200 {"id"} (a *running* child)
    POST /test           {"id"}                                  -> 200 {"passed": bool}
                                                                    404 {"error"} if the test run broke
    POST /stop           {"id"}                                  -> 200 {"status": "stopped"}
                                                                    500 for an unknown id

Cold snapshots (cold-snapshots branch of our container-manager-public fork):

    POST /snapshot         {"id"}           -> 200 {"snapshot_id"}; 404 unknown terminal
    POST /restore          {"snapshot_id"}  -> 200 {"id"} (a new running terminal)
    POST /delete_snapshot  {"snapshot_id"}  -> 200 {"status": "deleted"}; 404 unknown

Not wrapped: /fork_async + /done_forking (unused by the sandbox), /solve
(oracle solution; not part of a rollout), /hash (returns the constant "1111").

Every request has a client-side timeout. This matters most for /fork: the
manager's fork loop waits forever if the commit or child start fails.
"""

import json
from dataclasses import dataclass
from typing import Any, Optional

import httpx


class ManagerError(Exception):
    """The manager answered with an error status."""

    def __init__(self, endpoint: str, status: int, body: str):
        super().__init__(f"{endpoint} returned HTTP {status}: {body[:500]}")
        self.endpoint = endpoint
        self.status = status
        self.body = body


class CommandTimeout(ManagerError):
    """/execute hit its timeout; the command may still be running in the shell."""


class ManagerUnavailable(Exception):
    """The manager could not be reached or did not answer in time."""


class ManagerProtocolError(Exception):
    """The manager answered 2xx with a body that does not match the expected shape."""


@dataclass(frozen=True)
class ManagerTimeouts:
    """Client-side bounds in seconds. Defaults leave room for image builds
    (/start), commit + child start (/fork), and the task's test setup (/test,
    whose server-side limit is 180 s plus package installs)."""

    connect: float = 10.0
    start: float = 900.0
    fork: float = 600.0
    snapshot: float = 600.0
    restore: float = 900.0
    stop: float = 300.0
    pane_logs: float = 60.0
    test: float = 900.0
    # Added to the per-command timeout for /execute's HTTP round trip.
    execute_slack: float = 30.0


class ManagerClient:
    def __init__(
        self,
        base_url: str,
        timeouts: ManagerTimeouts = ManagerTimeouts(),
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeouts = timeouts
        # Injected in tests (httpx.MockTransport); None means real network.
        self._transport = transport

    async def start(self, task_name: str) -> str:
        body = await self._post("/start", {"task_name": task_name}, self.timeouts.start)
        return self._field(body, "/start", "id", str)

    async def execute(self, terminal_id: str, command: str, timeout_sec: float) -> None:
        body = await self._post(
            "/execute",
            {"id": terminal_id, "command": command, "blocking": True, "timeout": timeout_sec},
            timeout_sec + self.timeouts.execute_slack,
        )
        status = self._field(body, "/execute", "status", str)
        if status != "executed":
            raise ManagerProtocolError(f"/execute returned unexpected status {status!r}")

    async def get_pane_logs(self, terminal_id: str) -> str:
        body = await self._post("/get_pane_logs", {"id": terminal_id}, self.timeouts.pane_logs)
        return self._field(body, "/get_pane_logs", "logs", str)

    async def fork(self, terminal_id: str) -> str:
        body = await self._post("/fork", {"id": terminal_id}, self.timeouts.fork)
        return self._field(body, "/fork", "id", str)

    async def snapshot(self, terminal_id: str) -> str:
        body = await self._post("/snapshot", {"id": terminal_id}, self.timeouts.snapshot)
        return self._field(body, "/snapshot", "snapshot_id", str)

    async def restore(self, snapshot_id: str) -> str:
        body = await self._post("/restore", {"snapshot_id": snapshot_id}, self.timeouts.restore)
        return self._field(body, "/restore", "id", str)

    async def delete_snapshot(self, snapshot_id: str) -> None:
        await self._post("/delete_snapshot", {"snapshot_id": snapshot_id}, self.timeouts.stop)

    async def test(self, terminal_id: str) -> bool:
        body = await self._post("/test", {"id": terminal_id}, self.timeouts.test)
        return self._field(body, "/test", "passed", bool)

    async def stop(self, terminal_id: str) -> None:
        await self._post("/stop", {"id": terminal_id}, self.timeouts.stop)

    async def _post(self, endpoint: str, payload: dict, timeout: float) -> Any:
        try:
            async with httpx.AsyncClient(
                transport=self._transport,
                timeout=httpx.Timeout(timeout, connect=self.timeouts.connect),
            ) as client:
                response = await client.post(self.base_url + endpoint, json=payload)
        except httpx.TimeoutException as error:
            raise ManagerUnavailable(
                f"{endpoint} did not answer within {timeout:.0f}s ({type(error).__name__})"
            ) from error
        except httpx.TransportError as error:
            raise ManagerUnavailable(
                f"cannot reach manager at {self.base_url}{endpoint}: {error}"
            ) from error

        if response.status_code == 504 and endpoint == "/execute":
            raise CommandTimeout(endpoint, response.status_code, response.text)
        if response.status_code >= 400:
            raise ManagerError(endpoint, response.status_code, response.text)
        try:
            return response.json()
        except json.JSONDecodeError as error:
            raise ManagerProtocolError(
                f"{endpoint} returned non-JSON body: {response.text[:200]!r}"
            ) from error

    @staticmethod
    def _field(body: Any, endpoint: str, key: str, expected: type) -> Any:
        if not isinstance(body, dict) or key not in body:
            raise ManagerProtocolError(f"{endpoint} response lacks {key!r}: {body!r}"[:300])
        value = body[key]
        if not isinstance(value, expected):
            raise ManagerProtocolError(
                f"{endpoint} field {key!r} is {type(value).__name__}, expected {expected.__name__}"
            )
        return value
