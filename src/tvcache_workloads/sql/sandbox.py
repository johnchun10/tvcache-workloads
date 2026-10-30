"""SQLite sandbox for a small, stateful SmallBank-inspired workload."""

import asyncio
import json
import os
import sqlite3
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, ClassVar, Optional

from tvclient import Sandbox, TestToolCall, ToolCall

from tvcache_workloads.sql.tool_call import READ, SQLToolCall, WRITE

LIVE_PREFIX = "sql-live:"
SNAPSHOT_PREFIX = "sql-snap:"

INITIAL_ACCOUNTS = ((1, "Alice", 10_000, 5_000), (2, "Bob", 20_000, 7_000), (3, "Carol", 30_000, 9_000))
INITIAL_TOTAL_CENTS = sum(checking + savings for _, _, checking, savings in INITIAL_ACCOUNTS)


def _default_database_dir() -> Path:
    configured = os.environ.get("TVCACHE_SQL_DATABASE_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path(tempfile.gettempdir()) / f"tvcache-workloads-sql-{os.getuid()}"


@dataclass(frozen=True)
class SQLSandboxConfig:
    """Settings shared by all workers that restore these sandbox handles.

    For multi-host training, database_dir must be on storage visible at the
    same path to every rollout worker.
    """

    database_dir: Path = field(default_factory=_default_database_dir)
    simulated_sql_latency_seconds: float = 0.02
    on_event: Optional[Callable[[dict], None]] = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        if self.simulated_sql_latency_seconds < 0:
            raise ValueError("simulated_sql_latency_seconds must be non-negative")


class SnapshotContractError(Exception):
    """A live sandbox id was used where a snapshot handle is required, or vice versa."""


class SQLiteSandbox(Sandbox):
    """A fresh SQLite database per rollout with file-backed immutable snapshots."""

    config: ClassVar[SQLSandboxConfig] = SQLSandboxConfig()

    @classmethod
    def with_config(cls, config: SQLSandboxConfig) -> type:
        return type(cls.__name__, (cls,), {"config": config})

    def _emit(self, operation: str, started: float, **details: object) -> None:
        if self.config.on_event is not None:
            self.config.on_event(
                {"op": operation, "seconds": time.perf_counter() - started, **details}
            )

    def _database_root(self) -> Path:
        root = self.config.database_dir.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _new_path(self) -> Path:
        return self._database_root() / f"{uuid.uuid4().hex}.sqlite"

    def _path_from_handle(self, handle: str, prefix: str) -> Path:
        if not handle.startswith(prefix):
            raise SnapshotContractError(f"expected {prefix} handle, got {handle!r}")
        path = Path(handle[len(prefix) :]).resolve()
        if not path.is_relative_to(self._database_root()):
            raise SnapshotContractError("sandbox handle points outside the configured database dir")
        return path

    def _require_live_path(self, action: str) -> Path:
        if self.env_id is None:
            raise SnapshotContractError(f"cannot {action}: sandbox not started")
        if not self.env_id.startswith(LIVE_PREFIX):
            raise SnapshotContractError(
                f"cannot {action} in snapshot {self.env_id}; restore it with start()"
            )
        return self._path_from_handle(self.env_id, LIVE_PREFIX)

    async def start(self) -> None:
        started = time.perf_counter()
        if self.env_id is None:
            path = self._new_path()
            await asyncio.to_thread(self._initialize_database, path)
            self.env_id = LIVE_PREFIX + str(path)
            self._emit("start", started, env_id=self.env_id)
            return

        source = self._path_from_handle(self.env_id, SNAPSHOT_PREFIX)
        if not source.is_file():
            raise SnapshotContractError(f"snapshot file does not exist: {source}")
        destination = self._new_path()
        await asyncio.to_thread(self._backup_database, source, destination)
        self.env_id = LIVE_PREFIX + str(destination)
        self._emit("restore", started, source=str(source), env_id=self.env_id)

    async def fork(self) -> "SQLiteSandbox":
        source = self._require_live_path("fork")
        started = time.perf_counter()
        snapshot_path = self._new_path()
        await asyncio.to_thread(self._backup_database, source, snapshot_path)
        handle = SNAPSHOT_PREFIX + str(snapshot_path)
        self._emit("snapshot", started, parent=self.env_id, handle=handle)
        return type(self)(task_name=self.task_name, env_id=handle)

    async def execute(self, tool_call: ToolCall) -> str:
        path = self._require_live_path("execute")
        started = time.perf_counter()
        if self.config.simulated_sql_latency_seconds:
            await asyncio.sleep(self.config.simulated_sql_latency_seconds)

        try:
            if isinstance(tool_call, TestToolCall):
                result = await asyncio.to_thread(self._test_database, path)
            elif isinstance(tool_call, SQLToolCall):
                result = await asyncio.to_thread(self._execute_sql, path, tool_call)
            else:
                raise TypeError(f"SQLiteSandbox cannot execute {type(tool_call).__name__}")
        except sqlite3.Error as error:
            result = f"Failed to execute SQL: {error}"

        self._emit("execute", started, env_id=self.env_id)
        return result

    async def stop(self) -> None:
        if self.env_id is None:
            return
        started = time.perf_counter()
        prefix = SNAPSHOT_PREFIX if self.env_id.startswith(SNAPSHOT_PREFIX) else LIVE_PREFIX
        path = self._path_from_handle(self.env_id, prefix)
        self.env_id = None
        await asyncio.to_thread(self._delete_database, path)
        self._emit("stop", started, path=str(path))

    @staticmethod
    def _initialize_database(path: Path) -> None:
        with sqlite3.connect(path) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript(
                "CREATE TABLE accounts ("
                "custid INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);"
                "CREATE TABLE checking ("
                "custid INTEGER PRIMARY KEY REFERENCES accounts(custid), "
                "bal_cents INTEGER NOT NULL CHECK (bal_cents >= 0));"
                "CREATE TABLE savings ("
                "custid INTEGER PRIMARY KEY REFERENCES accounts(custid), "
                "bal_cents INTEGER NOT NULL CHECK (bal_cents >= 0));"
            )
            connection.executemany(
                "INSERT INTO accounts(custid, name) VALUES (?, ?)",
                [(custid, name) for custid, name, _, _ in INITIAL_ACCOUNTS],
            )
            connection.executemany(
                "INSERT INTO checking(custid, bal_cents) VALUES (?, ?)",
                [(custid, checking) for custid, _, checking, _ in INITIAL_ACCOUNTS],
            )
            connection.executemany(
                "INSERT INTO savings(custid, bal_cents) VALUES (?, ?)",
                [(custid, savings) for custid, _, _, savings in INITIAL_ACCOUNTS],
            )

    @staticmethod
    def _backup_database(source_path: Path, destination_path: Path) -> None:
        source_uri = source_path.as_uri() + "?mode=ro"
        with sqlite3.connect(source_uri, uri=True) as source, sqlite3.connect(
            destination_path
        ) as destination:
            source.backup(destination)

    @staticmethod
    def _delete_database(path: Path) -> None:
        path.unlink(missing_ok=True)
        Path(f"{path}-wal").unlink(missing_ok=True)
        Path(f"{path}-shm").unlink(missing_ok=True)

    @staticmethod
    def _authorize(action: int, mode: str) -> int:
        read_actions = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
        if hasattr(sqlite3, "SQLITE_RECURSIVE"):
            read_actions.add(sqlite3.SQLITE_RECURSIVE)
        if mode == READ:
            allowed = read_actions
        else:
            allowed = read_actions | {
                sqlite3.SQLITE_INSERT,
                sqlite3.SQLITE_UPDATE,
                sqlite3.SQLITE_DELETE,
                sqlite3.SQLITE_TRANSACTION,
            }
        return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

    @classmethod
    def _execute_sql(cls, path: Path, tool_call: SQLToolCall) -> str:
        read_only = tool_call.mode == READ
        if read_only:
            uri = path.as_uri() + "?mode=ro"
            connection = sqlite3.connect(uri, uri=True)
        else:
            connection = sqlite3.connect(path, isolation_level=None)
        connection.set_authorizer(
            lambda action, _arg1, _arg2, _database, _trigger: cls._authorize(
                action, tool_call.mode
            )
        )

        try:
            if not read_only:
                connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(tool_call.query, tool_call.params)
            if (
                tool_call.expected_rowcount is not None
                and cursor.rowcount != tool_call.expected_rowcount
            ):
                connection.rollback()
                return (
                    "Failed to execute SQL: expected "
                    f"{tool_call.expected_rowcount} changed rows, got {cursor.rowcount}"
                )

            if cursor.description is not None:
                value = {
                    "columns": [column[0] for column in cursor.description],
                    "rows": [list(row) for row in cursor.fetchall()],
                }
            else:
                value = {"rowcount": cursor.rowcount}
            if not read_only:
                connection.commit()
            return json.dumps(value, sort_keys=True)
        except sqlite3.Error as error:
            if not read_only:
                connection.rollback()
            return f"Failed to execute SQL: {error}"
        finally:
            connection.close()

    @staticmethod
    def _test_database(path: Path) -> str:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            account_count = connection.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
            negative_count = connection.execute(
                "SELECT "
                "(SELECT COUNT(*) FROM checking WHERE bal_cents < 0) + "
                "(SELECT COUNT(*) FROM savings WHERE bal_cents < 0)"
            ).fetchone()[0]
            total_cents = connection.execute(
                "SELECT "
                "(SELECT SUM(bal_cents) FROM checking) + "
                "(SELECT SUM(bal_cents) FROM savings)"
            ).fetchone()[0]

        passed = (
            account_count == len(INITIAL_ACCOUNTS)
            and negative_count == 0
            and total_cents == INITIAL_TOTAL_CENTS
        )
        return json.dumps(
            {
                "passed": passed,
                "account_count": account_count,
                "negative_balance_count": negative_count,
                "total_cents": total_cents,
            },
            sort_keys=True,
        )