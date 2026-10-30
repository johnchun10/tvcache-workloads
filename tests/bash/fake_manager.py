"""TEST DOUBLE for container-manager-public. Not a real environment.

Mimics the manager behaviors the adapter depends on, as observed against the
real server (51a90bb): command echo "<prompt><cmd>; tmux wait -S done", rows
wrapped at 160 columns, blank lines dropped from logs, a fork's log prefixed
with its parent's, cwd carried across forks and snapshots (ENABLE_CACHING=1),
the cold-snapshot endpoints of our fork (optional), HTTP 500 for
stopping an unknown id, 504 for a command timeout, and multi-line commands
joined with " && ".

The "shell" is a toy interpreter for a handful of commands used in tests; the
state is a dict of files plus a cwd.
"""

import itertools
import json
import re
import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx

WIDTH = 160
HISTORY_ROWS = 2000 + 40  # tmux default history-limit plus the visible screen
SUFFIX = "; tmux wait -S done"


@dataclass
class Terminal:
    task: str
    files: Dict[str, str] = field(default_factory=dict)
    dirs: set = field(default_factory=lambda: {"/app"})
    cwd: str = "/app"
    rows: List[str] = field(default_factory=list)
    inherited_log: str = ""

    def prompt(self) -> str:
        return f"root@terminal:{self.cwd}# "


class FakeManager:
    def __init__(self, tasks=("hello-world",), cold_snapshots=True):
        self.tasks = set(tasks)
        # False mimics the stock manager, which has no snapshot endpoints.
        self.cold_snapshots = cold_snapshots
        self.terminals: Dict[str, Terminal] = {}
        self.snapshots: Dict[str, Terminal] = {}
        self.requests: List[tuple] = []  # (endpoint, payload)
        self.fail: Dict[str, tuple] = {}  # endpoint -> (status, body), one-shot
        self.test_error: Optional[str] = None
        self._ids = itertools.count()

    # --- transport -----------------------------------------------------------
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        endpoint = request.url.path
        payload = json.loads(request.content or b"{}")
        self.requests.append((endpoint, payload))
        if endpoint in self.fail:
            status, body = self.fail.pop(endpoint)
            return httpx.Response(status, text=body)
        handler = getattr(self, "_" + endpoint.strip("/"), None)
        if endpoint in ("/snapshot", "/restore", "/delete_snapshot") and not self.cold_snapshots:
            handler = None
        if handler is None:
            return httpx.Response(404, json={"error": "no such endpoint"})
        return handler(payload)

    def live_ids(self) -> List[str]:
        return sorted(self.terminals)

    def calls(self, endpoint: str) -> List[dict]:
        return [p for e, p in self.requests if e == endpoint]

    # --- endpoints -----------------------------------------------------------
    def _start(self, payload):
        if payload["task_name"] not in self.tasks:
            return httpx.Response(400, json={"error": "docker-compose.yaml not found"})
        terminal_id = f"term_{next(self._ids)}"
        terminal = Terminal(task=payload["task_name"])
        terminal.rows.append(terminal.prompt())
        self.terminals[terminal_id] = terminal
        return httpx.Response(200, json={"id": terminal_id})

    def _execute(self, payload):
        terminal = self.terminals.get(payload["id"])
        if terminal is None:
            return httpx.Response(404, json={"error": "Session not found"})
        lines = [line.strip() for line in payload["command"].splitlines() if line.strip()]
        command = " && ".join(lines)
        typed = terminal.rows.pop() + command + SUFFIX
        terminal.rows.extend(typed[i : i + WIDTH] for i in range(0, len(typed), WIDTH))
        output, timed_out = self._run(terminal, command, payload["timeout"])
        if output:
            pending = output.split("\n")
            if not output.endswith("\n"):
                # Unterminated output shares its row with the next prompt.
                terminal.rows.extend(pending[:-1])
                terminal.rows.append(pending[-1] + terminal.prompt())
            else:
                terminal.rows.extend(pending[:-1])
                terminal.rows.append(terminal.prompt())
        else:
            terminal.rows.append(terminal.prompt())
        terminal.rows = terminal.rows[-HISTORY_ROWS:]
        if timed_out:
            return httpx.Response(504, json={"error": "Command timed out"})
        return httpx.Response(200, json={"status": "executed"})

    def _get_pane_logs(self, payload):
        terminal = self.terminals.get(payload["id"])
        if terminal is None:
            return httpx.Response(404, json={"error": "Session not found"})
        return httpx.Response(200, json={"logs": self._logs(terminal)})

    def _fork(self, payload):
        parent = self.terminals.get(payload["id"])
        if parent is None:
            # The real manager loops forever here; the client timeout covers it.
            return httpx.Response(500, json={"error": "Terminal not found"})
        child_id = f"term_{next(self._ids)}"
        child = Terminal(
            task=parent.task,
            files=dict(parent.files),
            dirs=set(parent.dirs),
            cwd=parent.cwd,
            inherited_log=self._logs(parent),
        )
        child.rows.append(child.prompt())
        self.terminals[child_id] = child
        return httpx.Response(200, json={"id": child_id})

    def _snapshot(self, payload):
        parent = self.terminals.get(payload["id"])
        if parent is None:
            return httpx.Response(404, json={"error": "Terminal not found"})
        snapshot_id = f"snap_{next(self._ids)}"
        self.snapshots[snapshot_id] = Terminal(
            task=parent.task,
            files=dict(parent.files),
            dirs=set(parent.dirs),
            cwd=parent.cwd,
            inherited_log=self._logs(parent),
        )
        return httpx.Response(200, json={"snapshot_id": snapshot_id})

    def _restore(self, payload):
        saved = self.snapshots.get(payload["snapshot_id"])
        if saved is None:
            return httpx.Response(404, json={"error": "Snapshot not found"})
        terminal_id = f"term_{next(self._ids)}"
        terminal = Terminal(
            task=saved.task,
            files=dict(saved.files),
            dirs=set(saved.dirs),
            cwd=saved.cwd,
            inherited_log=saved.inherited_log,
        )
        terminal.rows.append(terminal.prompt())
        self.terminals[terminal_id] = terminal
        return httpx.Response(200, json={"id": terminal_id})

    def _delete_snapshot(self, payload):
        if self.snapshots.pop(payload["snapshot_id"], None) is None:
            return httpx.Response(404, json={"error": "Snapshot not found"})
        return httpx.Response(200, json={"status": "deleted"})

    def _test(self, payload):
        terminal = self.terminals.get(payload["id"])
        if terminal is None:
            return httpx.Response(404, json={"error": "trying to test a non-existent session"})
        if self.test_error:
            return httpx.Response(404, json={"error": self.test_error})
        return httpx.Response(
            200, json={"passed": terminal.files.get("/app/hello.txt") == "Hello, world!\n"}
        )

    def _stop(self, payload):
        if self.terminals.pop(payload["id"], None) is None:
            return httpx.Response(500, json={"error": repr(payload["id"])})
        return httpx.Response(200, json={"status": "stopped"})

    # --- toy shell -----------------------------------------------------------
    @staticmethod
    def _logs(terminal: Terminal) -> str:
        rows = [row.rstrip() for row in terminal.rows if row.strip()]
        log = "\n".join(rows)
        return terminal.inherited_log + "\n" + log if terminal.inherited_log else log

    def _path(self, terminal: Terminal, name: str) -> str:
        return name if name.startswith("/") else f"{terminal.cwd.rstrip('/')}/{name}"

    def _run(self, terminal: Terminal, command: str, timeout: float):
        out = []
        for part in re.split(r"\s*(?:&&|;)\s*", command):
            if not part:
                continue
            words = shlex.split(part)
            name, args = words[0], words[1:]
            if name == "printf" and args and args[0] == "__TVB%s_%s__\\n":
                out.append(f"__TVB{args[1]}_{args[2]}__\n")
            elif name == "echo" and ">" in args:
                at = args.index(">")
                terminal.files[self._path(terminal, args[at + 1])] = " ".join(args[:at]) + "\n"
            elif name == "echo":
                out.append(" ".join(args) + "\n")
            elif name == "echo-n":
                out.append(" ".join(args))
            elif name == "cat":
                path = self._path(terminal, args[0])
                out.append(terminal.files.get(path, f"cat: {args[0]}: No such file or directory\n"))
            elif name == "mkdir":
                terminal.dirs.update(self._path(terminal, a) for a in args if a != "-p")
            elif name == "ls":
                base = self._path(terminal, args[0] if args else ".").rstrip("/")
                names = sorted(
                    {p[len(base) + 1 :].split("/")[0] for p in [*terminal.files, *terminal.dirs]
                     if p.startswith(base + "/")}
                )
                out.append("".join(n + "\n" for n in names))
            elif name == "cd":
                terminal.cwd = self._path(terminal, args[0])
            elif name == "pwd":
                out.append(terminal.cwd + "\n")
            elif name == "seq":
                out.append("".join(f"{i}\n" for i in range(1, int(args[0]) + 1)))
            elif name == "sleep" and float(args[0]) > timeout:
                return "".join(out), True
            elif name == "sleep":
                pass
            else:
                out.append(f"bash: {name}: command not found\n")
        return "".join(out), False
