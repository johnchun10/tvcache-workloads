"""Launches the Rust TVCache server for tests.

The server is found at $TVCACHE_RUST_DIR, or else in a `tvcache-rust`
directory next to one of this file's parent directories. A server that
supports `--port` gets a free port; an older one (no `--port`, always 7878)
works too, one server at a time. Either way the port is read from the
server's own "listening on ..." log line.
"""

import os
import re
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional


def _find_rust_dir() -> Path:
    if os.environ.get("TVCACHE_RUST_DIR"):
        return Path(os.environ["TVCACHE_RUST_DIR"]).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "tvcache-rust" / "Cargo.toml").is_file():
            return parent / "tvcache-rust"
    raise RuntimeError("tvcache-rust not found; set TVCACHE_RUST_DIR")


RUST_DIR = _find_rust_dir()
_LISTENING = re.compile(r"listening on [^\s:]+:(\d+)")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_built = False


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def build_server() -> Path:
    """cargo build (once per process); returns the binary path."""
    global _built
    binary = RUST_DIR / "target" / "debug" / "tvcache"
    if not _built:
        subprocess.run(
            ["cargo", "build", "--quiet"], cwd=RUST_DIR, check=True, capture_output=True
        )
        _built = True
    return binary


class RustServer:
    """Context manager: a fresh server process. `port` is the one it really
    listens on, known once it has started."""

    def __init__(self, extra_args: Optional[List[str]] = None, eviction_policy: str = "none"):
        self.port = _free_port()
        self.args = [f"--port={self.port}", f"--eviction-policy={eviction_policy}"] + (
            extra_args or []
        )
        self.process: Optional[subprocess.Popen] = None

    def _drain(self, announced: List[int], ready: threading.Event) -> None:
        # Keep reading so the pipe never fills; note the port when announced.
        for line in self.process.stdout:
            match = _LISTENING.search(_ANSI.sub("", line))
            if match and not ready.is_set():
                announced.append(int(match.group(1)))
                ready.set()

    def start(self) -> "RustServer":
        env = dict(os.environ, RUST_LOG="info")
        self.process = subprocess.Popen(
            [str(build_server())] + self.args,
            cwd=RUST_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        announced: List[int] = []
        ready = threading.Event()
        threading.Thread(target=self._drain, args=(announced, ready), daemon=True).start()
        if not ready.wait(timeout=10):
            self.stop()
            raise RuntimeError("tvcache server did not announce its port")
        self.port = announced[0]
        deadline = time.time() + 10
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"tvcache server exited with {self.process.returncode} (is port {self.port} in use?)")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    return self
            except OSError:
                time.sleep(0.05)
        self.stop()
        raise RuntimeError("tvcache server did not start listening")

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def __enter__(self) -> "RustServer":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
