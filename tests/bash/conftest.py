import asyncio
import os
import socket
import uuid

import pytest

CACHE_HOST = os.environ.get("TVCACHE_HOST", "127.0.0.1")
CACHE_PORT = int(os.environ.get("TVCACHE_PORT", "7878"))
MANAGER_URL = os.environ.get("TB_MANAGER_URL", "http://127.0.0.1:9901")


def _reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


@pytest.fixture
def cache_server():
    # Selecting a real-service tier is an explicit request, so a missing
    # server is a failure, not a silent skip.
    if not _reachable(CACHE_HOST, CACHE_PORT):
        pytest.fail(
            f"tvcache-rust is not listening on {CACHE_HOST}:{CACHE_PORT}; "
            "start it with `cargo run --release` in ../tvcache-rust"
        )
    return CACHE_HOST, CACHE_PORT


@pytest.fixture
def cache_task_id():
    """A fresh cache namespace for the hello-world task, so each test starts uncached."""
    return f"hello-world::test-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def run():
    return asyncio.run
