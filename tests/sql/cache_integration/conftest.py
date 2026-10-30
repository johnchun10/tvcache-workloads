import os
import socket
import uuid

import pytest


@pytest.fixture
def cache_server():
    host = os.environ.get("TVCACHE_HOST", "127.0.0.1")
    port = int(os.environ.get("TVCACHE_PORT", "7878"))
    try:
        with socket.create_connection((host, port), timeout=2):
            pass
    except OSError:
        pytest.fail(
            f"tvcache-rust is not listening on {host}:{port}; "
            "start it with `cargo run --release` in ../tvcache-rust"
        )
    return host, port


@pytest.fixture
def cache_task_id():
    return f"smallbank-send-payment::test-{uuid.uuid4().hex[:8]}"