"""Unit tests (no services): HTTP translation and error handling, via httpx.MockTransport."""

import asyncio
import json

import httpx
import pytest

from tvcache_workloads.bash import (
    CommandTimeout,
    ManagerClient,
    ManagerError,
    ManagerProtocolError,
    ManagerUnavailable,
)


def client_with(handler):
    seen = []

    def record(request):
        seen.append((request.url.path, json.loads(request.content)))
        return handler(request)

    return ManagerClient("http://manager:9901/", transport=httpx.MockTransport(record)), seen


def run(coro):
    return asyncio.run(coro)


def test_request_payloads_match_manager_schema():
    def handler(request):
        return {
            "/start": httpx.Response(200, json={"id": "term_1"}),
            "/execute": httpx.Response(200, json={"status": "executed"}),
            "/get_pane_logs": httpx.Response(200, json={"logs": "x"}),
            "/fork": httpx.Response(200, json={"id": "term_2"}),
            "/test": httpx.Response(200, json={"passed": True}),
            "/stop": httpx.Response(200, json={"status": "stopped"}),
            "/snapshot": httpx.Response(200, json={"snapshot_id": "snap_1"}),
            "/restore": httpx.Response(200, json={"id": "term_3"}),
            "/delete_snapshot": httpx.Response(200, json={"status": "deleted"}),
        }[request.url.path]

    client, seen = client_with(handler)

    async def scenario():
        assert await client.start("hello-world") == "term_1"
        await client.execute("term_1", "ls", 12.5)
        assert await client.get_pane_logs("term_1") == "x"
        assert await client.fork("term_1") == "term_2"
        assert await client.test("term_2") is True
        await client.stop("term_2")
        assert await client.snapshot("term_1") == "snap_1"
        assert await client.restore("snap_1") == "term_3"
        await client.delete_snapshot("snap_1")

    run(scenario())
    assert seen == [
        ("/start", {"task_name": "hello-world"}),
        ("/execute", {"id": "term_1", "command": "ls", "blocking": True, "timeout": 12.5}),
        ("/get_pane_logs", {"id": "term_1"}),
        ("/fork", {"id": "term_1"}),
        ("/test", {"id": "term_2"}),
        ("/stop", {"id": "term_2"}),
        ("/snapshot", {"id": "term_1"}),
        ("/restore", {"snapshot_id": "snap_1"}),
        ("/delete_snapshot", {"snapshot_id": "snap_1"}),
    ]


def test_execute_504_is_a_command_timeout():
    client, _ = client_with(lambda r: httpx.Response(504, json={"error": "timed out"}))
    with pytest.raises(CommandTimeout):
        run(client.execute("term_1", "sleep 999", 1))


def test_error_status_carries_endpoint_and_body():
    client, _ = client_with(lambda r: httpx.Response(500, json={"error": "'term_x'"}))
    with pytest.raises(ManagerError) as info:
        run(client.stop("term_x"))
    assert info.value.endpoint == "/stop" and info.value.status == 500
    assert "term_x" in info.value.body
    assert not isinstance(info.value, CommandTimeout)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>oops</html>"),
        httpx.Response(200, json={"wrong": "shape"}),
        httpx.Response(200, json={"id": 7}),
        httpx.Response(200, json=["term_1"]),
    ],
)
def test_malformed_success_bodies_are_protocol_errors(response):
    client, _ = client_with(lambda r: response)
    with pytest.raises(ManagerProtocolError):
        run(client.start("hello-world"))


def test_unexpected_execute_status_is_a_protocol_error():
    client, _ = client_with(lambda r: httpx.Response(200, json={"status": "queued"}))
    with pytest.raises(ManagerProtocolError):
        run(client.execute("term_1", "ls", 5))


def test_test_requires_boolean_result():
    client, _ = client_with(lambda r: httpx.Response(200, json={"passed": "yes"}))
    with pytest.raises(ManagerProtocolError):
        run(client.test("term_1"))


def test_transport_failures_and_timeouts_are_unavailable():
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    def hang(request):
        raise httpx.ReadTimeout("read timed out", request=request)

    for handler in (refuse, hang):
        client, _ = client_with(handler)
        with pytest.raises(ManagerUnavailable):
            run(client.fork("term_1"))
