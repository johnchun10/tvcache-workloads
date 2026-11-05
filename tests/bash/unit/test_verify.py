"""Unit tests (no services): the tvverify adapter and its state probe parser.
Skipped unless the `verify` dependency group is installed."""

import pytest

pytest.importorskip("tvverify")

from tvclient import TestToolCall  # noqa: E402

from tvcache_workloads.bash import BashToolCall  # noqa: E402
from tvcache_workloads.bash.verify import (  # noqa: E402
    HelloWorldAdapter,
    ProbeError,
    parse_probe,
    probe_command,
)

OUTPUT = """tvv-h cwd ad1405e0d73051b4
tvv-m cwd /app/work
tvv-h oldpwd 0123456789abcdef
tvv-h env 36f5a88562606cea
tvv-m env fcde2b2edba5 FOO
tvv-h fs 05290a8b38cfb781
tvv-m fs 98ea6e4f216f /app/hello.txt
tvv-m fs d755 /app/a-very-long-direc
tory-name-wrapped-by-tmux"""


def test_parse_probe_reads_hashes_and_rejoins_wrapped_rows():
    hashes, manifests = parse_probe(OUTPUT)
    assert hashes == {
        "cwd": "ad1405e0d73051b4", "oldpwd": "0123456789abcdef",
        "env": "36f5a88562606cea", "fs": "05290a8b38cfb781",
    }
    assert manifests["cwd"] == {"cwd": "/app/work"}
    assert manifests["env"] == {"FOO": "fcde2b2edba5"}
    assert manifests["fs"]["/app/a-very-long-directory-name-wrapped-by-tmux"] == "d755"


def test_oldpwd_is_advisory_so_restores_can_still_match():
    from tvcache_workloads.bash.verify import ADVISORY, STRICT

    assert "oldpwd" in ADVISORY and "oldpwd" not in STRICT


def test_parse_probe_rejects_output_without_hashes():
    with pytest.raises(ProbeError):
        parse_probe("bash: find: command not found")


def test_probe_is_one_line_and_never_changes_directory():
    command = probe_command(("/app",))
    assert "\n" not in command
    assert " cd " not in f" {command}"


def test_keys_round_trip_including_the_test_call():
    adapter = HelloWorldAdapter()
    call = BashToolCall("echo 'Hello, world!' > /app/hello.txt")
    assert adapter.tool_call_from_key(call.serialize()).serialize() == call.serialize()
    assert isinstance(adapter.tool_call_from_key(TestToolCall("").serialize()), TestToolCall)


def test_rollouts_come_from_the_trace_in_order():
    adapter = HelloWorldAdapter()
    task = adapter.tasks()[0].task_id
    assert task == "hello-world"
    rollouts = adapter.sample_rollouts(task, None, 5)
    assert [c.command for c in rollouts[0]] == [c.command for c in rollouts[1]]  # solve, solve-again


def test_terminal_bench_adapter_rebuilds_calls_from_keys_and_names_tasks(monkeypatch):
    from tvcache_workloads.bash.verify import TerminalBenchAdapter

    adapter = TerminalBenchAdapter()
    key = BashToolCall("cd /app && ls").serialize()
    call = adapter.tool_call_from_key(key)
    assert call.serialize() == key and call.will_mutate_state()
    assert adapter.is_test_key(adapter.tool_call_from_key(TestToolCall("").serialize()).serialize())
    monkeypatch.setenv("TVVERIFY_TASKS", "hello-world::run7, fix-git")
    assert [(t.task_id, t.root_image_id) for t in adapter.tasks()] == [("hello-world::run7", "hello-world"), ("fix-git", "fix-git")]

