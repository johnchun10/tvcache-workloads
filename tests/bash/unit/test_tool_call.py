"""Unit tests (no services): BashToolCall as a cache-key unit."""

import json

import pytest

from tvcache_workloads.bash import BashToolCall
from tvcache_workloads.bash.tool_call import MAX_SERIALIZED_BYTES


def test_key_is_exact_command_text():
    call = BashToolCall("ls  -la /app")
    assert json.loads(call.serialize()) == {"command": "ls  -la /app", "tool": "bash"}
    # No normalization: differently spaced commands are different keys.
    assert call.serialize() != BashToolCall("ls -la /app").serialize()


def test_serialization_is_canonical():
    assert BashToolCall("echo hi").serialize() == BashToolCall("echo hi").serialize()


def test_mutates_by_default_and_only_explicitly_read_only():
    assert BashToolCall("cat x").will_mutate_state() is True
    assert BashToolCall("cat x", mutates=False).will_mutate_state() is False
    # The flag is not part of the key; it only controls the stateful chain.
    assert BashToolCall("cat x").serialize() == BashToolCall("cat x", mutates=False).serialize()


@pytest.mark.parametrize("bad", ["", "   ", "\n"])
def test_rejects_empty_commands(bad):
    with pytest.raises(ValueError):
        BashToolCall(bad)


def test_rejects_non_string():
    with pytest.raises(TypeError):
        BashToolCall(["ls"])


def test_rejects_commands_too_large_for_the_wire_protocol():
    with pytest.raises(ValueError, match="at most"):
        BashToolCall("x" * MAX_SERIALIZED_BYTES)
    BashToolCall("x" * (MAX_SERIALIZED_BYTES - 100))
