"""Unit tests (no services): extracting one command's output from a pane log.

The first fixture is shaped like a real /get_pane_logs response captured from
container-manager-public; the rest are constructed edge cases.
"""

import pytest

from tvcache_workloads.bash.observation import Markers, ObservationError, extract_output

M = Markers("abc123")
OTHER = Markers("old999")


def block(markers, prompt, command, output_rows, next_prompt=None):
    """Rows for BEGIN marker, command, END marker as the manager renders them."""
    next_prompt = next_prompt or prompt
    return [
        f"{prompt}{markers.begin_command}; tmux wait -S done",
        markers.begin_line,
        f"{prompt}{command}; tmux wait -S done",
        *output_rows,
        f"{next_prompt}{markers.end_command}; tmux wait -S done",
        markers.end_line,
        next_prompt.rstrip(),
    ]


def test_simple_output():
    log = "\n".join(block(M, "root@terminal:/app# ", "cat /app/hello.txt", ["Hello, world!"]))
    assert extract_output(log, M) == "Hello, world!"


def test_no_output():
    log = "\n".join(block(M, "root@terminal:/app# ", "touch x", []))
    assert extract_output(log, M) == ""


def test_multi_row_output_and_cwd_change():
    rows = block(M, "root@terminal:/app# ", "cd /tmp && ls", ["a", "b"], "root@terminal:/tmp# ")
    assert extract_output("\n".join(rows), M) == "a\nb"


def test_wrapped_command_echo_is_removed():
    typed = "root@terminal:/app# echo " + "y" * 300 + "; tmux wait -S done"
    wrapped = [typed[i : i + 160] for i in range(0, len(typed), 160)]
    end_echo = f"root@terminal:/app# {M.end_command}; tmux wait -S done"
    rows = [M.begin_line, *wrapped, "OUT", end_echo, M.end_line]
    assert extract_output("\n".join(rows), M) == "OUT"


@pytest.mark.parametrize("host", ["terminal", "3f2a9c1d"])
def test_output_without_trailing_newline_is_kept(host):
    # The manager rewrites the hostname only at the start of a row, so the
    # mid-row prompt may still carry the container's real hostname.
    rows = [
        M.begin_line,
        "root@terminal:/app# printf hi; tmux wait -S done",
        f"hiroot@{host}:/app# {M.end_command}; tmux wait -S done",
        M.end_line,
        "root@terminal:/app#",
    ]
    assert extract_output("\n".join(rows), M) == "hi"


def test_typed_marker_commands_are_not_mistaken_for_markers():
    # The typed END command appears before the printed END line; only exact
    # printed lines count.
    assert M.end_line not in M.end_command
    assert M.begin_line not in M.begin_command


def test_uses_latest_markers_when_parent_history_is_prepended():
    parent = block(OTHER, "root@terminal:/app# ", "echo parent", ["parent"])
    child = block(M, "root@terminal:/app# ", "echo child", ["child"])
    assert extract_output("\n".join(parent + child), M) == "child"
    assert extract_output("\n".join(parent + child), OTHER) == "parent"


def test_output_that_contains_a_prompt_like_line_is_kept():
    rows = block(M, "root@terminal:/app# ", "cat notes", ["root@terminal:/x# not a real prompt"])
    assert extract_output("\n".join(rows), M) == "root@terminal:/x# not a real prompt"


def test_missing_end_marker_is_an_error():
    rows = block(M, "root@terminal:/app# ", "ls", ["a"])
    rows.remove(M.end_line)
    with pytest.raises(ObservationError, match="END"):
        extract_output("\n".join(rows), M)


def test_scrolled_out_begin_marker_is_an_error_not_a_partial_result():
    rows = block(M, "root@terminal:/app# ", "seq 5000", [str(i) for i in range(5000)])
    rows = rows[-2040:]
    with pytest.raises(ObservationError, match="scrollback"):
        extract_output("\n".join(rows), M)


def test_markers_are_unique_per_call():
    assert Markers.new().nonce != Markers.new().nonce
