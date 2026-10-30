"""Per-command output extraction from the manager's tmux pane log.

container-manager-public's /execute returns only a status, and /get_pane_logs
returns the whole terminal history (for forked terminals, prefixed with the
parent's history). To get one command's output, the sandbox runs

    1. a BEGIN marker command,
    2. the user's command, sent unchanged,
    3. an END marker command,

as three separate blocking /execute calls and keeps the text between the two
marker *output* lines. The marker commands use printf so their typed form
(``printf '__TVB%s_%s__\\n' BEGIN <nonce>``) never equals their printed form
(``__TVBBEGIN_<nonce>__``), and every call uses a fresh nonce.

What the observation is: the terminal-rendered text the command printed
(stdout and stderr interleaved), without the prompt or command echo. What it is
not: raw stdout. The manager drops blank lines, tmux wraps lines at the pane
width (160 columns), and the exit status is unavailable because the manager
appends ``; tmux wait -S done`` to every command, overwriting ``$?``.
"""

import re
import uuid
from dataclasses import dataclass
from typing import List, Optional

# Appended by Terminal-Bench's TmuxSession to every blocking command.
COMPLETION_SUFFIX = "; tmux wait -S done"
_COMPACT_SUFFIX = re.sub(r"\s+", "", COMPLETION_SUFFIX)
# A whole shell prompt such as "root@terminal:/app# " or "user@host:~$ ".
_PROMPT = re.compile(r"[^\s@]+@[^\s:]+:[^#$]*[#$]")


class ObservationError(Exception):
    """The command's output could not be located in the pane log."""


@dataclass(frozen=True)
class Markers:
    nonce: str

    @classmethod
    def new(cls) -> "Markers":
        return cls(uuid.uuid4().hex[:16])

    @property
    def begin_command(self) -> str:
        return f"printf '__TVB%s_%s__\\n' BEGIN {self.nonce}"

    @property
    def end_command(self) -> str:
        return f"printf '__TVB%s_%s__\\n' END {self.nonce}"

    @property
    def begin_line(self) -> str:
        return f"__TVBBEGIN_{self.nonce}__"

    @property
    def end_line(self) -> str:
        return f"__TVBEND_{self.nonce}__"


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _drop_command_echo(rows: List[str]) -> List[str]:
    """Removes the prompt + typed command row(s) at the start of the segment.

    The echo ends with COMPLETION_SUFFIX and may wrap over several rows;
    whitespace is ignored because tmux trims spaces at wrap points.
    """
    accumulated = ""
    for index, row in enumerate(rows):
        accumulated += row
        if _compact(accumulated).endswith(_COMPACT_SUFFIX):
            return rows[index + 1 :]
    raise ObservationError("could not find the end of the command echo")


def _text_before_prompt(prefix: str, next_prompt: Optional[str]) -> str:
    """Splits "<unterminated output><prompt>" and returns the output part.

    next_prompt is the idle prompt row after the END marker, which is the
    prompt the END command was typed at. The manager rewrites hostnames to
    "terminal" only at the start of a row, so the host is matched loosely.
    """
    if next_prompt is not None:
        parts = re.fullmatch(r"([^\s@]+)@[^\s:]+:(.*)", next_prompt.strip())
        if parts:
            pattern = re.escape(parts[1]) + r"@[^\s:]+:" + re.escape(parts[2]) + r"\s*$"
            found = re.search(pattern, prefix)
            if found:
                return prefix[: found.start()]
    elif _PROMPT.fullmatch(prefix.strip()):
        # No idle prompt to learn from; a generic prompt cannot tell "hi" in
        # "hiroot@host:/app#" from the user name, so this assumes no output.
        return ""
    raise ObservationError("could not separate unterminated output from the prompt")


def _drop_end_marker_echo(
    rows: List[str], markers: Markers, next_prompt: Optional[str]
) -> List[str]:
    """Removes the prompt + typed END marker command at the end of the segment.

    Walks backwards until the typed END command is fully contained; the row
    where that happens is where it starts. Output printed without a trailing
    newline shares that row with the prompt, so the text before the prompt is
    kept.
    """
    target = _compact(markers.end_command + COMPLETION_SUFFIX)
    accumulated = ""
    for index in range(len(rows) - 1, -1, -1):
        accumulated = rows[index] + accumulated
        if target in _compact(accumulated):
            kept = rows[:index]
            start = rows[index].find("printf '__TVB")
            if start > 0:
                leftover = _text_before_prompt(rows[index][:start], next_prompt)
                if leftover:
                    kept.append(leftover)
            return kept
    raise ObservationError("could not find the echo of the END marker command")


def extract_output(pane_log: str, markers: Markers) -> str:
    """Returns the text the command printed between the two marker lines."""
    rows = pane_log.splitlines()

    end_index = None
    for index in range(len(rows) - 1, -1, -1):
        if rows[index].strip() == markers.end_line:
            end_index = index
            break
    if end_index is None:
        raise ObservationError("END marker not found in pane log")

    begin_index = None
    for index in range(end_index - 1, -1, -1):
        if rows[index].strip() == markers.begin_line:
            begin_index = index
            break
    if begin_index is None:
        # The output outgrew tmux's scrollback (2000 rows by default). What
        # survives cannot be told apart from history the manager prepends for
        # forked terminals, so it is reported as a failure, never cached.
        raise ObservationError(
            "BEGIN marker not found: output likely exceeded the tmux scrollback"
        )

    segment = rows[begin_index + 1 : end_index]
    segment = _drop_command_echo(segment)
    next_prompt = rows[end_index + 1] if end_index + 1 < len(rows) else None
    segment = _drop_end_marker_echo(segment, markers, next_prompt)
    return "\n".join(segment)
