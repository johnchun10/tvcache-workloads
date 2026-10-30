"""BashToolCall: one shell command, and the unit of the TVCache key."""

from tvclient import ToolCall

# tvclient sends each path entry as a u16-length string, so one serialized
# call must fit in 65535 UTF-8 bytes.
MAX_SERIALIZED_BYTES = 0xFFFF


class BashToolCall(ToolCall):
    """A single command typed into the task's interactive Bash shell.

    The cache key is the command text exactly as given: no whitespace or
    quoting normalization, so two calls share a key only when they are the
    same string. Recognizing differently written but equivalent commands is
    the semantic-cache layer's job, not this class's.

    Every command is treated as state-mutating unless the caller explicitly
    passes mutates=False. Marking a command read-only is a promise that it
    changes nothing a later command could observe (files, cwd, variables,
    processes); a wrong promise yields wrong cached results, so there is no
    automatic classification.
    """

    def __init__(self, command: str, mutates: bool = True):
        if not isinstance(command, str):
            raise TypeError(f"command must be a str, got {type(command).__name__}")
        if not command.strip():
            raise ValueError("command must not be empty")
        self.command = command
        self.mutates = mutates
        size = len(self.serialize().encode("utf-8"))
        if size > MAX_SERIALIZED_BYTES:
            raise ValueError(
                f"serialized command is {size} bytes; the cache protocol allows "
                f"at most {MAX_SERIALIZED_BYTES}"
            )

    def to_dict(self) -> dict:
        return {"tool": "bash", "command": self.command}

    def will_mutate_state(self) -> bool:
        return self.mutates

    def __repr__(self) -> str:
        suffix = "" if self.mutates else ", mutates=False"
        return f"BashToolCall({self.command!r}{suffix})"
