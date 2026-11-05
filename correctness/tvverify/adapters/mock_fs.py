"""A tiny in-memory "filesystem" sandbox for exercising the correctness tooling.

Unlike MockListSandbox, tool outputs here DEPEND ON STATE (reading a file
returns what was written, `cd` changes what relative paths mean), so a cache
that resumes from the wrong state produces a visibly wrong output.

State captured by snapshots (fork): files, cwd, env, tmp.
State NOT captured when `lossy_snapshots` is on: background processes. That
reproduces what `docker commit` does to running processes and lets tests show
the framework catching it.

Opt-in nondeterminism (never used by the "exact" suite):
  - `stamp f` appends a fresh random token to f (stateful AND noisy: the token
    is both the output and part of the state, so value/snapshot coherence can
    be checked);
  - `now` returns a fresh random number (stateless, noisy).
"""

import asyncio
import itertools
import random
from typing import Dict, List, Optional, Set, Tuple

from tvclient.sandbox import Sandbox
from tvclient.tool_call import TestToolCall, ToolCall

from tvverify.digest.schema import DigestComponent, DigestPolicy, StateDigest

STATEFUL_OPS = {"write", "append", "rm", "touch", "cd", "export", "tmpwrite", "stamp", "spawn"}
STATELESS_OPS = {"read", "ls", "printenv", "ps", "now", "tmpread"}


class ToyCall(ToolCall):
    def __init__(self, op: str, *args: str):
        assert op in STATEFUL_OPS | STATELESS_OPS, op
        self.op = op
        self.args = tuple(args)

    def to_dict(self) -> dict:
        return {"op": self.op, "args": list(self.args)}

    def will_mutate_state(self) -> bool:
        return self.op in STATEFUL_OPS

    def __repr__(self) -> str:
        return f"{self.op}({', '.join(self.args)})"


class _State:
    def __init__(self) -> None:
        self.files: Dict[str, str] = {}
        self.cwd = "/"
        self.env: Dict[str, str] = {}
        self.tmp: Dict[str, str] = {}
        self.procs: Set[str] = set()

    def copy(self, keep_procs: bool) -> "_State":
        other = _State()
        other.files = dict(self.files)
        other.cwd = self.cwd
        other.env = dict(self.env)
        other.tmp = dict(self.tmp)
        other.procs = set(self.procs) if keep_procs else set()
        return other

    def path(self, name: str) -> str:
        return self.cwd.rstrip("/") + "/" + name

    def reward(self) -> float:
        from tvverify.digest.schema import sha256_json

        h = sha256_json([self.files, self.cwd, self.env])
        return 1.0 if int(h[:2], 16) % 2 == 0 else 0.0


class ToyFsSandbox(Sandbox):
    live: Dict[str, _State] = {}
    snapshots: Dict[str, _State] = {}
    execute_log: List[Tuple[str, str]] = []
    lossy_snapshots = False  # snapshots drop background processes
    op_delay = 0.0005
    root_image_id = "toyfs-root"
    # Seeded per process; tests that need reproducibility call reseed().
    _noise = random.Random(0)
    _counter = itertools.count()

    @classmethod
    def reset(cls) -> None:
        cls.live.clear()
        cls.snapshots.clear()
        cls.execute_log.clear()
        cls.lossy_snapshots = False
        cls.reseed(0)

    @classmethod
    def isolated(cls):
        """Context manager: run with clean class-level state, then put back
        whatever was there (so a long-lived fixture's snapshots survive)."""
        from contextlib import contextmanager

        @contextmanager
        def scope():
            saved = (dict(cls.live), dict(cls.snapshots), list(cls.execute_log), cls.lossy_snapshots, cls._noise)
            cls.reset()
            try:
                yield
            finally:
                cls.live.clear(); cls.live.update(saved[0])
                cls.snapshots.clear(); cls.snapshots.update(saved[1])
                cls.execute_log[:] = saved[2]
                cls.lossy_snapshots, cls._noise = saved[3], saved[4]

        return scope()

    @classmethod
    def reseed(cls, seed: int) -> None:
        cls._noise = random.Random(seed)

    @classmethod
    def _new_id(cls) -> str:
        return f"toy-{next(cls._counter)}"

    @property
    def state(self) -> _State:
        return ToyFsSandbox.live[self.env_id]

    async def start(self) -> None:
        cls = ToyFsSandbox
        if self.env_id is None:
            self.env_id = cls._new_id()
            cls.live[self.env_id] = _State()
        elif self.env_id in cls.snapshots:
            restored = cls._new_id()
            cls.live[restored] = cls.snapshots[self.env_id].copy(keep_procs=True)
            self.env_id = restored
        elif self.env_id not in cls.live:
            raise KeyError(f"unknown sandbox or snapshot {self.env_id!r}")

    async def fork(self) -> "ToyFsSandbox":
        cls = ToyFsSandbox
        handle = cls._new_id()
        cls.snapshots[handle] = self.state.copy(keep_procs=not cls.lossy_snapshots)
        return ToyFsSandbox(task_name=self.task_name, env_id=handle)

    async def stop(self) -> None:
        ToyFsSandbox.live.pop(self.env_id, None)
        ToyFsSandbox.snapshots.pop(self.env_id, None)

    async def execute(self, tool_call: ToolCall) -> str:
        cls = ToyFsSandbox
        await asyncio.sleep(cls.op_delay)
        cls.execute_log.append((self.env_id, tool_call.serialize()))
        s = self.state

        if isinstance(tool_call, TestToolCall):
            return f"reward={s.reward()}"

        op, args = tool_call.op, tool_call.args
        if op == "write":
            s.files[s.path(args[0])] = args[1]
            return "ok"
        if op == "append":
            key = s.path(args[0])
            s.files[key] = s.files.get(key, "") + args[1]
            return f"len={len(s.files[key])}"
        if op == "rm":
            return "removed" if s.files.pop(s.path(args[0]), None) is not None else "no such file"
        if op == "touch":
            s.files.setdefault(s.path(args[0]), "")
            return "ok"
        if op == "cd":
            s.cwd = args[0]
            return f"cwd={s.cwd}"
        if op == "export":
            s.env[args[0]] = args[1]
            return "ok"
        if op == "tmpwrite":
            s.tmp[args[0]] = args[1]
            return "ok"
        if op == "stamp":
            token = f"<{cls._noise.randrange(10**9)}>"
            key = s.path(args[0])
            s.files[key] = s.files.get(key, "") + token
            return token
        if op == "spawn":
            s.procs.add(args[0])
            return "spawned"
        if op == "read":
            return s.files.get(s.path(args[0]), "no such file")
        if op == "ls":
            prefix = s.cwd.rstrip("/") + "/"
            return " ".join(sorted(p[len(prefix):] for p in s.files if p.startswith(prefix)))
        if op == "tmpread":
            return s.tmp.get(args[0], "")
        if op == "printenv":
            return s.env.get(args[0], "")
        if op == "ps":
            return " ".join(sorted(s.procs))
        if op == "now":
            return f"t={cls._noise.randrange(10**9)}"
        raise ValueError(f"unknown op {op}")

    async def state_digest(self) -> Optional[dict]:
        return toy_digest(self.state, DEFAULT_POLICY).to_dict()


DEFAULT_POLICY = DigestPolicy(
    strict_components=frozenset({"fs", "env", "cwd"}),
    advisory_components=frozenset({"tmp", "procs"}),
)


def toy_digest(state: _State, policy: DigestPolicy) -> StateDigest:
    """Digest of a ToyFs state under a policy. Components outside the policy
    are omitted, which is how tests imitate an incomplete digest."""
    comps: Dict[str, DigestComponent] = {}
    candidates = {
        "fs": dict(state.files),
        "env": dict(state.env),
        "cwd": {"cwd": state.cwd},
        "tmp": dict(state.tmp),
        "procs": {p: "1" for p in state.procs},
    }
    for name, entries in candidates.items():
        if name in policy.strict_components:
            comps[name] = DigestComponent.of(name, entries, strict=True)
        elif name in policy.advisory_components:
            comps[name] = DigestComponent.of(name, entries, strict=False)
    return StateDigest(
        comps, policy_hash=policy.policy_hash(), root_image_id=ToyFsSandbox.root_image_id
    )


def random_calls(rng: random.Random, length: int, *, noisy: bool = False, hidden: bool = False) -> List[ToolCall]:
    """A random rollout over a small alphabet, so different rollouts collide on
    the same prefixes (which is what makes the cache do anything)."""
    files, contents, dirs = ["a", "b"], ["x", "y"], ["/", "/d"]
    makers = [
        lambda: ToyCall("write", rng.choice(files), rng.choice(contents)),
        lambda: ToyCall("append", rng.choice(files), rng.choice(contents)),
        lambda: ToyCall("rm", rng.choice(files)),
        lambda: ToyCall("touch", rng.choice(files)),
        lambda: ToyCall("cd", rng.choice(dirs)),
        lambda: ToyCall("export", "K", rng.choice(contents)),
        lambda: ToyCall("tmpwrite", "t", rng.choice(contents)),
        lambda: ToyCall("read", rng.choice(files)),
        lambda: ToyCall("read", rng.choice(files)),
        lambda: ToyCall("ls"),
        lambda: ToyCall("printenv", "K"),
    ]
    if noisy:
        makers += [lambda: ToyCall("stamp", rng.choice(files)), lambda: ToyCall("now")]
    if hidden:
        makers += [lambda: ToyCall("spawn", rng.choice(["p", "q"])), lambda: ToyCall("ps")]
    return [rng.choice(makers)() for _ in range(length)]


def rollout_set(rng: random.Random, count: int, max_len: int, **kwargs) -> List[List[ToolCall]]:
    """Rollouts that share prefixes with a common trunk, so the cache is used."""
    trunk = random_calls(rng, max_len, **kwargs)
    rollouts = []
    for _ in range(count):
        keep = rng.randint(0, len(trunk))
        rollouts.append(trunk[:keep] + random_calls(rng, rng.randint(1, max_len), **kwargs))
    return rollouts
