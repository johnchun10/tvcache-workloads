"""Sandbox contract checks: does a user's Sandbox behave the way the cache and
the correctness framework assume? Run these before anything else, since every
later check trusts them.

Each check returns a ContractResult with a hint saying what to fix. Every
sandbox and snapshot a check creates is stopped afterwards, so real
(container-backed) sandboxes are not leaked.
"""

import random
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional

from tvclient.sandbox import Sandbox
from tvclient.tool_call import ToolCall

from tvverify.adapters.base import WorkloadAdapter, sandbox_equals


@dataclass
class ContractResult:
    name: str
    ok: bool
    hint: str = ""
    skipped: bool = False

    def __str__(self) -> str:
        if self.skipped:
            return f"SKIP  {self.name}  ({self.hint})"
        return f"{'PASS' if self.ok else 'FAIL'}  {self.name}" + (f"  -> {self.hint}" if self.hint and not self.ok else "")


class Skip(Exception):
    """A check that cannot run with the calls the adapter provided."""


class _Ctx:
    def __init__(self, adapter: WorkloadAdapter, seed: int):
        self.adapter = adapter
        tasks = adapter.tasks()
        if not tasks:
            raise ValueError("tasks() returned no tasks")
        # Sandboxes are booted for a real task: workloads such as Terminal-Bench
        # pick the environment from the task name.
        self.task = tasks[0].task_id
        calls = [c for r in adapter.sample_rollouts(self.task, random.Random(seed), 4) for c in r]
        self.calls = calls
        self.stateful: Optional[ToolCall] = next((c for c in calls if c.will_mutate_state()), None)
        self.stateless: Optional[ToolCall] = next((c for c in calls if not c.will_mutate_state()), None)
        # Whether equals() told some stateful call apart from a fresh sandbox.
        self.noticed = False
        self.stateful_tried = 0
        self.live: List[Sandbox] = []

    def sandbox(self, env_id: Optional[str] = None) -> Sandbox:
        return self.adapter.sandbox_class(task_name=self.task, env_id=env_id)

    async def fresh(self) -> Sandbox:
        sandbox = self.sandbox()
        await sandbox.start()
        self.live.append(sandbox)
        return sandbox

    async def fork(self, sandbox: Sandbox) -> str:
        """Snapshots `sandbox`; the snapshot is deleted in cleanup()."""
        handle = await sandbox.fork()
        self.live.append(handle)
        return handle.get_id()

    async def restore(self, handle: str) -> Sandbox:
        sandbox = self.sandbox(handle)
        await sandbox.start()
        self.live.append(sandbox)
        return sandbox

    def need_stateful(self) -> ToolCall:
        if self.stateful is None:
            raise Skip("no stateful call in the sampled rollouts")
        return self.stateful

    async def pick_effective_call(self) -> None:
        """Prefers a stateful call that equals() can tell apart from a fresh
        sandbox, so the later checks are not trivially true on a no-op call
        (e.g. `rm` of a missing file). Records whether any call was noticed."""
        seen = set()
        for call in self.calls:
            if not call.will_mutate_state() or call.serialize() in seen:
                continue
            seen.add(call.serialize())
            a, b = await self.fresh(), await self.fresh()
            await a.execute(call)
            different = not await sandbox_equals(a, b)
            await self.stop(a, b)
            if different:
                self.stateful, self.noticed = call, True
                break
        self.stateful_tried = len(seen)

    async def stop(self, *sandboxes: Sandbox) -> None:
        for sandbox in sandboxes:
            self.live.remove(sandbox)
            await sandbox.stop()

    async def cleanup(self) -> None:
        for sandbox in self.live:
            try:
                await sandbox.stop()
            except Exception:  # best effort
                pass
        self.live.clear()


Check = Callable[[_Ctx], Awaitable[Optional[str]]]
CHECKS: List[tuple] = []


def check(name: str):
    def register(fn: Check) -> Check:
        CHECKS.append((name, fn))
        return fn

    return register


@check("tool calls survive serialize -> tool_call_from_key")
async def _round_trip(ctx: _Ctx) -> Optional[str]:
    for call in ctx.calls:
        again = ctx.adapter.tool_call_from_key(call.serialize())
        if again.serialize() != call.serialize():
            return f"{call.serialize()[:60]} came back as {again.serialize()[:60]}"
    return None


@check("rollouts contain at least one stateful call")
async def _has_stateful(ctx: _Ctx) -> Optional[str]:
    if ctx.stateful is None:
        return "sample_rollouts() returned no call with will_mutate_state() == True"
    return None


@check("two fresh sandboxes are equal")
async def _fresh_equal(ctx: _Ctx) -> Optional[str]:
    a, b = await ctx.fresh(), await ctx.fresh()
    return None if await sandbox_equals(a, b) else "start() twice gave sandboxes that equals() calls different"


@check("equals() is reflexive")
async def _reflexive(ctx: _Ctx) -> Optional[str]:
    a = await ctx.fresh()
    return None if await sandbox_equals(a, a) else "a sandbox is not equal to itself"


@check("equals() notices a stateful call")
async def _notices(ctx: _Ctx) -> Optional[str]:
    ctx.need_stateful()
    if ctx.noticed:
        return None
    return f"equals() stayed True after each of {ctx.stateful_tried} stateful calls: it ignores the state they change"


@check("equals() agrees on identical histories")
async def _same_history(ctx: _Ctx) -> Optional[str]:
    call = ctx.need_stateful()
    a, b = await ctx.fresh(), await ctx.fresh()
    await a.execute(call)
    await b.execute(call)
    return None if await sandbox_equals(a, b) else "same call on two fresh sandboxes gave different states (nondeterministic call, or equals() is too strict)"


@check("stateless calls leave the state unchanged")
async def _stateless(ctx: _Ctx) -> Optional[str]:
    if ctx.stateless is None:
        raise Skip("no stateless call in the sampled rollouts")
    a, b = await ctx.fresh(), await ctx.fresh()
    if ctx.stateful is not None:
        await a.execute(ctx.stateful)
        await b.execute(ctx.stateful)
    await a.execute(ctx.stateless)
    return None if await sandbox_equals(a, b) else "a call with will_mutate_state() == False changed the state"


@check("fork() does not change the source")
async def _fork_pure(ctx: _Ctx) -> Optional[str]:
    call = ctx.need_stateful()
    a, twin = await ctx.fresh(), await ctx.fresh()
    await a.execute(call)
    await twin.execute(call)
    await ctx.fork(a)
    return None if await sandbox_equals(a, twin) else "the source sandbox changed when it was forked"


@check("a restored snapshot equals the source at fork time")
async def _restore(ctx: _Ctx) -> Optional[str]:
    call = ctx.need_stateful()
    a = await ctx.fresh()
    await a.execute(call)
    restored = await ctx.restore(await ctx.fork(a))
    return None if await sandbox_equals(a, restored) else "restoring a fork handle did not reproduce the state (is something not captured by fork()?)"


@check("a restored sandbox is independent of its source")
async def _independent(ctx: _Ctx) -> Optional[str]:
    call = ctx.need_stateful()
    a = await ctx.fresh()
    restored = await ctx.restore(await ctx.fork(a))
    twin = await ctx.fresh()
    await restored.execute(call)
    return None if await sandbox_equals(a, twin) else "executing in the restored sandbox changed the source"


@check("stop() works on a live sandbox and on a snapshot handle")
async def _stop(ctx: _Ctx) -> Optional[str]:
    a = await ctx.fresh()
    await ctx.fork(a)
    handle = ctx.live[-1]
    try:
        await ctx.stop(handle, a)
    except Exception as exc:
        return f"stop() raised {type(exc).__name__}: {exc}"
    return None


async def run_contract(adapter: WorkloadAdapter, seed: int = 0) -> List[ContractResult]:
    """Runs every check. A method the user has not implemented yet is reported
    as `not implemented`, not as a crash."""
    try:
        ctx = _Ctx(adapter, seed)
    except NotImplementedError as exc:
        return [ContractResult("adapter provides tasks and rollouts", False, f"not implemented: {exc}")]
    except Exception as exc:
        return [ContractResult("adapter provides tasks and rollouts", False, f"{type(exc).__name__}: {exc}")]
    results: List[ContractResult] = []
    try:
        try:
            await ctx.pick_effective_call()
        except Exception:  # the same failure is reported by the checks below
            await ctx.cleanup()
        for name, fn in CHECKS:
            try:
                problem = await fn(ctx)
            except Skip as skip:
                results.append(ContractResult(name, True, str(skip), skipped=True))
                continue
            except NotImplementedError as exc:
                results.append(ContractResult(name, False, f"not implemented: {exc}"))
                continue
            except Exception as exc:
                results.append(ContractResult(name, False, f"{type(exc).__name__}: {exc}"))
                continue
            finally:
                # Stop each check's sandboxes before the next one, so a real
                # workload never has more than a few containers running.
                await ctx.cleanup()
            results.append(ContractResult(name, problem is None, problem or ""))
    finally:
        await ctx.cleanup()
    return results
