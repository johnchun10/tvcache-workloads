"""One reading of the server's DumpAll, whichever server wrote it.

A server without graph contraction dumps each node as

    {"children": {<call>: <node>, ...}, "value": ..., "runtime": ..., "sandbox_id": ..., ...}

A server with contraction can hold several calls on one node (calls that left
the sandbox in the same state were merged), so it dumps

    {"children": [<node>, ...],
     "tool_calls": {<call>: {"value": ..., "runtime": ..., "embedding": ...}, ...},
     "settled": ..., "sandbox_id": ..., ...}

`plain_task` turns the second into the first, which is what the rest of
tvverify reads. A node with one call converts exactly. A merged node appears
under its first call (in sorted order) with its children; every other call
becomes an *alias*: a childless entry with that call's own value and the
node's snapshot, marked "alias_of". Replaying an alias checks the merge: its
call, run after the same parent, must give that value and match that snapshot.
Aliases are not extra nodes, so counts and snapshot ownership skip them.
"""

from typing import Iterator, List, Tuple

CONTRACTED_KEYS = ("children", "tool_calls", "settled")


def is_contracted(task: dict) -> bool:
    """Whether a task's dump entry comes from a server with contraction."""
    return isinstance(task.get("root", {}).get("children"), list)


def _plain_node(node: dict, call: str = "", alias_of: str = "") -> dict:
    out = {k: v for k, v in node.items() if k not in CONTRACTED_KEYS}
    entry = node.get("tool_calls", {}).get(call, {})
    out["value"] = entry.get("value")
    out["runtime"] = entry.get("runtime", 0)
    out["children"] = {}
    if alias_of:
        out["alias_of"] = alias_of
        return out
    for child in node["children"]:
        calls = sorted(child["tool_calls"])
        out["children"][calls[0]] = _plain_node(child, calls[0])
        for other in calls[1:]:
            out["children"][other] = _plain_node(child, other, alias_of=calls[0])
    return out


def plain_task(task: dict) -> dict:
    """A task's dump entry in the one-call-per-node shape (see the module
    docstring); returned unchanged if it already is."""
    if not is_contracted(task):
        return task
    return {**task, "root": _plain_node(task["root"])}


def plain_dump(dump: dict) -> dict:
    return {task_id: plain_task(task) for task_id, task in dump.items()}


def walk(node: dict, path: Tuple[str, ...] = ()) -> Iterator[Tuple[List[str], dict]]:
    """(path, node) for every node below `node` of a plain task, aliases included."""
    for call, child in node["children"].items():
        child_path = path + (call,)
        yield list(child_path), child
        yield from walk(child, child_path)
