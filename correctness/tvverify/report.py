"""`tvverify report`: combine `--json` results into one markdown report.

    python -m tvverify report results/*.json > report.md
"""

import json
from collections import Counter
from pathlib import Path
from typing import List


def render(paths: List[str]) -> str:
    runs = [(p, json.loads(Path(p).read_text())) for p in paths]
    tasks = [(p, t) for p, run in runs for t in run["tasks"]]
    totals = Counter()
    by_criterion = Counter()
    for _, t in tasks:
        for bucket in ("bugs", "noise", "inconclusive"):
            totals[bucket] += len(t.get(bucket, []))
        for f in t.get("bugs", []):
            by_criterion[f"{f['criterion']}/{f['code']}"] += 1
        totals["nodes_total"] += t.get("nodes_total", 0)
        totals["nodes_replayed"] += t.get("nodes_replayed", 0)
        totals["cache_mismatches"] += len(t.get("cache_mismatches", []))
        totals["snapshot_delete_failures"] += t.get("snapshot_delete_failures", 0)

    failing = [t for _, t in tasks if t.get("bugs") or t.get("cache_mismatches")]
    lines = [
        "# tvverify report",
        "",
        f"{len(runs)} result file(s), {len(tasks)} task(s): **{len(failing)} with bugs**.",
        "",
        "| | count |",
        "|---|---|",
        f"| nodes replayed / in the tries | {totals['nodes_replayed']} / {totals['nodes_total']} |",
        f"| bugs | {totals['bugs']} |",
        f"| noise (nondeterministic tools) | {totals['noise']} |",
        f"| inconclusive (replay could not run) | {totals['inconclusive']} |",
        f"| cache vs no-cache mismatches | {totals['cache_mismatches']} |",
        f"| snapshots that could not be deleted | {totals['snapshot_delete_failures']} |",
    ]
    if by_criterion:
        lines += ["", "## Bugs by criterion", "", "| criterion/code | count |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in by_criterion.most_common()]
    lines += ["", "## Tasks", "", "| task | replayed | bugs | noise | inconclusive | file |", "|---|---|---|---|---|---|"]
    for p, t in sorted(tasks, key=lambda pt: (-len(pt[1].get("bugs", [])), pt[1]["task"])):
        lines.append(
            f"| {t['task']} | {t.get('nodes_replayed', 0)}/{t.get('nodes_total', 0)} | {len(t.get('bugs', []))} "
            f"| {len(t.get('noise', []))} | {len(t.get('inconclusive', []))} | {Path(p).name} |"
        )
    shown = [(t["task"], f) for _, t in tasks for f in t.get("bugs", [])][:50]
    if shown:
        lines += ["", "## Bug details (first 50)", ""]
        for task, f in shown:
            path = " → ".join(str(k)[:60] for k in (f.get("path") or ()))
            lines.append(f"- **{task}** `{f['criterion']}/{f['code']}` node `{f.get('node_id')}`: {f['detail'][:300]}  ")
            if path:
                lines.append(f"  path: `{path}`")
    return "\n".join(lines) + "\n"
