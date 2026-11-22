"""Prints the memory the live sandboxes' processes use, in GB.

    python sandbox_memory.py

A sandbox is a container, and every container has its own PID namespace, so
this adds up the memory of every process whose PID namespace is not this
script's own. The measure is PSS (/proc/<pid>/smaps_rollup): memory shared by
several processes is divided among them, so the sum counts each page once.

`podman stats` cannot give this on Perlmutter: rootless podman-hpc puts every
container in one cgroup and reports that cgroup's total for each of them
(ISSUES.md #25). The files a sandbox writes are not process memory; they are
in the image store (logs/memory.csv: graphroot_gb), which is a RAM disk on
compute nodes.
"""

import os
import sys
from pathlib import Path


def sandbox_pss_kb(proc: Path = Path("/proc")) -> int:
    """Summed PSS, in kB, of the processes outside this process's PID
    namespace. A process that ends meanwhile, or cannot be read, is skipped."""
    try:
        own = os.readlink(proc / "self" / "ns" / "pid")
    except OSError:
        return 0
    total = 0
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if os.readlink(entry / "ns" / "pid") == own:
                continue
            for line in (entry / "smaps_rollup").read_text().splitlines():
                if line.startswith("Pss:"):
                    total += int(line.split()[1])
                    break
        except (OSError, ValueError, IndexError):
            continue
    return total


if __name__ == "__main__":
    sys.stdout.write(f"{sandbox_pss_kb() / 2**20:.2f}\n")
