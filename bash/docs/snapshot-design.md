# Snapshot handling

## What tvclient needs

tvclient (and the cache server, which only stores the id) needs three things
from a workload. Neither repository needed changes for this.

| tvclient call | Meaning |
|---|---|
| `live.fork()` | Save the live environment as a **checkpoint** that never changes. The live one keeps running. |
| `Sandbox(env_id=checkpoint).start()` | Start a **new** live environment from the checkpoint. Must work any number of times. |
| `Sandbox(env_id=checkpoint).stop()` | Delete the checkpoint. |

The stock container manager can't do this: its only copy operation, `/fork`,
returns a **running** copy and then deletes the committed images.

## Mode 1 — `cold` (default): new manager endpoints

On branch `cold-snapshots` of our fork (`johnchun10/container-manager-public`),
in `rlhf_server/snapshot/snapshot_store.py`:

| Endpoint | Does |
|---|---|
| `POST /snapshot {"id"}` → `{"snapshot_id"}` | `docker commit` each of the terminal's containers and **keep** the images, plus a copy of the task folder with its compose file pointing at them. Starts nothing; the terminal keeps running. |
| `POST /restore {"snapshot_id"}` → `{"id"}` | Start a new terminal from those images. |
| `POST /delete_snapshot {"snapshot_id"}` | Remove the images and the copied folder. |

`/fork` and every other existing endpoint are unchanged.

The connector maps `fork()`, `start(handle)`, and `stop(handle)` to these, with
handles written `tb-snap:<snapshot id>`.

Two Docker details the implementation handles:

1. **Each restore gets its own tag.** Stopping a terminal runs
   `docker compose down --rmi all`, which deletes the images that terminal was
   started from. So a restore tags the snapshot's images under a new name and
   starts from that tag. Stopping the restore removes only its own tag.
2. **Snapshot images get their own compose label.** `docker commit` copies the
   container's labels, including `com.docker.compose.project`, and
   `compose down --rmi all` deletes every image with its project's label. So
   stopping the *original* terminal deleted its snapshots. The real e2e test
   caught this (restore failed with `No such image` once the original stopped).
   The fix commits with `--change "LABEL com.docker.compose.project=tvcache-snapshot"`.
   There's a regression test for it in the manager repo.

The manager also now waits for its env-save script (`exit.sh`) to finish before
committing; `/fork` doesn't wait.

## Mode 2 — `parked`: works with the stock manager

`fork()` calls `/fork` and **parks** the running child: nothing ever runs in it.
Its handle is `tb-parked:<terminal id>`. A restore is `/fork` of the parked
container. This meets the contract, but every checkpoint is an idle running
container, and every restore pays a commit plus a start.

## Rules that apply in both modes

- A handle's prefix decides how it is restored or deleted, not the current
  setting, so a cache holding either kind is handled correctly.
- `execute()` refuses a handle, and `start()` refuses to "restore" a plain live
  id. A running container is never quietly used as a checkpoint.

## Measured on real containers (hello-world, `tests/bash/e2e`)

| | cold | parked |
|---|---|---|
| Save a checkpoint | ~0.33–0.39 s | ~1.27 s |
| Resume from a checkpoint | ~0.46 s | ~1.03–1.28 s |
| Delete a checkpoint | ~0.02 s | ~1.8 s (stops a container) |
| Containers running per checkpoint | 0 | 1 |
| Divergent rollout, end to end | ~9.7 s | ~13 s |

Both modes passed the same checks:
- the parent is unaffected by being snapshotted;
- restores don't see each other's writes;
- a third restore, made after the parent and a sibling changed, matches the snapshot exactly;
- nothing is left over afterwards — no containers, images, or restore tags.

## What survives a checkpoint (same in both modes)

| State | Survives? |
|---|---|
| Files in each container | yes |
| Current directory, **exported** variables | yes (needs the manager's `ENABLE_CACHING=1`) |
| Unexported variables, functions, aliases, shell options | **no** |
| Running / background processes | **no** |
| Mounted volumes | **no** (`docker commit` skips them; hello-world has none that matter) |

Correctness claims are limited to tasks and commands that don't depend on the
"no" rows. `hello-world` is such a task. Multi-container tasks haven't been
tested yet.

## Other limits

- Snapshot records live in the manager's memory, like terminals do. If the
  manager restarts, the records are lost but the images stay on disk. Clean
  them up with `docker images | grep _test_container`.
- Snapshot images use disk space until deleted. Release them with
  `cache_admin.release_task_snapshots`, or through tvclient's `SandboxManager`
  eviction.
