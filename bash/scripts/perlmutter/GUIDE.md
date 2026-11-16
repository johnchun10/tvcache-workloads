# One-time setup, and fixes for known errors

Do all of this on a login node, where it's free. Everything installs under
your own `$SCRATCH/tvcache`, so teammates don't share or overwrite each
other's setup, and your own uv, docker-compose, conda or Python installs are
left alone. To run and read results, see `README.md`. `ISSUES.md` has the
history of what broke during bring-up.

## Setup

**1. Code and everything it needs**

    git clone https://github.com/johnchun10/tvcache-workloads.git $SCRATCH/tvcache/tvcache-workloads
    cd $SCRATCH/tvcache/tvcache-workloads/bash/scripts/perlmutter
    bash bootstrap.sh    # clones main of our four forks into $SCRATCH/tvcache
    bash setup.sh        # about 20 minutes the first time

`setup.sh` installs whatever is missing and skips the rest:
- uv and docker-compose, into `$SCRATCH/tvcache/bin`. The container manager
  runs on uv. podman-hpc runs the containers, but it has no compose, and every
  task is a compose file.
- the veRL environment (`verl-env`, conda with veRL 0.9.1), and the
  embedders' environment (`embed-env`)
- the cache server, the Python packages, Qwen2.5-1.5B and the embedders

Expect `tools ok`, `python side ok: verl 0.9.1 ...`, and `embedder ... ok`
for each embedder. Run both scripts again whenever `main` changes.

**2. Podman probe**

    bash probe_podman.sh

Every line must say PASS. That matters most for `docker API socket` and `SDK
lists the probe container`; nothing works without them.

**3. Shakedown**

    bash run.sh shakedown --prebuild && bash submit_hetjob.sh shakedown

When it finishes, `report.md` should show veRL exit 0 and nothing left over.
If the two nodes don't seem to start or reach each other,
`sbatch probe_hetjob.sbatch` checks that in about 10 minutes.

## Things to know

- **Login-node ports:** `--prebuild` starts the container manager on port
  9901. If two people prebuild on the same login node at once, they clash.
  Pick your own first: `export FLASK_PORT=99xx`. Jobs get whole nodes, so
  runs aren't affected.
- **Images are node-local:** a job builds each task image again on its own
  node. Prebuilding only finds the tasks that don't build.
- **Scratch purge:** files on `$SCRATCH` that go unused for about 8 weeks are
  deleted. If anything goes missing, run `setup.sh` again.
- **Tasks that never build:** `fix-git` (its GitHub repository is gone) and
  `accelerate-maximal-square` (its uids are out of rootless podman's range).
  Neither is in a preset.

## Known errors

**Job dies early: `container manager did not open port 9901`, or `os error
524` / `Errno 524` on a lock.** `$HOME` on compute nodes has no file locks.
`env.sh` moves the caches to scratch. A new tool that hits this needs its
cache under `$TVROOT/cache` too; `probe_locks.sh` shows which filesystems
support locks.

**`OUT_OF_MEMORY` on a single GPU node.** On GPU nodes, podman's image store
is `/tmp`, which is RAM. Use the two-node job or a whole GPU node. If
`logs/memory.csv` still climbs, raise `--snapshot-threshold` or run fewer
tasks.

**`crun: create /sys/fs/cgroup/docker: Permission denied` during a build.**
Compose's BuildKit doesn't work under rootless podman. `env.sh` turns it
off; make sure it's sourced. Remove a leftover builder with `podman-hpc rm -f
buildx_buildkit_default`.

**Containers or snapshot images left over.** The manager doesn't clean up
after failed starts. Remove them with:

    podman-hpc rm -af
    podman-hpc rmi $(podman-hpc images --filter label=com.docker.compose.project=tvcache-snapshot -q)

**tvverify contract check: restored cwd is `/app`, not `/app/work`.** The
manager checkout is missing `container-manager-public/rlhf_server/tmux-res`, so
the working directory and environment variables are never saved. It fails
silently; only this check catches it.

**`RuntimeError: tvcache-rust not found` from tvverify.** `env.sh` sets
`TVCACHE_RUST_DIR`; source it first.

**tvverify reports `ls -l /tmp`, `wget` or `apt-get` commands.** These are
almost always noise (timestamps, network), not cache bugs. Check the node in
both dumps before reporting it.
