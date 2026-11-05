#!/usr/bin/env bash
# Compute-node follow-up to probe_podman.sh. Run inside a CPU allocation:
#   srun -q shared_interactive -C cpu -c 16 -t 00:20:00 -A <project> bash probe_compute.sh
# Before it, on a login node: podman-hpc migrate docker.io/library/ubuntu:22.04
# Checks: the probe itself, a real task's compose build/up/exec/down through
# the socket, and whether an image migrated from a login node is visible here.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
OUT="$HERE/probe_compute_$(hostname).txt"
bash "$HERE/probe_podman.sh"
: > "$OUT"
check() {
  local what="$1"; shift
  if out=$("$@" 2>&1); then echo "PASS  $what" | tee -a "$OUT"; else echo "FAIL  $what :: ${out: -400}" | tee -a "$OUT"; fi
}
echo "--- compute node: $(hostname)" | tee -a "$OUT"
check "migrated login-node image visible" bash -c "$CONTAINER_CLI images --format '{{.Repository}}:{{.Tag}}' | grep -q 'ubuntu:22.04'"

mkdir -p "$(dirname "$PODMAN_SOCKET")"
$CONTAINER_CLI system service --time=0 "unix://$PODMAN_SOCKET" > /tmp/tvprobe_service.log 2>&1 &
SVC=$!
sleep 3
TASK="$TASKS_ROOT/fix-permissions"
LOGS="$(mktemp -d)"
export T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME=tvprobe-fixperm T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME=tvprobe-fixperm
export T_BENCH_TEST_DIR=/tests T_BENCH_TASK_LOGS_PATH="$LOGS/a" T_BENCH_CONTAINER_LOGS_PATH=/logs
export T_BENCH_TASK_AGENT_LOGS_PATH="$LOGS/b" T_BENCH_CONTAINER_AGENT_LOGS_PATH=/agent-logs
mkdir -p "$LOGS/a" "$LOGS/b"
COMPOSE=("$COMPOSE_CMD" -p tvprobe -f "$TASK/docker-compose.yaml")
start=$(date +%s)
check "compose build (fix-permissions)"  "${COMPOSE[@]}" build
echo "      build took $(( $(date +%s) - start )) s" | tee -a "$OUT"
check "compose up -d"                    "${COMPOSE[@]}" up -d
check "exec in compose container"        $CONTAINER_CLI exec tvprobe-fixperm bash -c 'echo ok'
check "volume mount writable"            bash -c "$CONTAINER_CLI exec tvprobe-fixperm bash -c 'echo hi > /logs/x' && [ -f '$LOGS/a/x' ]"
check "compose down --rmi all"           "${COMPOSE[@]}" down --rmi all
kill $SVC 2>/dev/null
rm -rf "$LOGS"
echo "written to $OUT"
