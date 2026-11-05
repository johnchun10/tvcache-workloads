#!/usr/bin/env bash
# Part B step 1: can podman-hpc stand in for Docker here? Run on a compute
# node (salloc -q interactive -C cpu -N 1 -t 01:00:00 -A <project>).
# Prints PASS/FAIL per item and writes probe_<host>.txt next to this script.
# Gate: if the socket items fail, stop and report back before going further.
set -u
source "$(dirname "$0")/env.sh"
OUT="$(dirname "$0")/probe_$(hostname).txt"
IMG=docker.io/library/ubuntu:22.04
NAME=tvprobe_$$
: > "$OUT"
check() {  # check "description" command...
  local what="$1"; shift
  if out=$("$@" 2>&1); then echo "PASS  $what" | tee -a "$OUT"; else echo "FAIL  $what :: ${out:0:300}" | tee -a "$OUT"; fi
}

check "podman-hpc runs"                  $CONTAINER_CLI version
check "outbound network (pypi.org)"      curl -sSfI --max-time 10 https://pypi.org
check "pull $IMG"                        $CONTAINER_CLI pull -q $IMG
check "run a container"                  $CONTAINER_CLI run --rm $IMG true
check "start a detached container"       $CONTAINER_CLI run -d --name $NAME $IMG sleep 600
check "exec into it"                     $CONTAINER_CLI exec $NAME bash -c 'echo probe > /root/probe.txt'
check "commit --pause=false + LABEL"     $CONTAINER_CLI commit --pause=false --change "LABEL com.docker.compose.project=tvcache-snapshot" $NAME tvprobe_snap
check "committed image keeps the file"   bash -c "[ \"\$($CONTAINER_CLI run --rm tvprobe_snap cat /root/probe.txt)\" = probe ]"
check "tag"                              $CONTAINER_CLI tag tvprobe_snap tvprobe_snap2
check "untag (rmi a second name)"        $CONTAINER_CLI rmi tvprobe_snap2
check "cp into a container"              bash -c "echo hi > /tmp/tvprobe_cp && $CONTAINER_CLI cp /tmp/tvprobe_cp $NAME:/root/cp.txt"
check "network create/rm"                bash -c "$CONTAINER_CLI network create tvprobe_net && $CONTAINER_CLI network rm tvprobe_net"
check "tmux installable in a container"  $CONTAINER_CLI exec $NAME bash -c 'apt-get update -qq && apt-get install -y -qq tmux >/dev/null && tmux -V'

# Docker API socket: the manager's Docker SDK calls and docker-compose need it.
mkdir -p "$(dirname "$PODMAN_SOCKET")"
$CONTAINER_CLI system service --time=0 "unix://$PODMAN_SOCKET" > /tmp/tvprobe_service.log 2>&1 &
SVC=$!
sleep 3
check "docker API socket ($PODMAN_SOCKET)" $VERL_PY -c "import docker; docker.DockerClient(base_url='$DOCKER_HOST').ping()"
check "SDK lists the probe container"      $VERL_PY -c "import docker; c=docker.DockerClient(base_url='$DOCKER_HOST'); assert any(x.name=='$NAME' for x in c.containers.list())"
check "compose command ($COMPOSE_CMD)"     $COMPOSE_CMD version

echo "--- storage" | tee -a "$OUT"
$CONTAINER_CLI info --format 'graphroot={{.Store.GraphRoot}}' 2>&1 | tee -a "$OUT"
df -h "$($CONTAINER_CLI info --format '{{.Store.GraphRoot}}' 2>/dev/null || echo /tmp)" 2>&1 | tail -1 | tee -a "$OUT"
echo "--- node: $(nproc) cpus, $(free -g | awk '/Mem:/{print $2}') GB RAM" | tee -a "$OUT"

$CONTAINER_CLI rm -f $NAME >/dev/null 2>&1; $CONTAINER_CLI rmi tvprobe_snap >/dev/null 2>&1; kill $SVC 2>/dev/null
echo "written to $OUT"
