#!/usr/bin/env bash
# Checks the CPU+GPU split before a real run (about 10 minutes on 2 nodes):
#   1. a CPU+GPU heterogeneous job is accepted and starts
#   2. the GPU node resolves the CPU node's hostname and reaches a TCP server
#      on it (the cache server and manager listen on 0.0.0.0 the same way)
#   3. scancel <job>+1 releases the GPU component alone and the CPU component
#      keeps running (needed for RELEASE_GPU=1 in hetjob.sbatch)
#
#   sbatch -o $SCRATCH/tvcache/runs/slurm/%x-%j.out probe_hetjob.sbatch
# Read the log: it ends with "PASS cpu component survived" if 3 holds.
#SBATCH -J tv-probe-het
#SBATCH -A m4999 -q debug -t 00:10:00
#SBATCH -C cpu -N 1
#SBATCH hetjob
#SBATCH -A m4999 -q debug -t 00:10:00
#SBATCH -C gpu -N 1 -G 1
set -u
PORT=7878
CPU_HOST="$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0" | head -1)"
echo "job $SLURM_JOB_ID: cpu $CPU_HOST, gpu $SLURM_JOB_NODELIST_HET_GROUP_1, batch script on $(hostname)"

srun --het-group=0 -N 1 -n 1 bash -c "echo server on \$(hostname); cd /tmp && exec python3 -m http.server $PORT --bind 0.0.0.0" &
sleep 10
srun --het-group=1 -N 1 -n 1 -G 1 bash -c "
  echo client on \$(hostname)
  getent hosts $CPU_HOST && echo 'PASS resolve' || echo 'FAIL resolve'
  curl -s -o /dev/null -w 'http %{http_code}, %{time_total} s\n' http://$CPU_HOST:$PORT/ && echo 'PASS tcp' || echo 'FAIL tcp'
  nvidia-smi -L"

echo "cancelling the GPU component (${SLURM_JOB_ID}+1)"
scancel "${SLURM_JOB_ID}+1"
sleep 30
squeue -j "$SLURM_JOB_ID" -o '%i %T %N'
echo "PASS cpu component survived"
kill %1 2>/dev/null
