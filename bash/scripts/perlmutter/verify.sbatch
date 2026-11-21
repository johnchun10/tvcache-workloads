#!/usr/bin/env bash
# tvverify for one finished run, as a job of its own on one CPU node
# (verify.sh). submit_hetjob.sh queues it to start when the training job ends;
# by hand, for a run that has none:
#
#   HERE=$PWD RUN=<run dir> sbatch --export=ALL -o <run dir>/slurm-verify-%j.out verify.sbatch
#
# The account and QOS are the #SBATCH -A/-q line below (the same as the CPU
# node's in hetjob.sbatch).
#SBATCH -J tv-verify
#SBATCH -A m4999 -q regular -t 02:00:00
#SBATCH -C cpu -N 1
set -uo pipefail
HERE="${HERE:?submit with submit_hetjob.sh, or set HERE to the scripts directory}"
RUN="${RUN:?submit with submit_hetjob.sh, or set RUN to the run directory}"
echo "$(date +%FT%T) job $SLURM_JOB_ID on $SLURM_JOB_NODELIST: tvverify for $RUN"
if [[ ! -f "$RUN/config.env" ]]; then
  echo "$(date +%FT%T) no $RUN/config.env: the training job did not get as far as its settings, so there is nothing to check"
  exit 1
fi
# The whole node, as the training job's server role has it (hetjob.sbatch).
srun -N 1 -n 1 -c 256 --mem=0 --cpu-bind=none bash "$HERE/verify.sh" "$RUN" > "$RUN/verify.out" 2>&1
status=$?
echo "$(date +%FT%T) verify.sh exited $status (output: $RUN/verify.out)"
exit "$status"
