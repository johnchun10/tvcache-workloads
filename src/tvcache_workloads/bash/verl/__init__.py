"""veRL integration for the Bash / Terminal-Bench workload: GRPO training
where every bash command goes through TVCache.

  tool.py        the `bash` tool (ToolAgentLoop), one TVCache executor per trajectory
  agent_loop.py  runs the task's tests when a trajectory ends
  reward.py      -1 no tool call, 0 tests failed, 1 tests passed
  dataset.py     the training parquet from Terminal-Bench task directories
  session.py     per-trajectory state shared by the above (no veRL import)
  configs/       tools.yaml and agent_loops.yaml for veRL

bash/scripts/perlmutter/train_pilot.sh puts it together.
"""
