"""veRL reward for Terminal-Bench trajectories (the TVCache paper's scheme):

  -1  the model never made a well-formed tool call
   0  it did, but the task's tests failed (or could not run)
   1  the tests passed

TvcacheToolAgentLoop puts the test output and the call count into the
trajectory's extra_fields, which veRL's reward manager merges into
`extra_info`. Point veRL at it with
  reward.custom_reward_function.path=<this file> reward.custom_reward_function.name=compute_score
"""

import json

TEST_KEY = "tb_test"
CALLS_KEY = "tb_tool_calls"


def passed(test_output) -> bool:
    try:
        return bool(json.loads(test_output).get("passed"))
    except (TypeError, ValueError, AttributeError):
        return False  # "Failed to execute ...", or no test ran


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> dict:
    """Returns the score plus per-trajectory numbers veRL logs as reward
    extra info: how many tool calls, how many failed to run, whether the
    tests passed. The sampling test reads them from a val_only run."""
    extra_info = extra_info or {}
    calls = int(extra_info.get(CALLS_KEY) or 0)
    ok = calls > 0 and passed(extra_info.get(TEST_KEY))
    return {
        "score": -1.0 if calls == 0 else (1.0 if ok else 0.0),
        "tool_calls": calls,
        "failed_calls": int(extra_info.get("tb_failed_calls") or 0),
        "passed": float(ok),
    }
