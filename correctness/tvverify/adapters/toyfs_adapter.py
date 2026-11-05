import json
import random
from typing import List, Sequence

from tvclient.tool_call import TestToolCall
from tvverify.adapters.base import TaskSpec, WorkloadAdapter
from tvverify.adapters.mock_fs import ToyCall, ToyFsSandbox, DEFAULT_POLICY, rollout_set, toy_digest
from tvverify.compare.base import Comparator, ExactComparator
from tvverify.digest.schema import DigestPolicy


class ToyFsAdapter(WorkloadAdapter):
    name = "mock_fs"
    sandbox_class = ToyFsSandbox
    tool_call_class = ToyCall

    def tool_call_from_key(self, key: str):
        if self.is_test_key(key):
            return TestToolCall("")
        data = json.loads(key)
        return ToyCall(data["op"], *data["args"])

    def tool_class(self, key: str) -> str:
        if self.is_test_key(key):
            return "test"
        return "noisy" if json.loads(key)["op"] in ("stamp", "now") else "deterministic"

    def comparator_for(self, key: str) -> Comparator:
        return ExactComparator()

    def digest_policy(self, task_id: str) -> DigestPolicy:
        return DEFAULT_POLICY

    def sample_rollouts(self, task_id: str, rng: random.Random, count: int) -> List[List[ToyCall]]:
        return rollout_set(rng, count, 5)

    def tasks(self) -> Sequence[TaskSpec]:
        return [TaskSpec("toyfs-task", ToyFsSandbox.root_image_id)]

    async def probe_state(self, sandbox, policy):
        # Digest the sandbox under the requested policy (the sandbox's own
        # state_digest() only knows the default one).
        return toy_digest(sandbox.state, policy)
