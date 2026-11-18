"""veRL's synchronous v1 trainer, plus the control messages TVCache needs at
step boundaries (registered as trainer.v1.trainer_mode=tvcache_sync).

For each step:
  prepare_step      -> begin(i): records the step start with all of the step's
                       tasks; then warms the fork bank for step i+1 (peeked
                       one step ahead) while step i rolls out
  rollouts finished -> end(i): records the end and each task's graph size
                       (on_sample_end)
Contraction is paused and resumed per rollout by its own coordinator
(bash/contraction.py), not here.

A step is train_batch_size prompts, but veRL fetches them one dataloader batch
at a time (_fetch_one_gen_batch), and for this trainer mode a dataloader batch
is a single prompt. So the step's boundary is prepare_step, not the fetch: the
prompts of this step and the next wait in a look-ahead queue that the fetches
drain (ISSUES.md #32).

The first step is warmed synchronously. Peeking ahead moves the dataloader one
step past the one being trained, so checkpoints save the dataloader state from
just before the first prompt still in the queue; a resumed run then starts at
the peeked step instead of skipping it.
"""

import uuid
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils import tensordict_utils as tu

from tvcache_workloads.bash import control


def batch_tasks(batch_dict: dict) -> List[str]:
    return [info["task_id"] for info in batch_dict["extra_info"]]


@register_trainer("tvcache_sync")
class TvcacheSyncTrainer(PPOTrainerSync):
    # Fetched but not yet handed to veRL: (dataloader state before the fetch,
    # the dataloader batch), oldest first.
    _tv_queue: Optional[Deque[Tuple[dict, dict]]] = None
    _tv_tasks: List[str] = []  # the tasks of the step now running

    def _tv_raw_batch(self) -> dict:
        try:
            if self.train_dataloader_it is None:
                self.train_dataloader_it = iter(self.train_dataloader)
            return next(self.train_dataloader_it)
        except StopIteration:
            self.train_dataloader_it = iter(self.train_dataloader)
            return next(self.train_dataloader_it)

    def _tv_epoch(self) -> int:
        # steps_per_epoch is veRL's own count (dataset size // train_batch_size).
        # Not len(train_dataloader): the dataloader yields one prompt per
        # batch, so that is the number of tasks, not of steps.
        return (self.global_steps - 1) // max(1, self.steps_per_epoch)

    def _tv_step_fetches(self) -> int:
        """Dataloader batches in one step (veRL's _next_train_batch)."""
        train_batch_size = self.config.data.train_batch_size
        gen_batch_size = self.config.data.get("gen_batch_size", None) or train_batch_size
        return max(1, train_batch_size // gen_batch_size)

    def _tv_fill(self, fetches: int) -> None:
        while len(self._tv_queue) < fetches:
            state = self.train_dataloader.state_dict()
            self._tv_queue.append((state, self._tv_raw_batch()))

    def _tv_queued_tasks(self, start: int, stop: int) -> List[str]:
        return [task for _, batch in list(self._tv_queue)[start:stop] for task in batch_tasks(batch)]

    def prepare_step(self) -> dict:
        per_step = self._tv_step_fetches()
        first = self._tv_queue is None
        if first:
            self._tv_queue = deque()
        self._tv_fill(per_step)
        self._tv_tasks = self._tv_queued_tasks(0, per_step)
        if first:
            control.call("warm", tasks=self._tv_tasks, wait=True)
        control.call("begin", step=self.global_steps, epoch=self._tv_epoch(), tasks=self._tv_tasks)
        if self.global_steps < self.total_training_steps:  # nothing follows the last step
            self._tv_fill(2 * per_step)
            control.call("warm", tasks=self._tv_queued_tasks(per_step, 2 * per_step), keep=self._tv_tasks, wait=False)
        return super().prepare_step()

    def _fetch_one_gen_batch(self):
        if self._tv_queue:
            _, current = self._tv_queue.popleft()
        else:  # before the first step, or more than was peeked
            current = self._tv_raw_batch()
        # Prompts veRL adds in the middle of a step (it replaces groups it
        # evicted) belong to that step as well.
        self._tv_tasks = self._tv_tasks + [t for t in batch_tasks(current) if t not in self._tv_tasks]
        current["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(current["raw_prompt"]))], dtype=object)
        return tu.get_tensordict(current)

    def _save_checkpoint(self):
        # veRL saves train_dataloader.state_dict(), which already counts the
        # peeked prompts; hand it the state from before the first of them.
        if not self._tv_queue:
            return super()._save_checkpoint()
        state = self._tv_queue[0][0]
        self.train_dataloader.state_dict = lambda: state
        try:
            return super()._save_checkpoint()
        finally:
            del self.train_dataloader.state_dict

    def on_sample_end(self):
        control.call(
            "end",
            step=self.global_steps,
            epoch=self._tv_epoch(),
            tasks=self._tv_tasks,
            last=self.global_steps >= self.total_training_steps,
        )
        super().on_sample_end()
