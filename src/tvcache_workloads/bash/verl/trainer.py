"""veRL's synchronous v1 trainer, plus the control messages TVCache needs at
step boundaries (registered as trainer.v1.trainer_mode=tvcache_sync).

For each step:
  fetch batch i     -> begin(i): records the step start; then warms the fork
                       bank for batch i+1 (peeked one batch ahead) while batch
                       i rolls out
  rollouts finished -> end(i): records the end and each task's graph size
                       (on_sample_end)
Contraction is paused and resumed per rollout by its own coordinator
(bash/contraction.py), not here.

The first batch is warmed synchronously. Peeking ahead moves the dataloader one
batch past the step being trained, so checkpoints save the dataloader state from
just before the peek; a resumed run then starts at the peeked batch instead of
skipping it.
"""

import uuid
from typing import List, Optional

import numpy as np
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils import tensordict_utils as tu

from tvcache_workloads.bash import control


def batch_tasks(batch_dict: dict) -> List[str]:
    return [info["task_id"] for info in batch_dict["extra_info"]]


@register_trainer("tvcache_sync")
class TvcacheSyncTrainer(PPOTrainerSync):
    _tv_next: Optional[dict] = None
    _tv_tasks: List[str] = []
    _tv_state: Optional[dict] = None  # dataloader state before the peek

    def _tv_raw_batch(self) -> dict:
        try:
            if self.train_dataloader_it is None:
                self.train_dataloader_it = iter(self.train_dataloader)
            return next(self.train_dataloader_it)
        except StopIteration:
            self.train_dataloader_it = iter(self.train_dataloader)
            return next(self.train_dataloader_it)

    def _tv_epoch(self) -> int:
        return (self.global_steps - 1) // len(self.train_dataloader)

    def _fetch_one_gen_batch(self):
        if self._tv_next is None:
            self._tv_next = self._tv_raw_batch()
            control.call("warm", tasks=batch_tasks(self._tv_next), wait=True)
        self._tv_state = self.train_dataloader.state_dict()
        current, self._tv_next = self._tv_next, self._tv_raw_batch()
        self._tv_tasks = batch_tasks(current)
        control.call("begin", step=self.global_steps, epoch=self._tv_epoch(), tasks=self._tv_tasks)
        control.call("warm", tasks=batch_tasks(self._tv_next), keep=self._tv_tasks, wait=False)
        current["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(current["raw_prompt"]))], dtype=object)
        return tu.get_tensordict(current)

    def _save_checkpoint(self):
        # veRL saves train_dataloader.state_dict(), which already counts the
        # peeked batch; hand it the state from before the peek instead.
        if self._tv_state is None:
            return super()._save_checkpoint()
        state = self._tv_state
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
