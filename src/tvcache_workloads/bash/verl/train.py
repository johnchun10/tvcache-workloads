"""veRL's main_ppo with TVCache's step hooks (trainer.py), so the fork bank is
warmed one batch ahead and each step boundary is logged for stats.py:

    python -m tvcache_workloads.bash.verl.train <the usual main_ppo overrides> \\
        trainer.v1.trainer_mode=tvcache_sync

The trainer is looked up by name inside veRL's TaskRunner actor, so the actor
imports trainer.py (registering it) before running veRL's own TaskRunner code.
"""

import os

import hydra
import ray
from verl.trainer import main_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device

# The plain class behind veRL's @ray.remote TaskRunnerV1.
_BaseTaskRunner = main_ppo.TaskRunnerV1.__ray_metadata__.modified_class.__mro__[1]


@ray.remote
class TvcacheTaskRunner(_BaseTaskRunner):
    def run(self, config):
        import tvcache_workloads.bash.verl.trainer  # noqa: F401  registers tvcache_sync

        return super().run(config)


@hydra.main(
    config_path=os.path.join(os.path.dirname(main_ppo.__file__), "config"),
    config_name="ppo_trainer",
    version_base=None,
)
def main(config):
    auto_set_device(config)
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    if not config.trainer.use_v1:
        raise SystemExit("tvcache training needs veRL's v1 trainer (trainer.use_v1=True)")
    main_ppo.run_ppo(config, task_runner_class=TvcacheTaskRunner)


if __name__ == "__main__":
    main()
