from ray.rllib.algorithms.ppo import PPOConfig
from environments.VecDrone import VecDrone, base_config
from ray.rllib.models.catalog import MODEL_DEFAULTS
from training import rollout_trajectory
import os

model_dir = 'models/PPO/LSTM/'
checkpoint_to_load = 'checkpoint_000010'

# environment configuration
eval_env_config = base_config
model_config = MODEL_DEFAULTS
model_config = {"fcnet_hiddens": [256, 256]}

if __name__ == '__main__':
    algo_config = PPOConfig()\
        .resources(num_gpus=1)\
        .framework(framework='torch') \
        .rollouts(num_rollout_workers=0) \
        .environment(env=VecDrone, env_config=eval_env_config)

    # create an environment for evaluation
    eval_env = VecDrone(eval_env_config)
    algo = algo_config.build()
    if os.path.exists(model_dir + checkpoint_to_load):
        algo.restore(model_dir + checkpoint_to_load)
    # rollout a trajectory using the learned model
    for _ in range(100):
        rollout_trajectory(eval_env, algo)
