import gymnasium
from environments.BaseDroneEnv import BaseDroneEnv, base_config
import os
import numpy as np
from ray.rllib.models import ModelCatalog
from models.PPO.MLP.CustomMLP import CustomMLP
import matplotlib.pyplot as plt
from distributions import MyBetaDist, MySquashedGaussian
from environments.ObservationWrappers import LocalFrameRPYEnv
from environments.transformation import mujoco_rpy2quat
import pickle as pkl
from ray.rllib.policy.policy import Policy
from evaluation import load_policy_state


model_dir = 'models/PPO/MLP/checkpoints/'
checkpoint_to_load = 'checkpoint_000120'
load_checkpoint = True

# environment configuration
eval_env_config = base_config
eval_env_config['controlled'] = True
eval_env_config['window_title'] = 'evaluation'
eval_env_config['max_steps'] = 4096

ModelCatalog.register_custom_model("CustomModel", CustomMLP)
ModelCatalog.register_custom_action_dist('MyBetaDist', MyBetaDist)
model_config = {
    "custom_model": "CustomModel",
    "custom_model_config": {'num_states': 18,
                            'num_params': 0,
                            'num_actions': 4
                            },
    "custom_action_dist": "MyBetaDist",
    # "max_seq_len": 32  # this is to set maximum sequence length for recurrent network observations
}

obs_space = gymnasium.spaces.Box(-np.inf, np.inf, (18, ))
act_space = gymnasium.spaces.Box(0, 1, (4, ))


if __name__ == '__main__':
    # load torch model
    model = CustomMLP(obs_space, act_space, 4, model_config, 'test_model')
    state = load_policy_state(model_dir + checkpoint_to_load)
    policy = Policy.from_state(state)
    eval_env = LocalFrameRPYEnv(eval_env_config)

    obs, _ = eval_env.vector_reset()
    eval_env.render()
    prev_action = np.array([0, 0, 0, 0])
    while 1:
        action, _, _ = policy.compute_single_action(obs[0], prev_action=prev_action)
        prev_action = action
        obs, reward, done, truncated, info = eval_env.vector_step([action])
