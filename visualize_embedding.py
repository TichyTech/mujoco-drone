import numpy as np
import torch

from environments.BaseDroneEnv import BaseDroneEnv, base_config
from ray.rllib.models import ModelCatalog
from models.PPO.RMA.RMA_model import RMA_model, RMA_model_smaller2, RMA_full
from distributions import MyBetaDist, MySquashedGaussian
from environments.ObservationWrappers import LocalFrameRPYParamsEnv, LocalFrameRPYFakeParamsEnv
from ray.rllib.policy.policy import Policy
from evaluation import load_policy_state
from environments.rewards import *
import matplotlib.pyplot as plt


environment = LocalFrameRPYParamsEnv  # observation transform
reward_fcn = distance_energy_reward_pendulum_angle
model = RMA_full
dist = MyBetaDist
num_drones = 256

checkpoint_dir = 'models/PPO/RMA/checkpoints/'  # directory where to look for checkpoints
checkpoint_to_load = 'checkpoint_000050'  # saved checkpoint name
load_checkpoint = 1


# environment configuration
eval_env_config = base_config
eval_env_config['num_drones'] = num_drones
eval_env_config['controlled'] = True
eval_env_config['max_distance'] = 3
eval_env_config['reward_fcn'] = reward_fcn
eval_env_config['max_steps'] = 4096
eval_env_config['state_difficulty'] = 0.3
eval_env_config['param_difficulty'] = 2

ModelCatalog.register_custom_model(model.__name__, model)
ModelCatalog.register_custom_action_dist(dist.__name__, dist)


if __name__ == '__main__':
    state = load_policy_state(checkpoint_dir + checkpoint_to_load)
    policy = Policy.from_state(state)

    eval_env = environment(eval_env_config)

    p_enc = policy.model.param_encoder
    params = torch.from_numpy(np.array(eval_env.get_drone_states()))[:, -6:]
    encodings = p_enc(params.to('cuda').float())
    encodings = encodings.cpu().detach().numpy()
    print(encodings)
    for i in range(encodings.shape[1]):
        print('min i ', np.min(encodings[:, i]))
        print('mean i ', np.mean(encodings[:, i]))
        print('max i ', np.max(encodings[:, i]))
        print('var i ', np.var(encodings[:, i]))

    plt.errorbar(np.arange(encodings.shape[1]), np.mean(encodings, axis=0), yerr=np.abs(np.max(encodings, axis=0) - np.min(encodings, axis=0))/2, xerr=0.2, fmt="o")
    plt.show()
