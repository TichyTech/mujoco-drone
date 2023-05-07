import pickle
import gymnasium

from environments.BaseDroneEnv import BaseDroneEnv, base_config
from ray.rllib.models import ModelCatalog
from models.PPO.RMA.RMA_model import RMA_model, RMA_model_smaller2, RMA_full
from distributions import MyBetaDist, MySquashedGaussian
from environments.observation_wrappers import LocalFrameRPYParamsEnv, LocalFrameRPYFakeParamsEnv, LocalFramePRYaccEnv
from ray.rllib.policy.policy import Policy
from evaluation import load_policy_state
from environments.rewards import *


environment = LocalFramePRYaccEnv  # observation transform
reward_fcn = distance_energy_reward
model = RMA_full
dist = MyBetaDist
num_drones = 8

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
eval_env_config['param_difficulty'] = 1
eval_env_config['param_difficulty'] = 1
eval_env_config['window_title'] = 'rollout'


ModelCatalog.register_custom_model(model.__name__, model)
ModelCatalog.register_custom_action_dist(dist.__name__, dist)
model_config = {
    "custom_model": model.__name__,
    "custom_model_config": {'num_states': 16,
                            'num_params': 6,
                            'num_actions': 0,
                            'param_embed_dim': 32
                            },
    "custom_action_dist": dist.__name__,
    # "max_seq_len": 32  # this is to set maximum sequence length for recurrent network observations
}

obs_space = gymnasium.spaces.Box(-np.inf, np.inf, (22, ))
act_space = gymnasium.spaces.Box(0, 1, (4, ))


if __name__ == '__main__':
    # load torch model
    # model = RMA_model(obs_space, act_space, 4, model_config, 'test_model')
    state = load_policy_state(checkpoint_dir + checkpoint_to_load)
    policy = Policy.from_state(state)
    eval_env = environment(eval_env_config)

    obs, _ = eval_env.vector_reset()
    eval_env.render()
    prev_actions = np.zeros((num_drones, 4))
    rollout_length = 512
    num_batches = 512
    batches = []
    for j in range(num_batches):
        obs = eval_env.reset_model(regen=True)
        action_batch = []
        obs_batch = []
        truncated_batch = []
        for i in range(rollout_length):
            actions, _, _ = policy.compute_actions(obs, prev_action_batch=prev_actions)
            prev_actions = actions
            obs, reward, done, truncated, info = eval_env.vector_step(actions)

            action_batch.append(actions)
            obs_batch.append(obs)
            truncated_batch.append(truncated)
        z = policy.model.z
        batches.append({'z': z, 'o': obs_batch, 'a': action_batch, 't': truncated_batch})

    with open('dataset.pickle', 'wb') as f:
        pickle.dump(batches, f)

