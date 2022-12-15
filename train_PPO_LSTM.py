from ray.rllib.algorithms.ppo import PPOConfig
from environments.VecDrone import VecDrone, base_config, distance_time_energy_reward, distance_reward
from ray.rllib.models.catalog import MODEL_DEFAULTS
from copy import copy
from training import train, save_model_config
import numpy as np
import os


def lstm_reward(env, state, action, num_steps):
    # penalize distance weighted by time steps and action magnitude
    ref = env.reference
    heading_err = np.linalg.norm(state[5] - ref[3])
    heading_err = abs((heading_err + np.pi) % (2 * np.pi) - np.pi)
    pos_err = ((state[:3] - ref[:3]) ** 2).sum()
    ctrl_effort = (np.array(action) ** 2).sum()
    too_far = (pos_err > env.max_distance ** 2)
    reward = 0.01*num_steps - (1 + num_steps//50)*pos_err - 500*too_far - heading_err - 0.02*ctrl_effort
    return reward


model_dir = 'models/PPO/LSTM/'
checkpoint_to_load = 'checkpoints/checkpoint_000660'

# training configuration
num_epochs = 300
train_vis = True  # toggle training process rendering
train_drones = 64  # number of drones per env
eval_rollouts = 5
num_processes = 2  # number parallel envs used for training
rollout_length = 256  # length of individual episodes used in training
train_batch_size = num_processes * train_drones * rollout_length  # total length of the training data batch

# environment configuration
eval_env_config = copy(base_config)
train_env_config = copy(base_config)
train_env_config['num_drones'] = train_drones  # set number of drones used per environment for training in parallel
train_env_config['reward_fcn'] = lstm_reward
train_env_config['pendulum'] = False
if not train_vis:
    train_env_config['render_mode'] = None

# model configuration
model_config = MODEL_DEFAULTS  # use default model configuration
model_config = {"fcnet_hiddens": [64, 64]}  # set the layer widths of the MLP
model_config['use_lstm'] = True
model_config['lstm_cell_size'] = 64
model_config["max_seq_len"] = 64

save_model_config(model_dir, model_config)

# PPO configuration
algo_config = PPOConfig() \
    .training(gamma=0.99, lr=0.001, sgd_minibatch_size=train_batch_size // 4,
              train_batch_size=train_batch_size, model=model_config) \
    .resources(num_gpus=1) \
    .rollouts(num_rollout_workers=num_processes, rollout_fragment_length=rollout_length, recreate_failed_workers=True) \
    .framework(framework='torch') \
    .environment(env=VecDrone, env_config=train_env_config)

if __name__ == '__main__':

    algo = algo_config.build()
    # print(algo.workers.local_worker().policy_map['default_policy'].model)
    # print(algo.workers.local_worker().policy_map['default_policy'].__dict__)
    # print(algo.get_policy().model)

    if os.path.exists(model_dir + checkpoint_to_load):
        algo.restore(model_dir + checkpoint_to_load)
        print('checkpoint from {} loaded'.format(model_dir + checkpoint_to_load))

    eval_env = VecDrone(eval_env_config)  # create an environment for evaluation
    train(algo, eval_env, num_epochs, model_dir, eval_rollouts=eval_rollouts)
