from ray.rllib.algorithms.ppo import PPOConfig
from environments.VecDrone import VecDrone, base_config, distance_time_energy_reward, distance_reward
from ray.rllib.models.catalog import MODEL_DEFAULTS
from copy import copy
from training import train, save_model_config
import os
import numpy as np


def custom_reward(env, state, action, num_steps):
    # penalize distance weighted by time steps and action magnitude
    ref = env.reference
    heading_err = np.linalg.norm(state[5] - ref[3])
    heading_err = ((heading_err + np.pi) % (2 * np.pi) - np.pi)**2
    pos_err = ((state[:3] - ref[:3]) ** 2).sum()
    ctrl_effort = (np.array(action) ** 2).sum()
    too_far = pos_err > env.max_distance**2
    reward = 0.05*num_steps - (1 + num_steps/50)*(pos_err + 0.2*heading_err) - 0.04*ctrl_effort - 100*too_far*(pos_err-env.max_distance**2)
    reward += 10*(pos_err < 0.1)*(0.1 - pos_err)
    return reward


model_dir = 'models/PPO/MLP/'
checkpoint_to_load = 'checkpoints/checkpoint_000060'

# training configuration
num_epochs = 300
train_vis = True  # toggle training process rendering
train_drones = 64  # number of drones per env
num_processes = 8  # number parallel envs used for training
rollout_length = 256  # length of individual episodes used in training
train_batch_size = num_processes * train_drones * rollout_length  # total length of the training data batch

# environment configuration
eval_env_config = copy(base_config)
eval_env_config['window_title'] = 'evaluation'
eval_env_config['num_drones'] = 1
eval_env_config['controlled'] = True
eval_env_config['max_distance'] = 3
eval_env_config['reward_fcn'] = custom_reward

train_env_config = copy(base_config)
train_env_config['reward_fcn'] = custom_reward
train_env_config['num_drones'] = train_drones  # set number of drones used per environment for training in parallel
train_env_config['window_title'] = 'training'
if not train_vis:
    train_env_config['render_mode'] = None

# model configuration
model_config = MODEL_DEFAULTS  # use default model configuration
model_config = {"fcnet_hiddens": [512, 256, 256, 64]}  # set the layer widths of the MLP
save_model_config(model_dir, model_config)

# PPO configuration
algo_config = PPOConfig() \
    .training(gamma=0.99, lr=0.001, sgd_minibatch_size=train_batch_size // 4,
              train_batch_size=train_batch_size, model=model_config) \
    .resources(num_gpus=1) \
    .rollouts(num_rollout_workers=num_processes, rollout_fragment_length=rollout_length, recreate_failed_workers=True) \
    .framework(framework='torch') \
    .environment(env=VecDrone, env_config=train_env_config)\
    .evaluation(evaluation_duration='auto', evaluation_interval=1, evaluation_parallel_to_training=True,
                evaluation_config={'env_config': eval_env_config, 'explore': False}, evaluation_num_workers=1)


if __name__ == '__main__':
    algo = algo_config.build()
    if os.path.exists(model_dir + checkpoint_to_load):
        algo.restore(model_dir + checkpoint_to_load)
        print('checkpoint from {} loaded'.format(model_dir + checkpoint_to_load))

    # eval_env = VecDrone(eval_env_config)  # create an environment for evaluation
    train(algo, num_epochs, model_dir)

