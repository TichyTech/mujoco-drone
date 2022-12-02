from ray.rllib.algorithms.ppo import PPOConfig
from environments.VecDrone import VecDrone, base_config, distance_time_energy_reward, distance_reward
from ray.rllib.models.catalog import MODEL_DEFAULTS
from copy import copy
from training import train
import os

model_dir = 'models/PPO/MLP/'
checkpoint_to_load = 'checkpoint-000100'

# training configuration
num_epochs = 300
train_vis = True  # toggle training process rendering
train_drones = 64  # number of drones per env
eval_rollouts = 5
num_processes = 8  # number parallel envs used for training
rollout_length = 256  # length of individual episodes used in training
train_batch_size = num_processes * train_drones * rollout_length  # total length of the training data batch

# environment configuration
eval_env_config = copy(base_config)
train_env_config = copy(base_config)
train_env_config['reward_fcn'] = distance_time_energy_reward
train_env_config['num_drones'] = train_drones  # set number of drones used per environment for training in parallel
if not train_vis:
    train_env_config['render_mode'] = None

# model configuration
model_config = MODEL_DEFAULTS  # use default model configuration
model_config = {"fcnet_hiddens": [256, 256]}  # set the layer widths of the MLP


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
    if os.path.exists(model_dir + checkpoint_to_load):
        algo.restore(model_dir + checkpoint_to_load)

    eval_env = VecDrone(eval_env_config)  # create an environment for evaluation
    train(algo, eval_env, num_epochs, eval_rollouts, model_dir)

