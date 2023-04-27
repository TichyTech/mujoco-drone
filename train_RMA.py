import torch
from ray.rllib.algorithms.ppo import PPOConfig
from environments.BaseDroneEnv import BaseDroneEnv, base_config
from copy import copy
from training import train
import os
from datetime import datetime
from models.PPO.RMA.RMA_model import RMA_full
from environments.rewards import *
from ray.rllib.models import ModelCatalog
from custom_logging import MyCallbacks, custom_logger_creator
from environments.ObservationWrappers import *
from distributions import MyBetaDist
from ray.tune.result import DEFAULT_RESULTS_DIR
from evaluation import load_policy_state


seed = 42

# model configuration
environment = LocalFrameRPYParamsEnv  # observation transform
model = RMA_full  # custom model to use
dist = MyBetaDist
experiment_logdir = 'RMA2'  # name of the directory in ~/ray_results to log to
reward_fcn = distance_energy_reward

# load checkpoint?
checkpoint_dir = 'models/PPO/RMA/checkpoints/'  # directory where to look for checkpoints
checkpoint_to_load = 'checkpoint_000050'  # saved checkpoint name
load_checkpoint = 1

# setting the parameters
ModelCatalog.register_custom_model(model.__name__, model)
ModelCatalog.register_custom_action_dist(dist.__name__, MyBetaDist)
model_config = {
    "custom_model": model.__name__,
    "custom_model_config": {'num_states': 16,
                            'num_params': 6,
                            'num_actions': 4,
                            'param_embed_dim': 32,
                            'train_adaptation': True,
                            'adapt_seq_len': 32
                            },
    "custom_action_dist": dist.__name__,
    # "max_seq_len": 32  # this is to set maximum sequence length for recurrent network observations
}

# training configuration
num_epochs = 500
train_drones = 64  # number of drones per training environment
num_processes = 8  # number parallel envs used for training
rollout_length = 1024  # length of individual rollouts used in training
train_batch_size = num_processes * train_drones * rollout_length  # total length of the training data batch

train_env_config = copy(base_config)
train_env_config['reward_fcn'] = reward_fcn
train_env_config['num_drones'] = train_drones  # set number of drones used per environment for training in parallel
train_env_config['window_title'] = 'training'
train_env_config['regen_env_at_steps'] = 1024  # regenerate simulation after 2000 timesteps
train_env_config['max_steps'] = 1024
train_env_config['train_vis'] = 1   # how many training windows to render and show
train_env_config['seed'] = seed
train_env_config['state_difficulty'] = 0.2
train_env_config['param_difficulty'] = 1

# evaluation environment configuration
eval_env_config = copy(base_config)
eval_env_config['window_title'] = 'evaluation'
eval_env_config['num_drones'] = 1
eval_env_config['controlled'] = True
eval_env_config['max_distance'] = 4
eval_env_config['reward_fcn'] = reward_fcn
eval_env_config['max_steps'] = 2048
eval_env_config['state_difficulty'] = 0.4
eval_env_config['param_difficulty'] = 2.5

# define custom logging dir
timestr = datetime.today().strftime("%d-%m_%H-%M")  # current time
logdir_prefix = f"PPO_{model.__name__}_{environment.__name__}_{timestr}"
logdir = os.path.join(DEFAULT_RESULTS_DIR, experiment_logdir, logdir_prefix)
os.mkdir(logdir)  # create empty directory for logging


# PPO configuration

policy_training_config = PPOConfig() \
    .training(gamma=0.985, lambda_=0.96, lr=0.0001, sgd_minibatch_size=train_batch_size//16, clip_param=0.2,
              train_batch_size=train_batch_size, model=model_config, num_sgd_iter=5, kl_coeff=0) \
    .resources(num_gpus=1) \
    .rollouts(num_rollout_workers=num_processes, rollout_fragment_length=rollout_length)\
    .framework(framework='torch') \
    .environment(env=environment, env_config=train_env_config, normalize_actions=False)\
    .exploration(explore=True, exploration_config={"type": "StochasticSampling", "random_timesteps": (1-load_checkpoint)*10000})\
    .debugging(seed=seed, logger_creator=custom_logger_creator(logdir))\
    .callbacks(callbacks_class=MyCallbacks)\
    .evaluation(evaluation_duration='auto', evaluation_interval=1, evaluation_parallel_to_training=True,
                evaluation_config={'env_config': eval_env_config, 'explore': False}, evaluation_num_workers=1) \


if __name__ == '__main__':
    policy_algo = policy_training_config.build()

    loaded_state = load_policy_state(checkpoint_dir + checkpoint_to_load)
    new_state_dict = policy_algo.get_policy().model.state_dict()  # load state dict
    for (ks, vs) in loaded_state['weights'].items():
        if 'adaptation_module' in ks:  # skip adaptation module
            continue
        new_state_dict[ks].copy_(torch.from_numpy(vs))  # copy all other modules
    policy_algo.get_policy().model.load_state_dict(new_state_dict)
    policy_algo.workers.sync_weights()

    train(policy_algo, 100, checkpoint_dir)
    policy_algo.stop()


