from ray.rllib.algorithms.ppo import PPOConfig
from environments.BaseDroneEnv import BaseDroneEnv, base_config
from copy import copy
from training import train
import os
from datetime import datetime
import tempfile
from models.PPO.MLP.CustomMLP import CustomMLP
from models.PPO.SimpleMLP.SimpleMLP import SimpleMLPmodel2
from models.PPO.CustomLSTM.CustomLSTM import CustomLSTM
from environments.rewards import reward_1, default_reward_fcn, reward_pendulum_dist
from ray.rllib.models import ModelCatalog
from custom_logging import MyCallbacks, custom_logger_creator
from environments.ObservationWrappers import LocalFrameRPYEnv
from distributions import MyBetaDist, MySquashedGaussian
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.result import DEFAULT_RESULTS_DIR


# model configuration
environment = LocalFrameRPYEnv  # observation transform
model = CustomMLP  # custom model to use
experiment_logdir = 'MLP'  # name of the directory in ~/ray_results to log to

# load checkpoint?
checkpoint_dir = 'models/PPO/MLP/checkpoints'  # directory where to look for checkpoints
checkpoint_to_load = 'checkpoint_000020'  # saved checkpoint name
load_checkpoint = 1

# setting the parameters
ModelCatalog.register_custom_model("CustomModel", model)
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

# training configuration
num_epochs = 500
train_drones = 64  # number of drones per training environment
num_processes = 8  # number parallel envs used for training
rollout_length = 512  # length of individual rollouts used in training
train_batch_size = num_processes * train_drones * rollout_length  # total length of the training data batch

train_env_config = copy(base_config)
train_env_config['reward_fcn'] = reward_1
train_env_config['num_drones'] = train_drones  # set number of drones used per environment for training in parallel
train_env_config['window_title'] = 'training'
train_env_config['regen_env_at_steps'] = 2048  # regenerate simulation after 2000 timesteps
train_env_config['max_steps'] = 1024
train_env_config['train_vis'] = 1   # how many training windows to render and show

# evaluation environment configuration
eval_env_config = copy(base_config)
eval_env_config['window_title'] = 'evaluation'
eval_env_config['num_drones'] = 1
eval_env_config['controlled'] = True
eval_env_config['max_distance'] = 3
eval_env_config['reward_fcn'] = reward_1

# define custom logging dir
timestr = datetime.today().strftime("%d-%m_%H-%M")  # current time
logdir_prefix = f"PPO_{str(model.__name__)}_{environment.__name__}_{timestr}"
logdir = os.path.join(DEFAULT_RESULTS_DIR, experiment_logdir, logdir_prefix)
os.mkdir(logdir)  # create empty directory for logging

# PPO configuration
algo_config = PPOConfig() \
    .training(gamma=0.985, lambda_=0.98, lr=0.001, sgd_minibatch_size=train_batch_size//4, clip_param=0.2,
              train_batch_size=train_batch_size, model=model_config, num_sgd_iter=20) \
    .resources(num_gpus=1) \
    .rollouts(num_rollout_workers=num_processes, rollout_fragment_length=rollout_length)\
    .framework(framework='torch') \
    .environment(env=environment, env_config=train_env_config, normalize_actions=False)\
    .exploration(explore=True, exploration_config={"type": "StochasticSampling", "random_timesteps": 10000})\
    .debugging(seed=42, logger_creator=custom_logger_creator(logdir))\
    .callbacks(callbacks_class=MyCallbacks)\
    .evaluation(evaluation_duration='auto', evaluation_interval=1, evaluation_parallel_to_training=True,
                evaluation_config={'env_config': eval_env_config, 'explore': False}, evaluation_num_workers=1) \

# pbt = PopulationBasedTraining(
#     time_attr="time_total_s",
#     perturbation_interval=120,
#     resample_probability=0.25,
#     hyperparam_mutations={
#         "gamma": tune.uniform(0.9, 1.0),
#         "clip_param": tune.uniform(0.01, 0.4),
#         "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
#         "num_sgd_iter": tune.randint(1, 30),
#         "sgd_minibatch_size": tune.randint(512, 4*16384),
#         "train_batch_size": tune.randint(10000, 160000),
#     }
# )
#
# tuner = tune.Tuner(
#     "PPO",
#     tune_config=tune.TuneConfig(
#         metric="episode_reward_mean",
#         mode="max",
#         scheduler=pbt,
#         num_samples=4,
#     ),
#     param_space=algo_config.to_dict()
# )


if __name__ == '__main__':
    algo = algo_config.build()
    if load_checkpoint and os.path.exists(checkpoint_dir + checkpoint_to_load):
        algo.restore(checkpoint_dir + checkpoint_to_load)
        print('checkpoint from {} loaded'.format(checkpoint_dir + checkpoint_to_load))

    # eval_env = VecDrone(eval_env_config)  # create an environment for evaluation
    train(algo, num_epochs, checkpoint_dir)

    # results = tuner.fit()
    # print("best hyperparameters: ", results.get_best_result().config)

    algo.cleanup()

