from ray.rllib.algorithms.ppo import PPOConfig
from environments.BaseDroneEnv import BaseDroneEnv, base_config
from copy import copy
from training import train
import os
from models.PPO.MLP.CustomMLP import CustomMLP
from models.PPO.CustomLSTM.CustomLSTM import CustomLSTM
from environments.rewards import reward_2
from ray.rllib.models import ModelCatalog
from custom_logging.CustomCallback import MyCallbacks
from environments.wrappers.LocalFrameRPYacc import LocalFrameRPYaccEnv
from distributions import MyBetaDist


# define used model
model_dir = 'models/PPO/MLP/'
checkpoint_to_load = 'checkpoints/checkpoint_000270'
load_checkpoint = False

# model configuration
ModelCatalog.register_custom_model("CustomLSTM", CustomLSTM)
ModelCatalog.register_custom_action_dist('MyBetaDist', MyBetaDist)
model_config = {
    "custom_model": "CustomLSTM",
    "custom_model_config": {'num_states': 21,
                            'num_params': 0,
                            'num_actions': 4
                            },
    # "custom_action_dist": "MyBetaDist"
}

# training configuration
num_epochs = 500
train_drones = 64  # number of drones per training environment
num_processes = 8  # number parallel envs used for training
rollout_length = 512  # length of individual rollouts used in training
train_batch_size = num_processes * train_drones * rollout_length  # total length of the training data batch

train_env_config = copy(base_config)
train_env_config['reward_fcn'] = reward_2
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
eval_env_config['reward_fcn'] = reward_2

# PPO configuration
algo_config = PPOConfig() \
    .training(gamma=0.985, lr=0.002, sgd_minibatch_size=train_batch_size//4, clip_param=0.2,
              train_batch_size=train_batch_size, model=model_config, num_sgd_iter=15) \
    .resources(num_gpus=1) \
    .rollouts(num_rollout_workers=num_processes, rollout_fragment_length=rollout_length)\
    .framework(framework='torch') \
    .environment(env=LocalFrameRPYaccEnv, env_config=train_env_config, normalize_actions=True)\
    .exploration(explore=True, exploration_config={"type": "StochasticSampling", "random_timesteps": 5000})\
    .debugging(seed=42)\
    .callbacks(callbacks_class=MyCallbacks)\
    .evaluation(evaluation_duration='auto', evaluation_interval=1, evaluation_parallel_to_training=True,
                evaluation_config={'env_config': eval_env_config, 'explore': False}, evaluation_num_workers=1)\


# pbt = PopulationBasedTraining(
#     time_attr="time_total_s",
#     perturbation_interval=120,
#     resample_probability=0.25,
#     hyperparam_mutations={
#         "gamma": tune.uniform(0.9, 1.0),
#         "clip_param": tune.uniform(0.01, 0.5),
#         "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
#         "num_sgd_iter": tune.randint(1, 30),
#         "sgd_minibatch_size": tune.randint(512, 4*16384),
#         "train_batch_size": tune.randint(10000, 160000),
#     }
# )

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

# results = tuner.fit()
# print("best hyperparameters: ", results.get_best_result().config)

if __name__ == '__main__':
    algo = algo_config.build()
    if load_checkpoint and os.path.exists(model_dir + checkpoint_to_load):
        algo.restore(model_dir + checkpoint_to_load)
        print('checkpoint from {} loaded'.format(model_dir + checkpoint_to_load))

    # eval_env = VecDrone(eval_env_config)  # create an environment for evaluation
    train(algo, num_epochs, model_dir)
    algo.cleanup()

