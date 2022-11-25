from ray.rllib.algorithms.ppo import PPO
import environments as envs
from time import time
from ray.rllib.models.catalog import MODEL_DEFAULTS
from copy import copy


# config common to both training and evaluation environments

base_config = {'num_drones': 1,
              'reference': [0.2, 0, 1.4, 0],  # x,y,z,yaw
              'start_pos': [0, 0, 1, 0],
              'pos_variance': [0, 0, 0],
              'angle_variance': [0, 0, 0],
              'vel_variance': [0, 0, 0],
              'pendulum': False,
              'render_mode': 'human'}

eval_env_config = copy(base_config)
train_env_config = copy(base_config)

if __name__ == '__main__':

    # training related hyperparameters
    train_vis = True  # toggle training process rendering
    train_drones = 64  # number of drones per env
    eval_rollouts = 5
    num_processes = 8  # number parallel envs used for training
    rollout_length = 256  # length of individual episodes used in training
    train_batch_size = num_processes * train_drones * rollout_length  # total length of the training data batch

    if not train_vis:
        train_env_config['render_mode'] = None
    train_env_config['num_drones'] = train_drones

    algo = PPO(env=envs.VecDroneEnv, config={
        "framework": 'torch',
        "num_workers": num_processes,
        # "fcnet_hiddens": [512, 256, 64],
        "num_gpus": 1,
        "gamma": 0.99,
        "lr": 0.001,
        "rollout_fragment_length": rollout_length,
        "sgd_minibatch_size": train_batch_size // 4,
        "train_batch_size": train_batch_size,
        "model": MODEL_DEFAULTS,
        "env_config": train_env_config,  # config to pass to env class
        }
    )

    # algo.load_checkpoint("./models/PPO/checkpoint_000001")

    # create an environment for evaluation
    # eval_env = envs.VecDroneEnv(eval_env_config)

    # train_epochs = [30, 30, 50, 50]
    # rollout_lens = [64, 128, 256, 512]
    for i in range(100):
        start = time()
        results = algo.train()
        print("Epoch {:d} took {:.2f} seconds; avg. reward={:.3f}".format(i, (time() - start),
                                                                          results['episode_reward_mean']))
        # if i % 10 == 0:
        #     print("Saving checkpoint to {}".format(algo.save("./models/PPO")))
        #     # rollout a trajectory using the learned model
        #     for _ in range(eval_rollouts):
        #         obs = eval_env.vector_reset()
        #         eval_env.render()
        #         done = False
        #         total_reward = 0.0
        #         while not done:
        #             action = algo.compute_single_action(obs[0])
        #             obs, reward, done, info = eval_env.vector_step([action])
        #             eval_env.render()
        #             total_reward += reward[0]
        #         print(f"Rollout total-reward={total_reward}")
