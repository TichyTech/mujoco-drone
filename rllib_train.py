from ray.rllib.algorithms.ppo import PPO, PPOConfig
import environments as envs
from time import time
from ray.rllib.models.catalog import MODEL_DEFAULTS
from copy import copy


# config common to both training and evaluation environments

base_config = {'num_drones': 1,
              'reference': [0.5, 0.5, 3.5, 0.4],  # x,y,z,yaw
              'start_pos': [0, 0, 3, 0],
              'max_pos_offset': 0.4,
              'angle_variance': [0.3, 0.3, 0.3],
              'vel_variance': [0.04, 0.04, 0.04],
              'pendulum': False,
              'render_mode': 'human'}

eval_env_config = copy(base_config)
train_env_config = copy(base_config)

model_config = MODEL_DEFAULTS
model_config["fcnet_hiddens"] = [128, 128]

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

    algo_config = PPOConfig()\
        .training(gamma=0.99, lr=0.001, sgd_minibatch_size=train_batch_size // 4,
                  train_batch_size=train_batch_size, model=model_config)\
        .resources(num_gpus=1)\
        .rollouts(num_rollout_workers=num_processes, rollout_fragment_length=rollout_length)\
        .framework(framework='torch')\
        .environment(env=envs.VecDroneEnv, env_config=train_env_config)\

    # create an environment for evaluation
    # eval_env = envs.VecDroneEnv(eval_env_config)

    # train_epochs = [2, 2, 30, 30]
    rollout_lens = [32, 64, 128, 256]
    ep = 0
    algo = algo_config.build()

    # def change_rl(worker):
    #     worker.rollout_fragment_length = rollout_lens[0]
    #     # worker.config['sgd_minibatch_size'] = 8*64*rollout_lens[i]
    #     print(worker.rollout_fragment_length)

    # algo.workers.foreach_worker(change_rl)
    # # algo.workers.local_worker()
    # algo.config = new_config.to_dict()
    # algo.config['rollout_fragment_length'] = rollout_lens[0]
    # algo.config['train_batch_size'] = 8 * 64 * rollout_lens[0]
    # algo.config['sgd_minibatch_size'] = 2 * 64 * rollout_lens[0]
    # algo.config["min_train_timesteps_per_iteration"]
    # algo.load_checkpoint("./models/PPO/checkpoint_000201/checkpoint-201")
    for i in range(200):
        # if i > 0:
        #     weights = algo.get_weights()  # save previous weights
        # new_config = algo_config.training(gamma=0.99, lr=0.001, sgd_minibatch_size=8*64*rollout_lens[i] // 4,
        #           train_batch_size=8*64*rollout_lens[i], model=model_config)\
        #     .rollouts(num_rollout_workers=num_processes, rollout_fragment_length=rollout_lens[i])
        # algo = new_config.build()
        # algo.reset_config(new_config)

        # def change_rl(worker):
        #     worker.rollout_fragment_length = rollout_lens[i]
        #     # worker.config['sgd_minibatch_size'] = 8*64*rollout_lens[i]
        #     print(worker.rollout_fragment_length)
        # algo.workers.foreach_worker(change_rl)
        # # # algo.workers.local_worker()
        # # algo.config = new_config.to_dict()
        # algo.config['train_batch_size'] = 8*64*rollout_lens[0]
        # if i > 0:
        #     algo.set_weights(weights)
        # for _ in range(train_epochs[i]):
        ep = ep + 1
        start = time()
        results = algo.train()
        print("Epoch {:d} took {:.2f} seconds; avg. reward={:.3f}".format(ep, (time() - start),
                                                                              results['episode_reward_mean']))
        if ep % 10 == 0:
            print("Saving checkpoint to {}".format(algo.save("./models/PPO")))
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
