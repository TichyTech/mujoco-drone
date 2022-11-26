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
              'pendulum': True,
              'render_mode': 'human'}

eval_env_config = copy(base_config)
train_env_config = copy(base_config)

model_config = {"fcnet_hiddens": [128, 128]}
# model_config = {
#     "use_attention": True,
#     "max_seq_len": 10,
#     "attention_num_transformer_units": 1,
#     "attention_dim": 16,
#     "attention_memory_inference": 10,
#     "attention_memory_training": 10,
#     "attention_num_heads": 1,
#     "attention_head_dim": 16,
#     "attention_position_wise_mlp_dim": 32,
#     "attention_use_n_prev_actions": 0
#     }
# model_config['use_lstm'] = True
# model_config["max_seq_len"] = 20

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
    eval_env = envs.VecDroneEnv(eval_env_config)

    ep = 0
    algo = algo_config.build()
    # algo.load_checkpoint("C:/Users/tomtc/Desktop/reinforcement_learning/mujoco-drone/models/PPO/checkpoint_000010/checkpoint-10")
    for i in range(200):
        ep = ep + 1
        start = time()
        results = algo.train()
        print("Epoch {:d} took {:.2f} seconds; avg. reward={:.3f}".format(ep, (time() - start),
                                                                              results['episode_reward_mean']))
        if ep % 10 == 0:
            print("Saving checkpoint to {}".format(algo.save("./models/PPO/LSTM")))
            # rollout a trajectory using the learned model
            # for _ in range(eval_rollouts):
            #     obs = eval_env.vector_reset()
            #     eval_env.render()
            #     done = False
            #     total_reward = 0.0
            #     while not done:
            #         action = algo.compute_single_action(obs[0])
            #         obs, reward, done, info = eval_env.vector_step([action])
            #         eval_env.render()
            #         total_reward += reward[0]
            #     print(f"Rollout total-reward={total_reward}")
