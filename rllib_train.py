from ray.rllib.algorithms.ppo import PPO
import environments as envs
from time import time
from ray.rllib.models.catalog import MODEL_DEFAULTS

num_drones = 1

eval_env_config = {'num_drones': 1,
              'reference': [0.2, 0, 1.4, 0],
              'start_pos': [0, 0, 1],
              'pendulum': False,
              'render_mode': 'human'}

train_env_config = {'num_drones': 64,
              'reference': [0.2, 0, 1.2, 0],
              'start_pos': [0, 0, 1],
              'pendulum': False,
              'render_mode': 'human'}

env = envs.VecDroneEnv(eval_env_config)
obs = env.vector_reset()
print(obs)
obs = env.vector_step([[0, 0, 0, 0]])
print(obs)

algo = PPO(env=envs.VecDroneEnv, config={
    "framework": 'torch',
    "num_workers": 8,
    # "fcnet_hiddens": [512, 256, 64],
    "num_gpus": 1,
    "gamma": 0.99,
    "lr": 0.0001,
    "rollout_fragment_length": 256,
    "sgd_minibatch_size": 32768,
    "train_batch_size": 131072,
    "model": MODEL_DEFAULTS,
    "env_config": train_env_config,  # config to pass to env class
    }
)

for i in range(1000):
    start = time()
    results = algo.train()
    print('took ', (time() - start), ' seconds')
    print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")
    if i % 10 == 0:
        obs = env.vector_reset()
        env.render()
        done = False
        total_reward = 0.0
        while not done:
            action = algo.compute_single_action(obs[0])
            obs, reward, done, info = env.vector_step([action])
            env.render()
            total_reward += reward[0]
        print(f"Played 1 episode; total-reward={total_reward}")

