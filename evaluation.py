from ray.rllib.algorithms.ppo import PPOConfig
from environments.VecDrone import VecDrone, base_config
from training import load_model_config
import os
import numpy as np
from ray.rllib.models import ModelCatalog
from models.PPO.RMA.RMA_model import RMA_model
import matplotlib.pyplot as plt


def rollout_trajectory(env, algo):
    lstm = False
    if algo.config['model']['use_lstm']:
        lstm = True
        state = np.zeros((2, algo.config['model']['lstm_cell_size']))  # init state to all zeros
    obs = env.vector_reset()
    env.render()
    done = [False]
    total_reward = 0.0
    while not np.array(done).all():
        if lstm:  # pass the state as well
            action, state, _ = algo.compute_single_action(obs[0], state)
        else:
            action = algo.compute_single_action(obs[0])
        obs, reward, done, info = env.vector_step([action])
        env.render()
        total_reward += reward[0]
    print(f"Rollout total-reward={total_reward}")


def evaluate_trajectory(env, algo, trajectory=[[0, 0, 3, 0]]*400):
    assert trajectory is not None
    lstm = False
    if algo.config['model']['use_lstm']:
        lstm = True
        state = np.zeros((2, algo.config['model']['lstm_cell_size']))  # init state to all zeros
    env.move_reference_to(trajectory[0])
    obs = env.vector_reset()
    env.render()
    total_reward = 0.0
    observations = [obs[0]]
    rewards = []
    for x in trajectory:
        env.move_reference_to(x)
        if lstm:  # pass the state as well
            action, state, _ = algo.compute_single_action(obs[0], state)
        else:
            action = algo.compute_single_action(obs[0])
        obs, reward, done, info = env.vector_step([action])
        observations.append(obs[0])
        rewards.append(reward[0])
        env.render()
        total_reward += reward[0]
    return observations, rewards


def gen_circle_trajectory(T=10, f=0.5, r=1, h=3):
    t = np.arange(0, T, 0.02)
    trajectory = np.array([r*np.cos(2 * np.pi * f * t), r*np.sin(2 * np.pi * f * t), h * np.ones_like(t), np.zeros_like(t)]).T
    return t, np.array(trajectory)


def gen_step_trajectory(step_time=5, duration=10, start_pos=[0, 0, 3, 0], end_pos=[0, 0, 5, 0]):
    t = np.arange(0, duration, 0.02)
    trajectory = [start_pos if i < step_time else end_pos for i in t]
    return t, np.array(trajectory)


def gen_ramp_trajectory(start_time=5, duration=10, start_pos=[0, 0, 3, 0], end_pos=[0, 0, 5, 0]):
    t = np.arange(0, duration, 0.02)
    start_pos = np.array(start_pos)
    end_pos = np.array(end_pos)
    trajectory = [start_pos if i < start_time else (start_pos + (i-start_time)/(duration - start_time)*(end_pos - start_pos)) for i in t]
    return t, np.array(trajectory)


model_dir = 'models/PPO/RMA_model/'
checkpoint_to_load = 'checkpoints/checkpoint_000090'
load_checkpoint = True

# environment configuration
eval_env_config = base_config
eval_env_config['controlled'] = True
eval_env_config['window_title'] = 'evaluation'

# model_config = load_model_config(model_dir)

ModelCatalog.register_custom_model("RMA_model", RMA_model)
model_config = {
    "custom_model": "RMA_model",
}

if __name__ == '__main__':
    algo_config = PPOConfig() \
        .training(model=model_config) \
        .resources(num_gpus=1)\
        .framework(framework='torch') \
        .rollouts(num_rollout_workers=0) \
        .environment(env=VecDrone, env_config=eval_env_config)

    # create an environment for evaluation
    eval_env = VecDrone(eval_env_config)
    algo = algo_config.build()
    if load_checkpoint and os.path.exists(model_dir + checkpoint_to_load):
        algo.restore(model_dir + checkpoint_to_load)
        print('checkpoint from {} loaded'.format(model_dir + checkpoint_to_load))
    # rollout a trajectory using the learned model

    t, trajectory = gen_ramp_trajectory()
    observations, rewards = evaluate_trajectory(eval_env, algo, trajectory)
    plt.figure()
    plt.plot(t, np.array(observations)[1:, :3], label=['x err', 'y err', 'z err'])
    plt.plot(t, trajectory[:, :3], label=['x ref', 'y ref', 'z ref'])
    plt.show()
    plt.xlabel('time [s]')
    plt.legend()

