import gymnasium
from environments.BaseDroneEnv import BaseDroneEnv, base_config
import os
import numpy as np
from ray.rllib.models import ModelCatalog
from models.PPO.MLP.CustomMLP import CustomMLP
import matplotlib.pyplot as plt
from distributions import MyBetaDist, MySquashedGaussian
from environments.ObservationWrappers import LocalFrameRPYEnv
from environments.transformation import mujoco_rpy2quat
import pickle as pkl
from ray.rllib.policy.policy import Policy


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
        obs, reward, done, truncated, info = env.vector_step([action])
        env.render()
        total_reward += reward[0]
    print(f"Rollout total-reward={total_reward}")


def evaluate_trajectory(env, policy, trajectory=[[0, 0, 1, 0]]*400):
    assert trajectory is not None
    lstm = False
    if policy.config['model']['use_lstm']:
        lstm = True
        state = np.zeros((2, policy.config['model']['lstm_cell_size']))  # init state to all zeros
    wp = trajectory[0]  # first waypoint
    pos = wp[:3]
    quat = mujoco_rpy2quat([0, 0, wp[3]])  # get quaternion
    env.reference = wp
    env.move_mocap_to(np.concatenate((pos, quat)), 0)
    obs, _ = env.vector_reset()
    env.render()
    total_reward = 0.0
    observations = [obs[0]]
    prev_action = np.array([0,0,0,0])
    rewards = []
    actions = []
    for x in trajectory:
        if lstm:  # pass the state as well
            action, state, _ = policy.compute_single_action(obs[0], state, prev_action=prev_action)
        else:
            action, _, _ = policy.compute_single_action(obs[0], prev_action=prev_action)
        prev_action = action
        pos = x[:3]
        quat = mujoco_rpy2quat([0, 0, x[3]])  # get quaternion
        env.move_mocap_to(np.concatenate((pos, quat)), 0)
        env.reference = x
        obs, reward, done, truncated, info = env.vector_step([action])
        observations.append(obs[0])
        rewards.append(reward[0])
        actions.append(action)
        env.render()
        total_reward += reward[0]
    return observations, actions, rewards


def gen_circle_trajectory(T=10, f=0.5, r=1, h=1):
    t = np.arange(0, T, 0.01)
    trajectory = np.array([r*np.cos(2 * np.pi * f * t), r*np.sin(2 * np.pi * f * t), h * np.ones_like(t), np.zeros_like(t)]).T
    return t, np.array(trajectory)


def gen_step_trajectory(step_time=5, duration=10, start_pos=[0, 0, 0, 0], end_pos=[0, 0, 1, 0]):
    t = np.arange(0, duration, 0.01)
    trajectory = [start_pos if i < step_time else end_pos for i in t]
    return t, np.array(trajectory)


def gen_ramp_trajectory(start_time=5, duration=10, start_pos=[0, 0, 0, 0], end_pos=[0, 0, 1, 0]):
    t = np.arange(0, duration, 0.01)
    start_pos = np.array(start_pos)
    end_pos = np.array(end_pos)
    trajectory = [start_pos if i < start_time else (start_pos + (i-start_time)/(duration - start_time)*(end_pos - start_pos)) for i in t]
    return t, np.array(trajectory)


def load_policy_state(checkpoint):
    path = os.path.join(checkpoint, 'policies/default_policy/policy_state.pkl')
    with open(path, 'rb') as f:
        policy_state = pkl.load(f)
    return policy_state


model_dir = 'models/PPO/MLP/checkpoints/'
checkpoint_to_load = 'checkpoint_000140'
load_checkpoint = True

# environment configuration
eval_env_config = base_config
eval_env_config['controlled'] = True
eval_env_config['window_title'] = 'evaluation'

# model_config = load_model_config(model_dir)

ModelCatalog.register_custom_model("CustomModel", CustomMLP)
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

obs_space = gymnasium.spaces.Box(-np.inf, np.inf, (18, ))
act_space = gymnasium.spaces.Box(0, 1, (4, ))


if __name__ == '__main__':
    # load torch model
    model = CustomMLP(obs_space, act_space, 4, model_config, 'test_model')
    state = load_policy_state(model_dir + checkpoint_to_load)
    policy = Policy.from_state(state)
    eval_env = LocalFrameRPYEnv(eval_env_config)

    t, trajectory = gen_circle_trajectory()
    trajectory = trajectory + [[0, 0, 15, 0]]
    observations, actions, rewards = evaluate_trajectory(eval_env, policy, trajectory)
    plt.subplot(3, 1, 1)
    plt.plot(t, np.array(observations)[1:, :3], label=['x err', 'y err', 'z err'])
    plt.xlabel('time [s]')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t, trajectory[:, :3], label=['x ref', 'y ref', 'z ref'])
    plt.xlabel('time [s]')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t, np.array(actions), label=['m1', 'm2', 'm3', 'm4'])
    plt.legend()
    plt.show()

