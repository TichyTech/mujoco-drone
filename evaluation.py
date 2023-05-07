import gymnasium
from environments.BaseDroneEnv import BaseDroneEnv, base_config
import os
import numpy as np
from ray.rllib.models import ModelCatalog
from models.PPO.MLP.CustomMLP import CustomMLP
from models.PPO.RMA.RMA_model import RMA_model_smaller2
from models.PPO.CustomLSTM.StateEstimatorLSTM import CNNestimator
import matplotlib.pyplot as plt
from distributions import MyBetaDist, MySquashedGaussian
from environments.observation_wrappers import LocalFramePRYEnv, LocalFrameRPYParamsEnv, LocalFrameRPYFakeParamsEnv, LocalFrameFullStateEnv
from environments.transformation import mujoco_rpy2quat
import pickle as pkl
from ray.rllib.policy.policy import Policy
from environments.rewards import *
from copy import copy


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


def evaluate_trajectory_lstmest(env, policy, trajectory=[[0, 0, 1, 0]]*400):
    assert trajectory is not None
    state = np.zeros((2, policy.config['model']['lstm_cell_size']))  # init state to all zeros
    wp = trajectory[0]  # first waypoint
    pos = wp[:3]
    quat = mujoco_rpy2quat([0, 0, wp[3]])  # get quaternion
    env.reference = wp
    env.move_mocap_to(np.concatenate((pos, quat)), 0)
    obs, _ = env.vector_reset()
    env.render()
    total_reward = 0.0
    obs_history = np.zeros((1, policy.model.seq_len, policy.model.num_states))  # create state history
    obs_history[0, -1] = obs[0]  # init last observation
    action_history = np.zeros((1, policy.model.seq_len, policy.model.num_actions))  # create state history
    observations = [obs[0]]
    rewards = []
    actions = []
    for x in trajectory:
        print('states', env.reference, env.states[0][:12])
        print('obs', obs[0])
        # compute action
        if (policy.model.train_estimator or policy.model.use_estimate) and policy.model.seq_len > 1:
            indict = {'obs': obs[0], 'obs_history': obs_history, 'action_history': action_history}
        else:
            indict = {'obs': obs[0], 'obs_history': obs_history[:, -1], 'action_history': action_history[:, -1]}
        action, state, _ = policy.compute_actions_from_input_dict(input_dict=indict)
        print('action', action[0])
        # update reference
        # pos = x[:3]
        # quat = mujoco_rpy2quat([0, 0, x[3]])  # get quaternion
        # env.move_mocap_to(np.concatenate((pos, quat)), 0)
        # env.reference = x
        # step

        state = obs[0]
        drone_R = mujoco_quat2DCM(mujoco_rpy2quat(state[3:6]))  # drone rotation in world frame
        pry = state[3:6]
        rpy = [pry[1], pry[0], pry[2]]
        env.move_mocap_to(np.concatenate(([0, 0, 0], mujoco_rpy2quat([0,0,0]))), 1)
        env.move_mocap_to(np.concatenate(([0,0,1], mujoco_rpy2quat(rpy))), 2)

        obs, reward, done, truncated, info = env.vector_step(action)
        # print(obs[0][3:6])
        # move the observations and actions backwards and add new ones in the histories
        for i in range(policy.model.seq_len - 1):
            obs_history[0, i] = obs_history[0, i+1]
            action_history[0, i] = action_history[0, i+1]
        obs_history[0, -1] = obs[0]
        action_history[0, -1] = action[0]

        # logging
        observations.append(obs[0])
        rewards.append(reward[0])
        actions.append(action[0])
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


environment = LocalFrameFullStateEnv  # observation transform
reward_fcn = distance_energy_reward_pendulum_angle
model = CNNestimator
dist = MyBetaDist

checkpoint_dir = 'models/PPO/CustomLSTM/checkpoints/'  # directory where to look for checkpoints
checkpoint_to_load = 'checkpoint_000120'  # saved checkpoint name
load_checkpoint = 1

# environment configuration
eval_env_config = copy(base_config)
eval_env_config['random_params'] = False
eval_env_config['random_start_pos'] = False
eval_env_config['start_pos'] = [0, 0, 15, np.pi]
eval_env_config['num_drones'] = 1
eval_env_config['controlled'] = True
eval_env_config['max_distance'] = 3
eval_env_config['reward_fcn'] = reward_fcn
eval_env_config['max_steps'] = 2048
eval_env_config['state_difficulty'] = 0.4
eval_env_config['param_difficulty'] = 2.5
eval_env_config['window_title'] = 'trajectory_test'
eval_env_config['mocaps'] = 3

ModelCatalog.register_custom_model(model.__name__, model)
ModelCatalog.register_custom_action_dist(dist.__name__, dist)
model_config = {
    "custom_model": model.__name__,
    "custom_model_config": {'num_states': 23,
                            'num_params': 0,
                            'num_actions': 4,
                            'param_embed_dim': 32,
                            'use_estimate': False,
                            'train_estimator': False
                            },
    "custom_action_dist": dist.__name__,
    "max_seq_len": 32  # this is to set maximum sequence length for recurrent network observations
}
obs_space = gymnasium.spaces.Box(-np.inf, np.inf, (18, ))
act_space = gymnasium.spaces.Box(0, 1, (4, ))

# import matplotlib; matplotlib.use('Qt5Agg')
if __name__ == '__main__':
    # load torch model
    # model = RMA_model_smaller2(obs_space, act_space, 4, model_config, 'test_model')
    state = load_policy_state(checkpoint_dir + checkpoint_to_load)
    policy = Policy.from_state(state)
    eval_env = environment(eval_env_config)

    t, trajectory = gen_ramp_trajectory(duration=20)
    trajectory = trajectory + [[0, 0, 15, np.pi]]
    observations, actions, rewards = evaluate_trajectory_lstmest(eval_env, policy, trajectory[:5])

    # plt.figure()
    # plt.subplot(4, 1, 1)
    # plt.plot(t, np.array(observations)[1:, :3], label=['x err', 'y err', 'z err'])
    # plt.xlabel('time [s]')
    # plt.legend()

    # plt.subplot(4, 1, 2)
    # plt.plot(t, np.array(observations)[1:, 3:6], label=['p err', 'r err', 'y err'])
    # plt.plot(t[:10], np.array(observations)[1:, :12], label=list(range(12)))
    # plt.xlabel('time [s]')
    # plt.legend()

    # plt.subplot(4, 1, 3)
    # plt.plot(t, trajectory[:, :3], label=['x ref', 'y ref', 'z ref'])
    # plt.xlabel('time [s]')
    # plt.legend()
    #
    # plt.subplot(4, 1, 4)
    # plt.plot(t, np.array(actions), label=['m1', 'm2', 'm3', 'm4'])
    # plt.legend()
    # plt.show()
    #
    # plt.figure()
    # plt.plot(t[:10], np.array(observations)[1:, 12:], label=list(range(12, 23)))
    # plt.xlabel('time [s]')
    # plt.legend()
    # plt.show()

