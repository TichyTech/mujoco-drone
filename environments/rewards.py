import numpy as np
from .transformation import mujoco_rpy2quat, mujoco_quat2DCM


def default_reward_fcn(env, state, action, num_steps):
    # penalize distance from reference
    ref = env.reference
    pos_err = np.linalg.norm(state[:3] - ref[:3])
    reward = 3 - pos_err
    return reward


def distance_reward_fcn(env, state, action, num_steps):
    # penalize distance from reference
    ref = env.reference
    heading_err = np.linalg.norm(state[5] - ref[3])
    heading_err = abs((heading_err + np.pi) % (2 * np.pi) - np.pi)
    pos_err = np.linalg.norm(state[:3] - ref[:3])
    reward = 5 - pos_err - heading_err
    return reward


def distance_energy_reward(env, state, action, num_steps):
    # penalize distance and action magnitude
    ref = env.reference
    heading_err = np.linalg.norm(state[5] - ref[3])
    heading_err = abs((heading_err + np.pi) % (2 * np.pi) - np.pi)
    pos_err = ((state[:3] - ref[:3]) ** 2).sum()
    ctrl_effort = (np.array(action) ** 2).sum()
    reward = - pos_err - 0.5*heading_err - 0.2*ctrl_effort
    return reward


def distance_time_energy_reward(env, state, action, num_steps):
    # penalize distance weighted by time steps and action magnitude
    ref = env.reference
    heading_err = np.linalg.norm(state[5] - ref[3])
    heading_err = abs((heading_err + np.pi) % (2 * np.pi) - np.pi)
    pos_err = ((state[:3] - ref[:3]) ** 2).sum()
    ctrl_effort = (np.array(action) ** 2).sum()
    too_far = (pos_err > env.max_distance ** 2)
    reward = - (1 + num_steps//50)*pos_err - 500*too_far - heading_err - 0.02*ctrl_effort
    return reward


def reward_1(env, state, action, num_steps):
    ref = env.reference
    heading_err = np.linalg.norm(state[5] - ref[3])
    heading_err = ((heading_err + np.pi) % (2 * np.pi) - np.pi)**2
    tilt_mag = (np.array(state[3:5]) ** 2).sum()
    pos_err = ((state[:3] - ref[:3]) ** 2).sum()
    close_enough = pos_err < 0.2
    ctrl_effort = (np.array(action) ** 2).sum()
    rot_energy = (np.array(state[6:9])**2).sum()
    pendulum_energy = (np.array(state[15:18])**2).sum()
    too_far = pos_err > env.max_distance**2 - 3
    reward = (7 + 20*close_enough -3*pos_err*(1 + num_steps/150) - 10*too_far -0.3*tilt_mag - 0.7*heading_err - 0.3*ctrl_effort - 0.3*rot_energy - 0.5*pendulum_energy)/10
    return reward


# def reward_pendulum_dist(env, state, action, num_steps):
#     ref = env.reference
#     heading_err = np.linalg.norm(state[5] - ref[3])
#     heading_err = ((heading_err + np.pi) % (2 * np.pi) - np.pi)**2
#     tilt_mag = (np.array(state[3:5]) ** 2).sum()
#
#     pendulum_rpy = state[12:15]
#     drone_R = mujoco_quat2DCM(mujoco_rpy2quat(state[3:6]))  # drone rotation in world frame
#     pendulum_R = mujoco_quat2DCM(mujoco_rpy2quat(pendulum_rpy))  # pendulum rotation in drone frame
#     params = state[25:]
#     pendulum_end = np.array([[0], [0], [-params[5]]])  # pendulum end in pendulum frame
#     R = drone_R @ pendulum_R
#     pendulum_pos = state[:3] + (R @ pendulum_end)[:, 0]
#     pos_err = ((pendulum_pos - ref[:3]) ** 2).sum()
#     close_enough = pos_err < 0.2
#     ctrl_effort = (np.array(action) ** 2).sum()
#     rot_energy = (np.array(state[6:9])**2).sum()
#     pendulum_energy = (np.array(state[15:18])**2).sum()
#     too_far = pos_err > env.max_distance**2 - 3
#     reward = (7 + 20*close_enough -3*pos_err*(1 + num_steps/150) - 10*too_far -0.3*tilt_mag - 0.7*heading_err - 0.3*ctrl_effort - 0.3*rot_energy - 0.5*pendulum_energy)/10
#     return reward


def reward_pendulum_dist(env, state, action, num_steps):
    ref = env.reference
    pendulum_rpy = state[12:15]
    drone_R = mujoco_quat2DCM(mujoco_rpy2quat(state[3:6]))  # drone rotation in world frame
    pendulum_R = mujoco_quat2DCM(mujoco_rpy2quat(pendulum_rpy))  # pendulum rotation in drone frame
    params = state[25:]
    pendulum_end = np.array([[0], [0], [-params[5]]])  # pendulum end in pendulum frame
    R = drone_R @ pendulum_R
    pendulum_pos = state[:3] + (R @ pendulum_end)[:, 0]
    pos_err = ((pendulum_pos - ref[:3]) ** 2).sum()
    reward = - pos_err
    return reward