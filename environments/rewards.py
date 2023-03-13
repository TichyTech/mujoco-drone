import numpy as np


def default_reward_fcn(env, state, action, num_steps):
    # penalize distance from reference
    ref = env.reference
    heading_err = np.linalg.norm(state[5] - ref[3])
    heading_err = abs((heading_err + np.pi) % (2 * np.pi) - np.pi)
    pos_err = np.linalg.norm(state[:3] - ref[:3])
    too_far = pos_err > env.max_distance
    reward = - pos_err - 200*too_far - heading_err
    return reward


def distance_energy_reward(env, state, action, num_steps):
    # penalize distance and action magnitude
    ref = env.reference
    heading_err = np.linalg.norm(state[5] - ref[3])
    heading_err = abs((heading_err + np.pi) % (2 * np.pi) - np.pi)
    pos_err = ((state[:3] - ref[:3]) ** 2).sum()
    ctrl_effort = (np.array(action) ** 2).sum()
    too_far = (pos_err > env.max_distance ** 2)
    reward = - pos_err - 500*too_far - heading_err - 0.02*ctrl_effort
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
    reward = (5 + 10*close_enough -3*pos_err*(1 + num_steps/100) - 10*too_far -0.3*tilt_mag - 0.7*heading_err - 0.3*ctrl_effort - 0.3*rot_energy - 0.5*pendulum_energy)/10
    return reward

def reward_2(env, state, action, num_steps):
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
    reward = (5 + 20*close_enough -3*pos_err*(1 + num_steps/50) - 20*too_far -0.5*tilt_mag - 1*heading_err - 0.3*ctrl_effort - 0.3*rot_energy - 1*pendulum_energy)/10
    return reward
