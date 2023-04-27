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
    reward = 5 - pos_err - 0.1*heading_err
    return reward


def distance_energy_reward(env, state, action, num_steps):
    # penalize distance and action magnitude
    ref = env.reference
    heading_err = np.linalg.norm(state[5] - ref[3])
    heading_err = abs((heading_err + np.pi) % (2 * np.pi) - np.pi)
    pos_err = ((state[:3] - ref[:3]) ** 2).sum()
    ctrl_effort = (np.array(action) ** 2).sum()
    reward = 3.5 - pos_err - 0.1*heading_err - 0.2*ctrl_effort
    return reward


def distance_energy_reward_pendulum_angle(env, state, action, num_steps):
    # penalize distance and action magnitude
    ref = env.reference
    heading_err = np.linalg.norm(state[5] - ref[3])
    heading_err = ((heading_err + np.pi) % (2 * np.pi) - np.pi)**2
    pos_err = ((state[:3] - ref[:3]) ** 2).sum()
    ctrl_effort = (np.array(action) ** 2).sum()
    pendulum_dev = (np.array(state[12:14]) ** 2).sum()
    reward = 3.5 - pos_err - 0.2*heading_err - 0.2*ctrl_effort - 0.2*pendulum_dev
    return reward


def distance_energy_reward_pendulum_angle2(env, state, action, num_steps):
    # penalize distance and action magnitude
    ref = env.reference
    heading_err = np.linalg.norm(state[5] - ref[3])
    heading_err = ((heading_err + np.pi) % (2 * np.pi) - np.pi)**2
    pos_err = ((state[:3] - ref[:3]) ** 2).sum()
    ctrl_effort = (np.array(action) ** 2).sum()
    pendulum_dev = (np.array(state[12:14]) ** 2).sum()
    ang_vel = (np.array(state[9:12]) ** 2).sum()
    reward = 3.5 - pos_err - 0.5*heading_err - 0.4*ctrl_effort - 0.2*pendulum_dev - 0.1*ang_vel
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
    pendulum_energy = (np.array(state[14:16])**2).sum()
    too_far = pos_err > env.max_distance**2 - 3
    reward = (7 + 20*close_enough -3*pos_err*(1 + num_steps/150) - 10*too_far -0.3*tilt_mag - 0.7*heading_err - 0.3*ctrl_effort - 0.3*rot_energy - 0.5*pendulum_energy)/10
    return reward


# def reward_pendulum_dist(env, state, action, num_steps):
#     ref = env.reference
#     heading_err = np.linalg.norm(state[5] - ref[3])
#     heading_err = ((heading_err + np.pi) % (2 * np.pi) - np.pi)**2
#     tilt_mag = (np.array(state[3:5]) ** 2).sum()
#
#     pendulum_rpy = state[12:14]
#     drone_R = mujoco_quat2DCM(mujoco_rpy2quat(state[3:6]))  # drone rotation in world frame
#     pendulum_R = mujoco_quat2DCM(mujoco_rpy2quat(pendulum_rpy))  # pendulum rotation in drone frame
#     params = state[23:]
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
    pendulum_rp = state[12:14]
    drone_R = mujoco_quat2DCM(mujoco_rpy2quat(state[3:6]))  # drone rotation in world frame
    pendulum_R = mujoco_quat2DCM(mujoco_rpy2quat(np.append(pendulum_rp, 0)))  # pendulum rotation in drone frame
    params = state[23:]
    pendulum_end = np.array([[0], [0], [-params[5]]])  # pendulum end in pendulum frame
    R = drone_R @ pendulum_R
    pendulum_pos = state[:3] + (R @ pendulum_end)[:, 0]
    pos_err = ((pendulum_pos - ref[:3]) ** 2).sum()
    reward = - pos_err
    return reward


def reward_pendulumDistHeading(env, state, action, num_steps):
    ref = env.reference
    heading_err = np.linalg.norm(state[5] - ref[3])
    heading_err = abs((heading_err + np.pi) % (2 * np.pi) - np.pi)
    pendulum_rp = state[12:14]
    drone_R = mujoco_quat2DCM(mujoco_rpy2quat(state[3:6]))  # drone rotation in world frame
    pendulum_R = mujoco_quat2DCM(mujoco_rpy2quat(np.append(pendulum_rp, 0)))  # pendulum rotation in drone frame
    params = state[23:]
    pendulum_end = np.array([[0], [0], [-params[4]]])  # pendulum end in pendulum frame
    R = drone_R @ pendulum_R
    pendulum_pos = state[:3] + (R @ pendulum_end)[:, 0]
    pos_err = ((pendulum_pos - ref[:3]) ** 2).sum()
    reward = 3 - pos_err - 0.1*heading_err
    return reward


def reward_2(env, state, action, num_steps):
    ref = env.reference
    heading_err = np.linalg.norm(state[5] - ref[3])
    heading_err = abs((heading_err + np.pi) % (2 * np.pi) - np.pi)
    pendulum_rp = state[12:14]
    drone_R = mujoco_quat2DCM(mujoco_rpy2quat(state[3:6]))  # drone rotation in world frame
    pendulum_R = mujoco_quat2DCM(mujoco_rpy2quat(np.append(pendulum_rp, 0)))  # pendulum rotation in drone frame
    params = state[23:]
    pendulum_end = np.array([[0], [0], [-params[4]]])  # pendulum end in pendulum frame
    R = drone_R @ pendulum_R
    pendulum_pos = state[:3] + (R @ pendulum_end)[:, 0]
    pos_err = ((pendulum_pos - ref[:3]) ** 2).sum()
    ctrl_effort = (np.array(action) ** 2).sum()
    reward = 4 - pos_err - 0.001*num_steps*pos_err - 0.1*heading_err - 0.05*ctrl_effort
    return reward


def reward_2_penergy(env, state, action, num_steps):
    ref = env.reference
    heading_err = np.linalg.norm(state[5] - ref[3])
    heading_err = abs((heading_err + np.pi) % (2 * np.pi) - np.pi)
    pendulum_rp = state[12:14]
    pendulum_omega = np.append(state[14:16], 0)
    drone_R = mujoco_quat2DCM(mujoco_rpy2quat(state[3:6]))  # drone rotation in world frame
    pendulum_R = mujoco_quat2DCM(mujoco_rpy2quat(np.append(pendulum_rp, 0)))  # pendulum rotation in drone frame
    params = state[23:]
    pendulum_end = np.array([[0], [0], [-params[4]]])  # pendulum end in pendulum frame
    pendulum_v_local = np.cross(pendulum_omega, (pendulum_R @ pendulum_end).T[0])  # omega in drone frame cross pendulum pos
    pendulum_v_global = state[6:9] + (drone_R @ pendulum_v_local[None].T).T[0]  # rotate the pendulum velocity to global frame and add drone velocity
    pendulum_energy = (pendulum_v_global**2).sum()
    R = drone_R @ pendulum_R
    pendulum_pos = state[:3] + (R @ pendulum_end)[:, 0]
    pos_err = ((pendulum_pos - ref[:3]) ** 2).sum()
    ctrl_effort = (np.array(action) ** 2).sum()
    reward = 4 - pos_err - 0.2*heading_err - 0.006*num_steps*(pos_err + 0.2*heading_err) - 0.05*ctrl_effort - 0.1*pendulum_energy
    return reward


def reward_3(env, state, action, num_steps):
    ref = env.reference
    heading_err = np.linalg.norm(state[5] - ref[3])
    heading_err = abs((heading_err + np.pi) % (2 * np.pi) - np.pi)
    pendulum_rp = state[12:14]
    pendulum_omega = np.append(state[14:16], 0)
    drone_R = mujoco_quat2DCM(mujoco_rpy2quat(state[3:6]))  # drone rotation in world frame
    pendulum_R = mujoco_quat2DCM(mujoco_rpy2quat(np.append(pendulum_rp, 0)))  # pendulum rotation in drone frame
    params = state[23:]
    pendulum_end = np.array([[0], [0], [-params[4]]])  # pendulum end in pendulum frame
    pendulum_v_local = np.cross(pendulum_omega, (pendulum_R @ pendulum_end).T[0])  # omega in drone frame cross pendulum pos
    pendulum_v_global = state[6:9] + (drone_R @ pendulum_v_local[None].T).T[0]  # rotate the pendulum velocity to global frame and add drone velocity
    pendulum_energy = (pendulum_v_global**2).sum()
    R = drone_R @ pendulum_R
    pos_err = ((state[:3] - ref[:3]) ** 2).sum()
    ctrl_effort = (np.minimum(np.array(action) - 0.5, 0) ** 2).sum()
    reward = 4 - pos_err - 0.2*heading_err - 0.006*num_steps*(pos_err + 0.2*heading_err + 0.01*pendulum_energy) - 0.1*ctrl_effort - 0.1*pendulum_energy
    return reward