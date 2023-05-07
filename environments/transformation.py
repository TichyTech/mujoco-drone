from scipy.spatial.transform import Rotation as R
import numpy as np


def mujoco_DCM2quat(DCM):
    """convert from a rotation matrix to mujoco quat representation"""
    quat = R.from_matrix(DCM).as_quat()
    return np.append(quat[3], quat[:3])


def mujoco_quat2DCM(quat):
    """convert from mujoco quaternion to rotation matrix"""
    return R.from_quat(np.append(quat[1:], quat[0])).as_matrix()


def mujoco_quat2rpy(quat):
    """convert from mujoco quaternion to roll pitch and yaw angles"""
    return R.from_quat(np.append(quat[1:], quat[0])).as_euler('ZYX')[::-1]


def mujoco_rpy2quat(rpy):
    """convert from roll pitch yaw angles to mujoco quaternion"""
    quat = R.from_euler('ZYX', rpy[::-1]).as_quat()
    return np.append(quat[3], quat[:3])


def mujoco_pendulumrp2quat(pendulum_rp):
    quat = R.from_euler('XY', pendulum_rp).as_quat()
    return np.append(quat[3], quat[:3])