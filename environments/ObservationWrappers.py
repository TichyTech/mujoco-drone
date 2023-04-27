from .BaseDroneEnv import BaseDroneEnv
from gymnasium.spaces import Box
import numpy as np
from .transformation import mujoco_quat2DCM, mujoco_rpy2quat


class GlobalFrameRPYEnv(BaseDroneEnv):

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_states = 16
        self.num_params = 0
        num_obs = self.num_states + self.num_params
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float64)

    def _get_obs(self):
        drone_states = super()._get_obs()
        out_obs = []
        for state in drone_states:
            xyz = state[:3]
            rpy = state[3:6]
            yaw = rpy[2]
            vel = state[6:9]
            ang_vel = state[9:12]
            pendulum_rp = state[12:14]
            pendulum_ang_vel = state[14:16]
            # acc = state[16:19]
            # ref = state[19:23]
            # params = state[23:]
            heading_diff = np.array((self.reference[3] - yaw + np.pi) % (2 * np.pi) - np.pi)[None]  # yaw signed difference
            glob_ref_err = np.array(self.reference[:3] - xyz)
            obs_i = np.concatenate([glob_ref_err, rpy[:2], heading_diff, vel, ang_vel, pendulum_rp, pendulum_ang_vel])
            out_obs.append(obs_i)
        return out_obs


class LocalFramePRYEnv(BaseDroneEnv):
    """Replaces global state observation with local drone frame relative states.
    This includes reference, velocity and angular velocity in local drone frame as well as roll and pitch angles and
    signed yaw angle difference with respect to the reference state."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_states = 16
        self.num_params = 0
        num_obs = self.num_states + self.num_params
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float64)

    def _get_obs(self):
        drone_states = super()._get_obs()
        out_obs = []
        for state in drone_states:
            xyz = state[:3]
            rpy = state[3:6]
            yaw = rpy[2]
            vel = state[6:9]
            ang_vel = state[9:12]
            pendulum_rp = state[12:14]
            pendulum_ang_vel = state[14:16]
            # acc = state[16:19]
            # ref = state[19:23]
            # params = state[23:]
            heading_diff = np.array((self.reference[3] - yaw + np.pi) % (2 * np.pi) - np.pi)[None]  # yaw signed difference
            glob_ref_err = np.array(self.reference[:3] - xyz)[None].T
            R = mujoco_quat2DCM(mujoco_rpy2quat(rpy)).T
            loc_ref_err = R @ glob_ref_err  # reference direction in local frame
            glob_vel = np.array(vel)[None].T
            loc_vel = R @ glob_vel  # velocity in local frame
            glob_ang_vel = np.array(ang_vel)[None].T
            loc_ang_vel = R @ glob_ang_vel
            obs_i = np.concatenate([loc_ref_err.squeeze(), rpy[:2][::-1], heading_diff, loc_vel.squeeze(), loc_ang_vel.squeeze(), pendulum_rp[::-1], pendulum_ang_vel])
            out_obs.append(obs_i)
        return out_obs


class LocalFramePRYaccEnv(BaseDroneEnv):
    """Replaces global state observation with local drone frame relative states.
    This includes reference, velocity and angular velocity in local drone frame as well as roll and pitch angles and
    signed yaw angle difference with respect to the reference state. Also includes accelerations of the drone in local
    frame."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_states = 19
        self.num_params = 0
        num_obs = self.num_states + self.num_params
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float64)

    def _get_obs(self):
        drone_states = super()._get_obs()
        out_obs = []
        for state in drone_states:
            xyz = state[:3]
            rpy = state[3:6]
            yaw = rpy[2]
            vel = state[6:9]
            ang_vel = state[9:12]
            pendulum_rp = state[12:14]
            pendulum_ang_vel = state[14:16]
            acc = state[16:19]
            # ref = state[19:23]
            # params = state[23:]
            heading_diff = np.array((self.reference[3] - yaw + np.pi) % (2 * np.pi) - np.pi)[None]  # yaw signed difference
            glob_ref_err = np.array(self.reference[:3] - xyz)[None].T
            R = mujoco_quat2DCM(mujoco_rpy2quat(rpy)).T
            loc_ref_err = R @ glob_ref_err  # reference direction in local frame
            glob_vel = np.array(vel)[None].T
            loc_vel = R @ glob_vel  # velocity in local frame
            glob_ang_vel = np.array(ang_vel)[None].T
            loc_ang_vel = R @ glob_ang_vel
            obs_i = np.concatenate([loc_ref_err.squeeze(), rpy[:2][::-1], heading_diff, loc_vel.squeeze(),
                                    loc_ang_vel.squeeze(), pendulum_rp[::-1], pendulum_ang_vel, acc])
            out_obs.append(obs_i)
        return out_obs


class LocalFramePRYParamsEnv(BaseDroneEnv):
    """Replaces global state observation with local drone frame relative states.
    This includes reference, velocity and angular velocity in local drone frame as well as roll and pitch angles and
    signed yaw angle difference with respect to the reference state. Also, per drone model parameters are appended to the
    observation vector."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_states = 16
        self.num_params = 6
        num_obs = self.num_states + self.num_params
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float64)

    def _get_obs(self):
        drone_states = super()._get_obs()
        out_obs = []
        for state in drone_states:
            xyz = state[:3]
            rpy = state[3:6]
            yaw = rpy[2]
            vel = state[6:9]
            ang_vel = state[9:12]
            pendulum_rp = state[12:14]
            pendulum_ang_vel = state[14:16]
            # acc = state[16:19]
            # ref = state[19:23]
            params = state[23:]
            heading_diff = np.array((self.reference[3] - yaw + np.pi) % (2 * np.pi) - np.pi)[None]  # yaw signed difference
            glob_ref_err = np.array(self.reference[:3] - xyz)[None].T
            R = mujoco_quat2DCM(mujoco_rpy2quat(rpy)).T
            loc_ref_err = R @ glob_ref_err  # reference direction in local frame
            glob_vel = np.array(vel)[None].T
            loc_vel = R @ glob_vel  # velocity in local frame
            glob_ang_vel = np.array(ang_vel)[None].T
            loc_ang_vel = R @ glob_ang_vel
            obs_i = np.concatenate([loc_ref_err.squeeze(), rpy[:2][::-1], heading_diff, loc_vel.squeeze(), loc_ang_vel.squeeze(), pendulum_rp[::-1], pendulum_ang_vel, params])
            out_obs.append(obs_i)
        return out_obs


class LocalFramePRYaccParamsEnv(BaseDroneEnv):

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_states = 19
        self.num_params = 6
        num_obs = self.num_states + self.num_params
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float64)

    def _get_obs(self):
        drone_states = super()._get_obs()
        out_obs = []
        for state in drone_states:
            xyz = state[:3]
            rpy = state[3:6]
            yaw = rpy[2]
            vel = state[6:9]
            ang_vel = state[9:12]
            pendulum_rp = state[12:14]
            pendulum_ang_vel = state[14:16]
            acc = state[16:19]
            # ref = state[19:23]
            params = state[23:]
            heading_diff = np.array((self.reference[3] - yaw + np.pi) % (2 * np.pi) - np.pi)[None]  # yaw signed difference
            glob_ref_err = np.array(self.reference[:3] - xyz)[None].T
            R = mujoco_quat2DCM(mujoco_rpy2quat(rpy)).T
            loc_ref_err = R @ glob_ref_err  # reference direction in local frame
            glob_vel = np.array(vel)[None].T
            loc_vel = R @ glob_vel  # velocity in local frame
            glob_ang_vel = np.array(ang_vel)[None].T
            loc_ang_vel = R @ glob_ang_vel
            obs_i = np.concatenate([loc_ref_err.squeeze(), rpy[:2][::-1], heading_diff, loc_vel.squeeze(), loc_ang_vel.squeeze(), pendulum_rp[::-1], acc, pendulum_ang_vel, params])
            out_obs.append(obs_i)
        return out_obs


class LocalFrameRPYParamsEnv(BaseDroneEnv):
    """Replaces global state observation with local drone frame relative states.
    This includes reference, velocity and angular velocity in local drone frame as well as roll and pitch angles and
    signed yaw angle difference with respect to the reference state. Also, per drone model parameters are appended to the
    observation vector."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_states = 16
        self.num_params = 6
        num_obs = self.num_states + self.num_params
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float64)

    def _get_obs(self):
        drone_states = super()._get_obs()
        out_obs = []
        for state in drone_states:
            xyz = state[:3]
            rpy = state[3:6]
            yaw = rpy[2]
            vel = state[6:9]
            ang_vel = state[9:12]
            pendulum_rp = state[12:14]
            pendulum_ang_vel = state[14:16]
            # acc = state[16:19]
            # ref = state[19:23]
            params = state[23:]
            heading_diff = np.array((self.reference[3] - yaw + np.pi) % (2 * np.pi) - np.pi)[None]  # yaw signed difference
            glob_ref_err = np.array(self.reference[:3] - xyz)[None].T
            R = mujoco_quat2DCM(mujoco_rpy2quat(rpy)).T
            loc_ref_err = R @ glob_ref_err  # reference direction in local frame
            glob_vel = np.array(vel)[None].T
            loc_vel = R @ glob_vel  # velocity in local frame
            glob_ang_vel = np.array(ang_vel)[None].T
            loc_ang_vel = R @ glob_ang_vel
            obs_i = np.concatenate([loc_ref_err.squeeze(), rpy[:2], heading_diff, loc_vel.squeeze(), loc_ang_vel.squeeze(), pendulum_rp, pendulum_ang_vel, params])
            out_obs.append(obs_i)
        return out_obs


class LocalFrameRPYFakeParamsEnv(BaseDroneEnv):
    """Replaces global state observation with local drone frame relative states.
    This includes reference, velocity and angular velocity in local drone frame as well as roll and pitch angles and
    signed yaw angle difference with respect to the reference state. Also, per drone model parameters are appended to the
    observation vector."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_states = 16
        self.num_params = 6
        num_obs = self.num_states + self.num_params
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float64)

    def _get_obs(self):
        drone_states = super()._get_obs()
        out_obs = []
        for state in drone_states:
            xyz = state[:3]
            rpy = state[3:6]
            yaw = rpy[2]
            vel = state[6:9]
            ang_vel = state[9:12]
            pendulum_rp = state[12:14]
            pendulum_ang_vel = state[14:16]
            # acc = state[16:19]
            # ref = state[19:23]
            params = [1, 0.17, 7, 0.01, 1.2, 0.3]
            heading_diff = np.array((self.reference[3] - yaw + np.pi) % (2 * np.pi) - np.pi)[None]  # yaw signed difference
            glob_ref_err = np.array(self.reference[:3] - xyz)[None].T
            R = mujoco_quat2DCM(mujoco_rpy2quat(rpy)).T
            loc_ref_err = R @ glob_ref_err  # reference direction in local frame
            glob_vel = np.array(vel)[None].T
            loc_vel = R @ glob_vel  # velocity in local frame
            glob_ang_vel = np.array(ang_vel)[None].T
            loc_ang_vel = R @ glob_ang_vel
            obs_i = np.concatenate([loc_ref_err.squeeze(), rpy[:2], heading_diff, loc_vel.squeeze(), loc_ang_vel.squeeze(), pendulum_rp, pendulum_ang_vel, params])
            out_obs.append(obs_i)
        return out_obs


class LocalFrameRPYEnv(BaseDroneEnv):
    """Replaces global state observation with local drone frame relative states.
    This includes reference, velocity and angular velocity in local drone frame as well as roll and pitch angles and
    signed yaw angle difference with respect to the reference state."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_states = 16
        self.num_params = 0
        num_obs = self.num_states + self.num_params
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float64)

    def _get_obs(self):
        drone_states = super()._get_obs()
        out_obs = []
        for state in drone_states:
            xyz = state[:3]
            rpy = state[3:6]
            yaw = rpy[2]
            vel = state[6:9]
            ang_vel = state[9:12]
            pendulum_rp = state[12:14]
            pendulum_ang_vel = state[14:16]
            # acc = state[16:19]
            # ref = state[19:23]
            # params = state[23:]
            heading_diff = np.array((self.reference[3] - yaw + np.pi) % (2 * np.pi) - np.pi)[None]  # yaw signed difference
            glob_ref_err = np.array(self.reference[:3] - xyz)[None].T
            R = mujoco_quat2DCM(mujoco_rpy2quat(rpy)).T
            loc_ref_err = R @ glob_ref_err  # reference direction in local frame
            glob_vel = np.array(vel)[None].T
            loc_vel = R @ glob_vel  # velocity in local frame
            glob_ang_vel = np.array(ang_vel)[None].T
            loc_ang_vel = R @ glob_ang_vel
            obs_i = np.concatenate([loc_ref_err.squeeze(), rpy[:2], heading_diff, loc_vel.squeeze(), loc_ang_vel.squeeze(), pendulum_rp, pendulum_ang_vel])
            out_obs.append(obs_i)
        return out_obs


class LocalFramePRYaccNoPendEnv(BaseDroneEnv):
    """Provides drone state in local frame in the x-y-z order as well as acceleration."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_states = 15
        self.num_params = 0
        num_obs = self.num_states + self.num_params
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float64)

    def _get_obs(self):
        drone_states = super()._get_obs()
        out_obs = []
        for state in drone_states:
            xyz = state[:3]
            rpy = state[3:6]
            yaw = rpy[2]
            vel = state[6:9]
            ang_vel = state[9:12]
            acc = state[16:19]
            # ref = state[19:23]
            # params = state[23:]
            heading_diff = np.array((self.reference[3] - yaw + np.pi) % (2 * np.pi) - np.pi)[None]  # yaw signed difference
            glob_ref_err = np.array(self.reference[:3] - xyz)[None].T
            R = mujoco_quat2DCM(mujoco_rpy2quat(rpy)).T
            loc_ref_err = R @ glob_ref_err  # reference direction in local frame
            glob_vel = np.array(vel)[None].T
            loc_vel = R @ glob_vel  # velocity in local frame
            glob_ang_vel = np.array(ang_vel)[None].T
            loc_ang_vel = R @ glob_ang_vel
            obs_i = np.concatenate([loc_ref_err.squeeze(), rpy[:2][::-1], heading_diff, loc_vel.squeeze(), loc_ang_vel.squeeze(), acc])
            out_obs.append(obs_i)
        return out_obs


class LocalFramePRYaccParamsNoPendEnv(BaseDroneEnv):
    """Provides drone state in local frame in the x-y-z order as well as acceleration and drone model parameters."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_states = 15
        self.num_params = 6
        num_obs = self.num_states + self.num_params
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float64)

    def _get_obs(self):
        drone_states = super()._get_obs()
        out_obs = []
        for state in drone_states:
            xyz = state[:3]
            rpy = state[3:6]
            yaw = rpy[2]
            vel = state[6:9]
            ang_vel = state[9:12]
            acc = state[16:19]
            # ref = state[19:23]
            params = state[23:]
            heading_diff = np.array((self.reference[3] - yaw + np.pi) % (2 * np.pi) - np.pi)[None]  # yaw signed difference
            glob_ref_err = np.array(self.reference[:3] - xyz)[None].T
            R = mujoco_quat2DCM(mujoco_rpy2quat(rpy)).T
            loc_ref_err = R @ glob_ref_err  # reference direction in local frame
            glob_vel = np.array(vel)[None].T
            loc_vel = R @ glob_vel  # velocity in local frame
            glob_ang_vel = np.array(ang_vel)[None].T
            loc_ang_vel = R @ glob_ang_vel
            obs_i = np.concatenate([loc_ref_err.squeeze(), rpy[:2][::-1], heading_diff, loc_vel.squeeze(), loc_ang_vel.squeeze(), acc, params])
            out_obs.append(obs_i)
        return out_obs


class LocalFrameRmParamsEnv(BaseDroneEnv):
    """Replaces global state observation with local drone frame relative states.
    This includes reference, velocity and angular velocity in local drone frame as well as a rotation matrix relative to
    the reference state yaw. Also, per drone model parameters are appended to the observation vector."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_states = 22
        self.num_params = 6
        num_obs = self.num_states + self.num_params
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float64)

    def _get_obs(self):
        drone_states = super()._get_obs()
        out_obs = []
        for state in drone_states:
            xyz = state[:3]
            rpy = state[3:6]
            yaw = rpy[2]
            vel = state[6:9]
            ang_vel = state[9:12]
            pendulum_rp = state[12:14]
            pendulum_ang_vel = state[14:16]
            # acc = state[16:19]
            # ref = state[19:23]
            params = state[23:]
            heading_diff = np.array((self.reference[3] - yaw + np.pi) % (2 * np.pi) - np.pi)[None]  # yaw signed difference
            glob_ref_err = np.array(self.reference[:3] - xyz)[None].T
            R = mujoco_quat2DCM(mujoco_rpy2quat(rpy)).T
            Rm = mujoco_quat2DCM(mujoco_rpy2quat(np.append(rpy[:2], -heading_diff))).T
            loc_ref_err = R @ glob_ref_err  # reference direction in local frame
            glob_vel = np.array(vel)[None].T
            loc_vel = R @ glob_vel  # velocity in local frame
            glob_ang_vel = np.array(ang_vel)[None].T
            loc_ang_vel = R @ glob_ang_vel
            obs_i = np.concatenate([loc_ref_err.squeeze(), Rm.flatten(), loc_vel.squeeze(), loc_ang_vel.squeeze(), pendulum_rp, pendulum_ang_vel, params])
            out_obs.append(obs_i)
        return out_obs


class LocalFrameZvecEnv(BaseDroneEnv):
    """Replaces global state observation with local drone frame relative states.
    This includes reference, velocity and angular velocity in local drone frame as well as the z vector in local frame
    and yaw angle signed difference with respect to the reference state."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_states = 17
        self.num_params = 0
        num_obs = self.num_states + self.num_params
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float64)

    def _get_obs(self):
        drone_states = super()._get_obs()
        out_obs = []
        for state in drone_states:
            xyz = state[:3]
            rpy = state[3:6]
            yaw = rpy[2]
            vel = state[6:9]
            ang_vel = state[9:12]
            pendulum_rp = state[12:14]
            pendulum_ang_vel = state[14:16]
            # acc = state[16:19]
            # ref = state[19:23]
            params = state[23:]
            heading_diff = np.array((self.reference[3] - yaw + np.pi) % (2 * np.pi) - np.pi)[None]  # yaw signed difference
            DCM = mujoco_quat2DCM(mujoco_rpy2quat(np.append(rpy[:2], 0)))  # compute DCM from roll, pitch, 0
            z_vec = DCM[:, 2]  # this is the direction of the z axis of the drone in the global frame rotated by yaw
            glob_ref_err = np.array(self.reference[:3] - xyz)[None].T
            R = mujoco_quat2DCM(mujoco_rpy2quat(rpy)).T
            loc_ref_err = R @ glob_ref_err  # reference direction in local frame
            glob_vel = np.array(vel)[None].T
            loc_vel = R @ glob_vel  # velocity in local frame
            glob_ang_vel = np.array(ang_vel)[None].T
            loc_ang_vel = R @ glob_ang_vel
            obs_i = np.concatenate([loc_ref_err.squeeze(), z_vec, heading_diff, loc_vel.squeeze(), loc_ang_vel.squeeze(), pendulum_rp, pendulum_ang_vel])
            out_obs.append(obs_i)
        return out_obs