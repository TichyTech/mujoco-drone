import numpy as np
from gym import utils
from .mujoco_env_custom import extendedEnv
from .env_gen import make_arena, mjcf_to_mjmodel
from gym.spaces import Dict, Box
from scipy.spatial.transform import Rotation as R
from ray.rllib.env.vector_env import VectorEnv

base_config = {'reference': [0.5, 0.5, 3.5, 0.4],  # x,y,z,yaw
               'start_pos': [0, 0, 3, 0],  # x,y,z,yaw
               'max_pos_offset': 0.4,  # maximum position offset used for random sampling of starting position
               'angle_variance': [0.3, 0.3, 0.3],  # variance used for random angle sampling
               'vel_variance': [0.04, 0.04, 0.04],  # variance used for random velocity sampling
               'pendulum': True,  # whether to include a pendulum on a drone
               'render_mode': 'human'}


def mujoco_quat2DCM(quat):
    return R.from_quat(np.append(quat[1:], quat[0])).as_matrix()


def mujoco_quat2rpy(quat):
    return R.from_quat(np.append(quat[1:], quat[0])).as_euler('zyx')[::-1]


def mujoco_rpy2quat(rpy):
    quat = R.from_euler('zyx', rpy[::-1]).as_quat()
    return np.append(quat[3], quat[:3])


class VecDrone(extendedEnv, VectorEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(self, config, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        width, height = 640, 480
        self.num_drones = config.get('num_drones', 1)
        self.pendulum = config['pendulum']
        self.reference = config['reference']

        self.action_space = Box(low=0, high=1, shape=(4, ), dtype=np.float64)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(12, ), dtype=np.float64)

        model = mjcf_to_mjmodel(make_arena(self.num_drones, self.pendulum, self.reference))
        extendedEnv.__init__(
            self,
            model,
            4,
            render_mode=config['render_mode'],
            observation_space=observation_space,
            width=width,
            height=height,
            **kwargs
        )

        if config['start_pos'] is None:
            self.start_pos = self.reference  # if no start specified, start in the reference
        else:
            self.start_pos = config['start_pos']

        self.max_pos_offset = config['max_pos_offset']
        self.angle_variance = config['angle_variance']
        self.vel_variance = config['vel_variance']
        self.max_distance = 2

        self.num_steps = np.zeros((self.num_drones, ), dtype=np.long)
        self.terminated = np.zeros((self.num_drones,), dtype=np.bool)
        self._agent_ids = set(range(self.num_drones))
        self.dones = set()
        self.resetted = False

        VectorEnv.__init__(self, observation_space, self.action_space, self.num_drones)

    def vector_step(self, actions):
        ctrl = np.array(actions).ravel()
        self.do_simulation(ctrl, self.frame_skip)
        self.num_steps = self.num_steps + 1
        states = self._get_obs()

        rewards = []
        dones = []
        obs = []
        infos = []
        for i in range(self.num_drones):
            state = states[i]
            heading_err = np.linalg.norm(state[5] - self.reference[3])
            heading_err = abs((heading_err + np.pi) % (2*np.pi) - np.pi)
            pos_err = ((state[:3] - self.reference[:3])**2).sum()
            ctrl_effort = (np.array(actions[i])**2).sum()
            tilt_magnitude = (np.array(state[3:5])**2).sum()
            too_far = (pos_err > self.max_distance**2)
            self.terminated[i] = too_far or self.num_steps[i] >= 400
            reward = 0.1*self.num_steps[i]*(0.5 - pos_err) - 50*too_far - 0.01*heading_err - 0.01*ctrl_effort - 0.01*tilt_magnitude
            rewards.append(reward)
            dones.append(self.terminated[i])
            obs.append(np.concatenate((self.reference[:3] - state[:3], state[3:])))
            infos.append({})

        if self.render_mode == 'human':
            self.render()
        return (obs, rewards, dones, infos)

    def reset_model(self):  # init the drone to start position plus some perturbation
        start_pos = self.start_pos
        qpos = self.init_qpos
        qvel = self.init_qvel
        pos_idx_offset = 4 * self.pendulum
        vel_idx_offset = 3 * self.pendulum
        for i in range(self.num_drones):
            direction = self.np_random.normal(size=3)
            direction /= np.linalg.norm(direction)
            r = self.np_random.uniform(0, self.max_pos_offset)
            qpos[(7 + pos_idx_offset)*i:(7 + pos_idx_offset)*i + 3] = start_pos[:3] + r*direction
            rpy = self.np_random.normal(scale=self.angle_variance, size=3) + [0, 0, start_pos[3]]
            qpos[(7 + pos_idx_offset)*i + 3:(7 + pos_idx_offset)*i + 7] = mujoco_rpy2quat(rpy)
            qvel[(6 + vel_idx_offset)*i: (6 + vel_idx_offset)*i + 3] = self.np_random.normal(scale=self.vel_variance, size=3)

        self.num_steps = np.zeros((self.num_drones, ), dtype=np.long)
        self.terminated = np.zeros((self.num_drones,), dtype=np.bool)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def vector_reset(self):
        ob = self.reset_model()
        return ob

    def reset_at(self, index):
        if index is None:
            index = 0
        pos_idx_offset = 4 * self.pendulum
        vel_idx_offset = 3 * self.pendulum
        start_pos = self.start_pos
        qpos = self.data.qpos[:]
        qvel = self.data.qvel[:]
        qpos[(7 + pos_idx_offset) * index:(7 + pos_idx_offset) * index + 7] = self.init_qpos[(7 + pos_idx_offset) * index:(7 + pos_idx_offset) * index + 7]
        direction = self.np_random.normal(size=3)  # draw a random sample inside a sphere
        direction /= np.linalg.norm(direction)
        r = self.np_random.uniform(0, self.max_pos_offset)
        qpos[(7 + pos_idx_offset) * index:(7 + pos_idx_offset) * index + 3] = start_pos[:3] + r*direction
        qvel[(6 + vel_idx_offset)*index: (6 + vel_idx_offset)*index + 6] = self.init_qvel[(6 + vel_idx_offset)*index: (6 + vel_idx_offset)*index + 6]
        rpy = self.np_random.normal(scale=self.angle_variance, size=3) + [0, 0, start_pos[3]]
        qpos[(7 + pos_idx_offset)*index + 3:(7 + pos_idx_offset)*index + 7] = mujoco_rpy2quat(rpy)
        qvel[(6 + vel_idx_offset)*index: (6 + vel_idx_offset)*index + 3] = self.np_random.normal(scale=self.vel_variance, size=3)
        self.set_state(qpos, qvel)
        self.num_steps[index] = 0
        return self._get_obs()[index]

    def _get_obs(self):
        states = []  # state info for every drone
        pos_idx_offset = 4 * self.pendulum
        vel_idx_offset = 3 * self.pendulum
        for i in range(self.num_drones):
            pos = self.data.qpos[(7 + pos_idx_offset)*i:(7 + pos_idx_offset)*i + 3]  # xyz positions
            angle = mujoco_quat2rpy(self.data.qpos[(7 + pos_idx_offset)*i + 3:(7 + pos_idx_offset)*i + 7])  # rpy angles
            vel = self.data.qvel[(6 + vel_idx_offset)*i:(6 + vel_idx_offset)*i + 3]  # xyz velocity
            ang_vel = self.data.qvel[(6 + vel_idx_offset)*i + 3:(6 + vel_idx_offset)*i + 6]  # rpy velocity (probably in different order)
            obs = np.concatenate((pos, angle, vel, ang_vel))
            states.append(obs)  # add state to state list

        # cols = np.zeros((self.num_drones,), dtype=np.bool)  # collision info for every drone
        # ncol = self.data.ncon  # number of collisions
        # for i in range(ncol):  # check for collisions (hardcoded and not pretty)
        #     con = self.data.contact[i]
        #     if con.geom1 > 0:
        #         if not self.pendulum:
        #             drone_id = (con.geom1 - 2) // 9
        #         else:
        #             drone_id = (con.geom1 - 2) // 10
        #         cols[drone_id] = True
        #     if con.geom2 > 0:
        #         if not self.pendulum:
        #             drone_id = (con.geom2 - 2) // 9
        #         else:
        #             drone_id = (con.geom2 - 2) // 10
        #         cols[drone_id] = True
        return states

    def viewer_setup(self):
        assert self.viewer is not None
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent