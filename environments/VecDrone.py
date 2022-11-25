import numpy as np
from gym import utils
from .mujoco_env_custom import extendedEnv
from .env_gen import make_arena, mjcf_to_mjmodel
from gym.spaces import Dict, Box
from scipy.spatial.transform import Rotation as R
from ray.rllib.env.vector_env import VectorEnv


def mujoco_quat2DCM(quat):
    return R.from_quat(np.append(quat[1:], quat[0])).as_matrix()


def mujoco_quat2rpy(quat):
    return R.from_quat(np.append(quat[1:], quat[0])).as_euler('zyx')[::-1]


class VecDroneEnv(extendedEnv, VectorEnv, utils.EzPickle):
    # drone simulation with PID attitude controller
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
        self.num_drones = config['num_drones']
        self.pendulum = config['pendulum']
        self.reference = config['reference']

        self.action_space = Box(low=0, high=1, shape=(4, ), dtype=np.float64)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(12, ), dtype=np.float64)

        model = mjcf_to_mjmodel(make_arena(self.num_drones, self.pendulum, self.reference[:3]))
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
        if self.pendulum:
            self.mass = 1.4
        else:
            self.mass = 1.1

        if config['start_pos'] is None:
            self.start_pos = self.reference[:3]
        else:
            self.start_pos = config['start_pos']

        self.max_distance = 2

        self.motor_force = 6
        self.errs = []
        self.ctrls = []

        self.first_obs = True
        self.prev_err = None

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
        states, cols = self._get_obs()

        rewards = []
        dones = []
        obs = []
        infos = []
        for i in range(self.num_drones):
            state = states[i]
            # heading_err = np.linalg.norm(state[5] - self.reference[3])
            # heading_err = abs((heading_err + np.pi) % (2*np.pi) - np.pi)
            reward = 0.5 - self.num_steps[i]*(np.linalg.norm(state[:3] - self.reference[:3]))
            rewards.append(reward)
            self.terminated[i] = (np.linalg.norm(state[:3] - self.reference[:3]) > self.max_distance) or cols[i] or self.num_steps[i] >= 400
            dones.append(self.terminated[i])
            obs.append(np.concatenate((self.reference[:3] - state[:3], state[3:])))
            infos.append({})

        self.render()
        return (obs, rewards, dones, infos)

    def reset_model(self):  # init the drone to start position and some perturbation
        start_pos = self.start_pos
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.04, high=0.04
        )

        for i in range(self.num_drones):
            qpos[7*i:7*i + 3] = start_pos + self.np_random.uniform(size=3, low=-0.2, high=0.2)

        self.num_steps = np.zeros((self.num_drones, ), dtype=np.long)
        self.set_state(qpos, qvel)
        return self._get_obs()[0]

    def vector_reset(self):
        self.num_steps = np.zeros((self.num_drones, ), dtype=np.long)
        self.terminated = np.zeros((self.num_drones,), dtype=np.bool)
        ob = self.reset_model()
        return ob

    def reset_at(self, index):
        if index is None:
            index = 0
        start_pos = self.start_pos
        qpos = self.data.qpos[:]
        qvel = self.data.qvel[:]
        qpos[7*index: 7*index + 7] = self.init_qpos[7*index: 7*index + 7] + self.np_random.uniform(size=7, low=-0.1, high=0.1)
        qpos[7 * index: 7*index + 3] = start_pos + self.np_random.uniform(size=3, low=-0.1, high=0.1)
        qvel[6*index: 6*index + 6] = self.init_qvel[6*index: 6*index + 6] + self.np_random.uniform(size=6, low=-0.04, high=0.04)
        # qvel[6 * index: 6 * index + 3] = self.np_random.uniform(size=3, low=-0.3, high=0.3)
        self.set_state(qpos, qvel)
        self.num_steps[index] = 0
        return self._get_obs()[0][index]

    def _get_obs(self):
        states = []  # state info for every drone
        for i in range(self.num_drones):
            pos = self.data.qpos[7*i:7*i + 3]  # xyz positions
            angle = mujoco_quat2rpy(self.data.qpos[7*i + 3:7*i + 7])  # rpy angles
            vel = self.data.qvel[6*i:6*i + 3]  # xyz velocity
            ang_vel = self.data.qvel[6*i + 3:6*i + 6]  # rpy velocity (probably in different order)
            obs = np.concatenate((pos, angle, vel, ang_vel))
            states.append(obs)  # add state to state list

        cols = np.zeros((self.num_drones,), dtype=np.bool)  # collision info for every drone
        ncol = self.data.ncon  # number of collisions
        for i in range(ncol):  # check for collisions (hardcoded and not pretty)
            con = self.data.contact[i]
            if con.geom1 > 0:
                if not self.pendulum:
                    drone_id = (con.geom1 - 2) // 9
                else:
                    drone_id = (con.geom1 - 2) // 10
                cols[drone_id] = True
            if con.geom2 > 0:
                if not self.pendulum:
                    drone_id = (con.geom2 - 2) // 9
                else:
                    drone_id = (con.geom2 - 2) // 10
                cols[drone_id] = True
        return states, cols

    def viewer_setup(self):
        assert self.viewer is not None
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent