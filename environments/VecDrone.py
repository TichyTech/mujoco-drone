import numpy as np
from gym import utils
from .mujoco_env_custom import extendedEnv
from .env_gen import make_arena, mjcf_to_mjmodel
from .joystick import PS4Controller
from gym.spaces import Dict, Box
from scipy.spatial.transform import Rotation as R
from ray.rllib.env.vector_env import VectorEnv


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


def distance_reward(env, state, action, num_steps):
    # penalize distance from reference
    ref = env.reference
    heading_err = np.linalg.norm(state[5] - ref[3])
    heading_err = abs((heading_err + np.pi) % (2 * np.pi) - np.pi)
    pos_err = np.linalg.norm(state[:3] - ref[:3])
    too_far = pos_err > env.max_distance
    reward = - pos_err - 200*too_far - heading_err
    return reward


base_config = {'reference': [0, 0, 3, 0],  # x,y,z,yaw
               'start_pos': [0, 0, 3, 0],  # x,y,z,yaw
               'max_distance': 2.5,
               'max_random_offset': 1.5,  # maximum position offset used for random sampling of starting position
               'angle_variance': [0.1, 0.1, 0.1],  # variance used for random angle sampling
               'vel_variance': [0.03, 0.03, 0.03],  # variance used for random velocity sampling
               'body_size_interval': [0.09, 0.14],  # drone main body size in meters
               'body_mass_interval': [0.95, 1.05],  # drone main body mass in kilograms
               'arm_mult_interval': [0.95, 1.05],  # drone arm length in meters
               'pendulum': False,  # whether to include a pendulum on a drone
               'pendulum_length_interval': [0.12, 0.18],  # pendulum length in meters
               'weight_mass_interval': [0.1, 0.3],  # weight of the pendulum mass in kilograms
               'reward_fcn': distance_reward,
               'max_steps': 400,
               'regen_env_at_steps': None,
               'render_mode': 'human',
               'window_title': 'mujoco',
               'controlled': False}


def mujoco_quat2DCM(quat):
    return R.from_quat(np.append(quat[1:], quat[0])).as_matrix()


def mujoco_quat2rpy(quat):
    return R.from_quat(np.append(quat[1:], quat[0])).as_euler('ZYX')[::-1]


def mujoco_rpy2quat(rpy):
    quat = R.from_euler('ZYX', rpy[::-1]).as_quat()
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
        self.width, self.height = 640, 480
        self.render_mode = config.get('render_mode', None)
        self.window_title = config.get('window_title', 'mujoco')
        self.action_space = Box(low=0, high=1, shape=(4,), dtype=np.float64)

        # get generate environment parameters
        self.reference = config.get('reference', [0, 0, 0, 0])
        self.num_drones = config.get('num_drones', 1)
        self.pendulum = config.get('pendulum', False)
        self.body_size_interval = np.array(config.get('body_size_interval', [0.04, 0.04]))
        self.body_mass_interval = np.array(config.get('body_mass_interval', [0.8, 1]))
        self.arm_mult_interval = np.array(config.get('arm_mult_interval', [0.015, 0.015]))
        self.pendulum_length_interval = np.array(config.get('pendulum_length_interval', [0.0125, 0.0125]))
        self.weight_mass_interval = np.array(config.get('weight_mass_interval', [0.2, 0.2]))

        # generate randomized parameters for each drone and save them into a list
        self.drone_params = self.generate_drone_params()
        self.num_states = 15
        self.num_params = len(self.drone_params[0])
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_states + self.num_params,), dtype=np.float64)
        model = mjcf_to_mjmodel(make_arena(self.drone_params, self.reference))  # create a mujoco model
        extendedEnv.__init__(
            self,
            model,
            4,
            render_mode=self.render_mode,
            observation_space=self.observation_space,
            width=self.width,
            height=self.height
        )

        self.controlled = config.get('controlled', False)  # is this environment using controlled reference?
        if self.controlled:
            print('Initializing controller')
            self.joystick = PS4Controller()  # also works with PS5 controller
            if not self.joystick.controller:  # if no controller found, disable control
                self.controlled = False
                print('Disabling reference control')

        # setup training parameters
        self.regen_env_at_steps = config.get('regen_env_at_steps', None)
        self.start_pos = config.get('start_pos', self.reference)
        self.max_distance = config.get('max_distance', 1)
        self.reward_fcn = config.get('reward_fcn', distance_reward)
        self.max_steps = config.get('max_steps', 400)
        self.max_pos_offset = config.get('max_random_offset', 0)
        self.angle_variance = np.array(config.get('angle_variance', [0, 0, 0]))
        self.vel_variance = np.array(config.get('vel_variance', [0, 0, 0]))

        # setup
        self.total_steps = 0
        self.num_steps = np.zeros((self.num_drones, ), dtype=np.long)
        self.terminated = np.zeros((self.num_drones,), dtype=np.bool)

        # init VectorEnv for rllib and reset the model
        VectorEnv.__init__(self, self.observation_space, self.action_space, self.num_drones)
        print('Environment ready')

    def control_reference(self):
        self.joystick.poll_events()  # update joystick values
        x = self.joystick.axis_data.get(0, 0)
        y = -self.joystick.axis_data.get(1, 0)
        z = -self.joystick.axis_data.get(3, 0)
        yaw = -self.joystick.axis_data.get(2, 0)

        pert = np.array([x, y, z, yaw])  # perturbation to reference
        xy_active = np.linalg.norm(pert[:2]) > 0.2  # joystick-vise deadzone
        zyaw_active = np.linalg.norm(pert[2:]) > 0.2
        mag = np.maximum(np.abs(pert) - 0.1, 0)  # individual value deadzones
        sg = np.sign(pert)
        pert = 0.1 * mag * sg * [xy_active, xy_active, zyaw_active, zyaw_active]

        self.move_reference_by(pert)

    def move_reference_by(self, offset):
        self.move_reference_to(self.reference + offset)

    def move_reference_to(self, target):
        self.reference = target
        self.reference[3] = (self.reference[3] + np.pi) % (2*np.pi) - np.pi
        self.reference = np.clip(self.reference, a_min=[-5, -5, 0, -np.pi], a_max=[5, 5, 6, np.pi])  # update reference
        self.data.mocap_pos[:3] = self.reference[:3]  # update reference visualization
        self.data.mocap_quat[:] = mujoco_rpy2quat([0, 0, self.reference[3]])

    def generate_drone_params(self):
        # generate a list of per-drone model parameters
        drone_params = []
        # load parameter intervals
        l_bs, h_bs = self.body_size_interval
        l_bm, h_bm = self.body_mass_interval
        l_am, h_am = self.arm_mult_interval
        l_pl, h_pl = self.pendulum_length_interval
        l_wm, h_wm = self.weight_mass_interval
        # generate random values uniformly from the intervals
        body_sizes = self.np_random.uniform(l_bs, h_bs, self.num_drones)
        body_masses = self.np_random.uniform(l_bm, h_bm, self.num_drones)
        arm_mults = self.np_random.uniform(l_am, h_am, self.num_drones)
        pendulum_lens = self.np_random.uniform(l_pl, h_pl, self.num_drones)
        weight_masses = self.np_random.uniform(l_wm, h_wm, self.num_drones)
        # save parameters into a list of dictionaries
        for i in range(self.num_drones):
            params = {'body_size': body_sizes[i],
                      'body_mass': body_masses[i],
                      'arm_mult': arm_mults[i],
                      'pendulum': 1*self.pendulum,
                      'pendulum_len': self.pendulum*pendulum_lens[i],
                      'weight_mass': self.pendulum*weight_masses[i]}
            drone_params.append(params)
        return drone_params

    def vector_step(self, actions):
        if self.controlled:
            self.control_reference()  # move reference accordingly

        ctrl = np.array(actions).ravel()
        self.do_simulation(ctrl, self.frame_skip)
        self.num_steps = self.num_steps + 1
        self.total_steps += 1
        states = self._get_obs()

        rewards, dones, obs, infos = [], [], [], []
        for i in range(self.num_drones):
            state = states[i]
            pos_err = np.linalg.norm(state[:3] - self.reference[:3])
            self.terminated[i] = pos_err > self.max_distance or self.num_steps[i] >= self.max_steps
            reward = self.reward_fcn(self, state, actions[i], self.num_steps[i])
            rewards.append(reward)
            dones.append(self.terminated[i])
            heading_diff = np.array((self.reference[3] - state[5] + np.pi) % (2 * np.pi) - np.pi)[None]
            # if self.controlled:
                # print(state[5], self.reference[3], heading_diff, state[3:5])
            ref_err = np.array(self.reference[:3] - state[:3])[None].T
            ref_err = mujoco_quat2DCM(mujoco_rpy2quat(state[3:6])).T @ ref_err
            obs.append(np.concatenate((ref_err.squeeze(), state[3:5], heading_diff, state[6:])))
            infos.append({})

        if self.render_mode == 'human':
            self.render()

        if self.regen_env_at_steps and self.total_steps == self.regen_env_at_steps:
            self.total_steps = 0
            self.reset_model(regen=True)
            dones = np.ones(self.num_drones, dtype=bool)

        return obs, rewards, dones, infos

    def reset_model(self, regen=False):  # init the drone to start position plus some perturbation

        if regen:  # regenerate parameters of the drones and restart the simulation
            self.drone_params = self.generate_drone_params()
            model = mjcf_to_mjmodel(make_arena(self.drone_params, self.reference))  # create a mujoco model
            self.close()
            extendedEnv.__init__(
                self,
                model,
                4,
                render_mode=self.render_mode,
                observation_space=self.observation_space,
                width=self.width,
                height=self.height
            )

        start_pos = self.start_pos
        qpos = self.init_qpos
        qvel = self.init_qvel
        pos_idx_offset = 4 * self.pendulum
        vel_idx_offset = 3 * self.pendulum
        for i in range(self.num_drones):
            direction = self.np_random.normal(size=3)
            direction /= np.linalg.norm(direction)
            r = self.max_pos_offset*np.cbrt(self.np_random.random())
            # r = self.max_pos_offset
            qpos[(7 + pos_idx_offset)*i:(7 + pos_idx_offset)*i + 3] = start_pos[:3] + r*direction
            rpy = self.np_random.normal(scale=self.angle_variance, size=3).clip(min=-2*self.angle_variance, max=2*self.angle_variance) + [0, 0, start_pos[3]]
            rpy[2] = np.pi - 2*np.pi * self.np_random.random()  # uniform yaw angle
            qpos[(7 + pos_idx_offset)*i + 3:(7 + pos_idx_offset)*i + 7] = mujoco_rpy2quat(rpy)
            qvel[(6 + vel_idx_offset)*i: (6 + vel_idx_offset)*i + 3] = self.np_random.normal(scale=self.vel_variance, size=3).clip(min=-2*self.vel_variance, max=2*self.vel_variance)

        self.num_steps = np.zeros((self.num_drones, ), dtype=np.long)
        self.terminated = np.zeros((self.num_drones,), dtype=np.bool)
        self.set_state(qpos, qvel)
        print('Environment reset')
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
        r = self.max_pos_offset*np.cbrt(self.np_random.random())
        # r = self.max_pos_offset
        qpos[(7 + pos_idx_offset) * index:(7 + pos_idx_offset) * index + 3] = start_pos[:3] + r*direction
        qvel[(6 + vel_idx_offset)*index: (6 + vel_idx_offset)*index + 6] = self.init_qvel[(6 + vel_idx_offset)*index: (6 + vel_idx_offset)*index + 6]
        rpy = self.np_random.normal(scale=self.angle_variance, size=3).clip(min=-2*self.angle_variance, max=2*self.angle_variance) + [0, 0, start_pos[3]]
        rpy[2] = np.pi - 2*np.pi*self.np_random.random()  # uniform yaw angle
        qpos[(7 + pos_idx_offset)*index + 3:(7 + pos_idx_offset)*index + 7] = mujoco_rpy2quat(rpy)
        qvel[(6 + vel_idx_offset)*index: (6 + vel_idx_offset)*index + 3] = self.np_random.normal(scale=self.vel_variance, size=3).clip(min=-2*self.vel_variance, max=2*self.vel_variance)
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
            acc = self.data.sensordata[i*3:i*3+3]  # accelerometer data (given there is only one sensor on each drone)
            obs = np.concatenate((pos, angle, vel, ang_vel, acc, list(self.drone_params[i].values())))
            assert len(obs) == self.num_states + self.num_params
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