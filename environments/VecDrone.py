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
               'max_distance': 2.5,  # if drone is further from reference than this number, terminate episode
               'random_start_pos': False,  # toggle initial pose position
               'random_params': False,  # toggle randomizing drone parameters
               'pendulum': False,  # whether to include a pendulum on a drone
               'max_random_offset': 1.5,  # maximum position offset used for random sampling of starting position
               'rp_variance': [0.1, 0.1],  # variance used for random roll and pitch angle sampling
               'vel_variance': [0.03, 0.03, 0.03],  # variance used for random velocity sampling
               'ang_vel_variance': [0.03, 0.03, 0.03],  # variance used for random velocity sampling
               'body_size_interval': [0.09, 0.14],  # drone main body size in meters
               'body_mass_interval': [0.95, 1.05],  # drone main body mass in kilograms
               'arm_mult_interval': [0.95, 1.05],  # drone arm length in meters
               'pendulum_rp_variance': [0.03, 0.03],  # variance used for random velocity sampling
               'pendulum_length_interval': [0.12, 0.18],  # pendulum length in meters
               'weight_mass_interval': [0.1, 0.3],  # weight of the pendulum mass in kilograms
               'reward_fcn': distance_reward,
               'max_steps': 400,  # maximum length of a single episode
               'regen_env_at_steps': None,  # after this many (total) steps, regenerate drone model parameters
               'train_vis': 0,  # number of training environments to visualize
               'window_title': 'mujoco',
               'controlled': False  # whether this instance of env has externally controller reference (for evaluation)
}


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

        # toggle reference control
        self.controlled = config.get('controlled', False)  # is this environment using controlled reference?
        # toggle visualization window
        if config.worker_index <= config.get('train_vis', 0) or self.controlled:
            self.render_mode = 'human'
        else:
            self.render_mode = None
        self.window_title = config.get('window_title', 'mujoco')
        # finally, disable control, if there is no controller found
        if self.controlled:
            print('Initializing controller')
            self.joystick = PS4Controller()  # also works with PS5 controller
            if not self.joystick.controller:  # if no controller found, disable control
                self.controlled = False
                print('Disabling reference control')

        # get generate environment parameters
        self.reference = config.get('reference', [0, 0, 0, 0])
        self.num_drones = config.get('num_drones', 1)
        self.pendulum = config.get('pendulum', False)
        self.body_size_interval = np.array(config.get('body_size_interval', [0.04, 0.04]))
        self.body_mass_interval = np.array(config.get('body_mass_interval', [0.8, 1]))
        self.arm_mult_interval = np.array(config.get('arm_mult_interval', [0.015, 0.015]))
        self.pendulum_length_interval = np.array(config.get('pendulum_length_interval', [0.0125, 0.0125]))
        self.weight_mass_interval = np.array(config.get('weight_mass_interval', [0.2, 0.2]))

        # setup training parameters
        self.random_start_pos = config.get('random_start_pos', False)
        self.random_params = config.get('random_params', False)
        self.regen_env_at_steps = config.get('regen_env_at_steps', None)
        self.start_pos = config.get('start_pos', self.reference)
        self.max_distance = config.get('max_distance', 1)
        self.reward_fcn = config.get('reward_fcn', distance_reward)
        self.max_steps = config.get('max_steps', 400)
        self.max_pos_offset = config.get('max_random_offset', 0)
        self.angle_variance = np.array(config.get('angle_variance', [0, 0]))
        self.ang_vel_variance = np.array(config.get('ang_vel_variance', [0, 0, 0]))
        self.vel_variance = np.array(config.get('vel_variance', [0, 0, 0]))
        self.pendulum_rp_variance = np.array(config.get('pendulum_rp_variance', [0, 0]))

        # setup
        self.total_steps = 0
        self.num_steps = np.zeros((self.num_drones, ), dtype=np.long)
        self.terminated = np.zeros((self.num_drones,), dtype=np.bool)

        # set random number generator seed for reproducibility
        rng, seed = utils.seeding.np_random(seed=config.worker_index)
        self.np_random = rng

        # generate randomized parameters for each drone and save them into a list
        self.drone_params = self.generate_drone_params()
        self.num_params = len(self.drone_params[0])
        self.num_states = 12
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_states + self.num_params,), dtype=np.float64)
        self.action_space = Box(low=0, high=1, shape=(4,), dtype=np.float64)
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

        # init VectorEnv for rllib
        VectorEnv.__init__(self, self.observation_space, self.action_space, self.num_drones)
        print('Environment ready')

    def control_reference(self):
        """read joystick state and apply changes to the reference vector"""
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
        """same as move_reference_to(), but using relative offset instead of absolute coordinates"""
        self.move_reference_to(self.reference + offset)

    def move_reference_to(self, target):
        """update reference vector as well as in-simulation reference visualization"""
        self.reference = target
        self.reference[3] = (self.reference[3] + np.pi) % (2*np.pi) - np.pi
        self.reference = np.clip(self.reference, a_min=[-5, -5, 0, -np.pi], a_max=[5, 5, 6, np.pi])  # update reference
        self.data.mocap_pos[:3] = self.reference[:3]  # update reference visualization
        self.data.mocap_quat[:] = mujoco_rpy2quat([0, 0, self.reference[3]])

    def generate_drone_params(self):
        """sample drone model parameters using specified parameters if enabled"""
        drone_params = []
        # load parameter intervals
        l_bs, h_bs = self.body_size_interval
        l_bm, h_bm = self.body_mass_interval
        l_am, h_am = self.arm_mult_interval
        l_pl, h_pl = self.pendulum_length_interval
        l_wm, h_wm = self.weight_mass_interval
        if self.random_params:
            # generate random values uniformly from the intervals
            body_sizes = self.np_random.uniform(l_bs, h_bs, self.num_drones)
            body_masses = self.np_random.uniform(l_bm, h_bm, self.num_drones)
            arm_mults = self.np_random.uniform(l_am, h_am, self.num_drones)
            pendulum_lens = self.np_random.uniform(l_pl, h_pl, self.num_drones)
            weight_masses = self.np_random.uniform(l_wm, h_wm, self.num_drones)
        else:
            # use mean values of the intervals instead
            body_sizes = 0.5*(h_bs + l_bs)*np.ones(self.num_drones)
            body_masses = 0.5*(h_bm + l_bm)*np.ones(self.num_drones)
            arm_mults = 0.5*(h_am + l_am)*np.ones(self.num_drones)
            pendulum_lens = 0.5*(h_pl + l_pl)*np.ones(self.num_drones)
            weight_masses = 0.5*(h_wm + l_wm)*np.ones(self.num_drones)
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

    def sample_state(self):
        """returns a drone state sampled randomly from specified parameters if enabled"""
        if self.random_start_pos:  # initial poses generated randomly
            # sample random point inside sphere uniformly
            direction = self.np_random.normal(size=3)
            direction /= np.linalg.norm(direction)
            r = self.max_pos_offset * np.cbrt(self.np_random.random())
            pos = self.start_pos[:3] + r * direction
            # sample rp angles from a clipped gaussian
            rp = self.np_random.normal(scale=self.angle_variance, size=2)\
                      .clip(min=-2 * self.angle_variance, max=2 * self.angle_variance)
            # sample yaw angle uniformly
            y = np.pi - 2 * np.pi * self.np_random.random()  # uniform yaw angle
            rpy = np.append(rp, y)
            qpos = np.concatenate((pos, mujoco_rpy2quat(rpy)))  # convert to mujoco format of qpos
            # sample velocity and angular velocity vectors from clipped normal distributions
            vel = self.np_random.normal(scale=self.vel_variance).clip(min=-2 * self.vel_variance, max=2 * self.vel_variance)
            ang_vel = self.np_random.normal(scale=self.ang_vel_variance).clip(min=-2 * self.ang_vel_variance, max=2 * self.ang_vel_variance)
            qvel = np.concatenate((vel, ang_vel))
            if self.pendulum:  # if pendulum is enabled, generate its state as well
                # sample roll and pitch from clipped gaussian
                pendulum_rp = self.np_random.normal(scale=self.pendulum_rp_variance)\
                    .clip(min=-2 * self.pendulum_rp_variance, max=2 * self.pendulum_rp_variance)
                pendulum_qpos = mujoco_rpy2quat(np.append(pendulum_rp, 0))
                pendulum_vel = [0, 0, 0]  # zero initial velocity for now
                qpos = np.concatenate((qpos, pendulum_qpos))
                qvel = np.concatenate((qvel, pendulum_vel))
        else:
            # deterministic inital pose
            pos = self.start_pos[:3]
            rpy = np.append([0, 0], self.start_pos[3])
            qpos = np.concatenate((pos, mujoco_rpy2quat(rpy)))
            # deterministic initial velocity
            vel = [0, 0, 0]
            ang_vel = [0, 0, 0]
            qvel = np.concatenate((vel, ang_vel))
            if self.pendulum:
                pendulum_qpos = mujoco_rpy2quat([0, 0, 0])  # all zeroes for now
                pendulum_vel = [0, 0, 0]
                qpos = np.concatenate((qpos, pendulum_qpos))
                qvel = np.concatenate((qvel, pendulum_vel))
        return qpos, qvel

    def vector_step(self, actions):
        """perform a simulation step on all drone given the specified actions"""
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
            pos_err = np.linalg.norm(state[:3] - self.reference[:3])  # distance from reference coordinates
            self.terminated[i] = pos_err > self.max_distance or self.num_steps[i] >= self.max_steps
            reward = self.reward_fcn(self, state, actions[i], self.num_steps[i])
            rewards.append(reward)
            dones.append(self.terminated[i])
            heading_diff = np.array((self.reference[3] - state[5] + np.pi) % (2 * np.pi) - np.pi)[None]
            glob_ref_err = np.array(self.reference[:3] - state[:3])[None].T
            loc_ref_err = mujoco_quat2DCM(mujoco_rpy2quat(state[3:6])).T @ glob_ref_err  # reference direction in local frame
            glob_vel = np.array(state[6:9])[None].T
            loc_vel = mujoco_quat2DCM(mujoco_rpy2quat(state[3:6])).T @ glob_vel  # velocity in local frame
            glob_ang_vel = np.array(state[9:12])[None].T
            loc_ang_vel = mujoco_quat2DCM(mujoco_rpy2quat(state[3:6])).T @ glob_ang_vel
            obs.append(np.concatenate((loc_ref_err.squeeze(), state[3:5], heading_diff, loc_vel.squeeze(), loc_ang_vel.squeeze(), state[12:])))
            infos.append({})

        if self.render_mode == 'human':
            self.render()

        if self.random_params and self.regen_env_at_steps and self.total_steps == self.regen_env_at_steps:
            self.total_steps = 0
            self.reset_model(regen=True)  # reset model with regenerated drone parameters
            dones = np.ones(self.num_drones, dtype=bool)  # set all drones to terminated

        return obs, rewards, dones, infos

    def reset_model(self, regen=False):
        """regenerates all the drone states and also their model parameters if enabled """
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

        qpos = self.init_qpos  # copy mujoco state vector
        qvel = self.init_qvel
        p_offset = 4 * self.pendulum  # if there is pendulum, add extra offset to indeces
        v_offset = 3 * self.pendulum
        for i in range(self.num_drones):  # generate initial poses and populate the mujoco state array
            qpos_i, qvel_i = self.sample_state()
            qpos[(7 + p_offset) * i:(7 + p_offset) * (i + 1)] = qpos_i
            qvel[(6 + v_offset) * i:(6 + v_offset) * (i + 1)] = qvel_i

        self.num_steps = np.zeros((self.num_drones, ), dtype=np.long)  # reset per drone number of steps
        self.terminated = np.zeros((self.num_drones,), dtype=np.bool)  # reset per drone termination info
        self.set_state(qpos, qvel)  # set the mujoco state
        print('Environment reset')
        return self._get_obs()

    def vector_reset(self):
        """reset all the drones"""
        ob = self.reset_model()
        print('Vector reset')
        return ob

    def reset_at(self, index):
        """reset state of a drone given by its index"""
        if index is None:
            index = 0
        assert index < self.num_drones

        p_offset = 4 * self.pendulum  # add extra offset if there is pendulum state enabled
        v_offset = 3 * self.pendulum
        qpos = self.data.qpos[:]  # copy mujoco state
        qvel = self.data.qvel[:]
        qpos_i, qvel_i = self.sample_state()  # generate an initial state
        qpos[(7 + p_offset) * index:(7 + p_offset) * (index + 1)] = qpos_i
        qvel[(6 + v_offset) * index:(6 + v_offset) * (index + 1)] = qvel_i
        self.set_state(qpos, qvel)  # update the mujoco state
        self.num_steps[index] = 0  # reset the per drone step count
        return self._get_obs()[index]

    def _get_obs(self):
        """returns an array of per drone observations"""
        states = []  # state info for every drone
        pos_idx_offset = 4 * self.pendulum  # add an index offset if pendulum is enabled
        vel_idx_offset = 3 * self.pendulum
        for i in range(self.num_drones):
            # all these observations correspond to the free joint coordinates and thus are in global coord. frame
            pos = self.data.qpos[(7 + pos_idx_offset)*i:(7 + pos_idx_offset)*i + 3]  # xyz position
            angle = mujoco_quat2rpy(self.data.qpos[(7 + pos_idx_offset)*i + 3:(7 + pos_idx_offset)*i + 7])  # rpy angles
            vel = self.data.qvel[(6 + vel_idx_offset)*i:(6 + vel_idx_offset)*i + 3]  # xyz velocity
            ang_vel = self.data.qvel[(6 + vel_idx_offset)*i + 3:(6 + vel_idx_offset)*i + 6]  # rpy velocity (probably in different order)
            # accelerometer data, should be in local coord. frame
            # acc = self.data.sensordata[i*3:i*3+3]  # accelerometer data (given there is only one sensor on each drone)
            obs = np.concatenate((pos, angle, vel, ang_vel, list(self.drone_params[i].values())))
            assert len(obs) == self.num_states + self.num_params
            states.append(obs)  # add state to state list

        # collision checking if needed
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