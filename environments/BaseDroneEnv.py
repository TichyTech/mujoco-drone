import numpy as np
from gymnasium import utils
from .mujoco_env_custom import extendedEnv
from .env_gen import make_arena, mjcf_to_mjmodel
from .joystick import PS4Controller
from gymnasium.spaces import Box
from ray.rllib.env.vector_env import VectorEnv
from .transformation import mujoco_quat2DCM, mujoco_quat2rpy, mujoco_rpy2quat
from .rewards import default_reward_fcn


def default_termination_fcn(env, state, action, num_steps):
    """terminates episode if too far from reference position or if the episode is too long"""
    pos_err = np.linalg.norm(state[:3] - env.reference[:3])  # distance from reference coordinates
    terminated = pos_err > env.max_distance or num_steps >= env.max_steps
    return terminated


base_config = {'frequency': 200,  # physics simulator frequency
               'skip_steps': 2,  # policy takes action every skip_steps steps
               'reference': [0, 0, 15, 0],  # x,y,z,yaw
               'start_pos': [0, 0, 15, 0],  # x,y,z,yaw
               'max_distance': 4,  # if drone is further from reference than this number, terminate episode
               'random_start_pos': True,  # toggle initial pose position
               'random_params': False,  # toggle randomizing drone parameters
               'pendulum': True,  # whether to include a pendulum on a drone
               'difficulty': 0.1,
               'max_random_offset': 2,  # maximum position offset used for random sampling of starting position
               'rp_variance': [0.8, 0.8],  # variance used for random roll and pitch angle sampling
               'vel_variance': [1, 1, 1],  # variance used for random velocity sampling
               'ang_vel_variance': [1, 1, 1],  # variance used for random velocity sampling
               'mass_interval': [1.2, 1.5],  # drone mass in kilograms
               'arm_len_interval': [0.15, 0.18],  # drone arm length in meters
               'motor_force_interval': [6, 9],  # what force the motor produces in Newtons, torque is also affected
               'motor_tau_interval': [0.001, 0.005],  # time constant of the motor in seconds, 1/tau is crossover f of the LP filter
               'pendulum_rp_variance': [0.5, 0.5],  # variance used for random velocity sampling
               'pendulum_length_interval': [0.2, 0.4],  # pendulum length in meters
               'weight_mass_interval': [0.2, 0.6],  # weight of the pendulum mass in kilograms
               'reward_fcn': default_reward_fcn,
               'terminated_fcn': default_termination_fcn,
               'max_steps': 512,  # maximum length of a single episode
               'regen_env_at_steps': None,  # after this many (total) steps, regenerate drone model parameters
               'train_vis': 0,  # number of training environments to visualize
               'window_title': 'mujoco',
               'controlled': False  # whether this instance of env has externally controller reference (for evaluation)
}


class BaseDroneEnv(extendedEnv, VectorEnv, utils.EzPickle):

    def __init__(self, config, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        self.width, self.height = 640, 480

        # toggle reference control
        self.controlled = config.get('controlled', False)  # is this environment using controlled reference?
        # toggle visualization window
        worker_index = getattr(config, 'worker_index', -1)
        if worker_index <= config.get('train_vis', 0) or self.controlled:
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
        self.skip_steps = config.get('skip_steps', 1)
        self.frequency = config.get('frequency', 200)
        self.reference = config.get('reference', [0, 0, 0, 0])
        self.num_drones = config.get('num_drones', 1)
        self.pendulum = config.get('pendulum', True)
        self.mass_interval = np.array(config.get('mass_interval', [0.8, 1.2]))
        self.arm_len_interval = np.array(config.get('arm_len_interval', [0.15, 0.18]))
        self.motor_force_interval = np.array(config.get('motor_force_interval', [5, 7]))
        self.motor_tau_interval = np.array(config.get('motor_tau_interval', [0.005, 0.01]))
        self.pendulum_length_interval = np.array(config.get('pendulum_length_interval', [0.0125, 0.0125]))
        self.weight_mass_interval = np.array(config.get('weight_mass_interval', [0.2, 0.2]))

        # setup training parameters
        self.difficulty = config.get('difficulty', 0.1)
        self.random_start_pos = config.get('random_start_pos', False)
        self.random_params = config.get('random_params', False)
        self.regen_env_at_steps = config.get('regen_env_at_steps', None)
        self.start_pos = config.get('start_pos', self.reference)
        self.max_distance = config.get('max_distance', 1)
        self.reward_fcn = config.get('reward_fcn', default_reward_fcn)
        self.terminated_fcn = config.get('terminated_fcn', default_termination_fcn)
        self.max_steps = config.get('max_steps', 512)
        self.max_pos_offset = self.difficulty*config.get('max_random_offset', 0)
        self.angle_variance = self.difficulty*np.array(config.get('angle_variance', [0, 0]))
        self.ang_vel_variance = self.difficulty*np.array(config.get('ang_vel_variance', [0, 0, 0]))
        self.vel_variance = self.difficulty*np.array(config.get('vel_variance', [0, 0, 0]))
        self.pendulum_rp_variance = self.difficulty*np.array(config.get('pendulum_rp_variance', [0, 0]))

        # setup
        self.total_steps = 0
        self.num_steps = np.zeros((self.num_drones, ), dtype=np.long)

        # set random number generator seed for reproducibility
        rng, seed = utils.seeding.np_random(seed=config.get('worker_index', -1) + 1)
        self.np_random = rng

        # generate randomized parameters for each drone and save them into a list
        self.drone_params = self.generate_drone_params()
        self.num_params = len(self.drone_params[0])
        if self.pendulum:  # pendulum enabled
            self.num_states = 25
        else:
            self.num_states = 19
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_states + self.num_params,), dtype=np.float64)
        self.action_space = Box(low=0, high=1, shape=(4,), dtype=np.float64)
        model = mjcf_to_mjmodel(make_arena(self.drone_params, self.reference, self.frequency))  # create a mujoco model

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": self.frequency//self.skip_steps,
        }

        extendedEnv.__init__(
            self,
            model,
            frame_skip=self.skip_steps,
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
        # clip reference to some box around start position
        center = np.array(self.start_pos[:3])
        self.reference[:3] = np.clip(self.reference[:3], a_min=center + [-5, -5, -6], a_max=center + [5, 5, 6])  # update reference
        self.data.mocap_pos[:3] = self.reference[:3]  # update reference visualization
        self.data.mocap_quat[:] = mujoco_rpy2quat([0, 0, self.reference[3]])

    def generate_drone_params(self):
        """sample drone model parameters using specified parameters if enabled"""
        drone_params = []
        # load parameter intervals
        l_bm, h_bm = self.mass_interval
        l_al, h_al = self.arm_len_interval
        l_mf, h_mf = self.motor_force_interval
        l_mt, h_mt = self.motor_tau_interval
        l_pl, h_pl = self.pendulum_length_interval
        l_wm, h_wm = self.weight_mass_interval
        if self.random_params:
            # generate random values uniformly from the intervals
            masses = self.np_random.uniform(l_bm, h_bm, self.num_drones)
            arm_lens = self.np_random.uniform(l_al, h_al, self.num_drones)
            motor_forces = self.np_random.uniform(l_mf, h_mf, self.num_drones)
            motor_taus = self.np_random.uniform(l_mt, h_mt, self.num_drones)
            pendulum_lens = self.np_random.uniform(l_pl, h_pl, self.num_drones)
            weight_masses = self.np_random.uniform(l_wm, h_wm, self.num_drones)
        else:
            # use mean values of the intervals instead
            masses = 0.5*(h_bm + l_bm)*np.ones(self.num_drones)
            arm_lens = 0.5*(l_al + h_al)*np.ones(self.num_drones)
            motor_forces = 0.5*(l_mf + h_mf)*np.ones(self.num_drones)
            motor_taus = 0.5*(l_mt + h_mt)*np.ones(self.num_drones)
            pendulum_lens = 0.5*(h_pl + l_pl)*np.ones(self.num_drones)
            weight_masses = 0.5*(h_wm + l_wm)*np.ones(self.num_drones)
        # save parameters into a list of dictionaries
        for i in range(self.num_drones):
            params = {'mass': masses[i],
                      'arm_len': arm_lens[i],
                      'motor_force': motor_forces[i],
                      'motor_tau': motor_taus[i],
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
        """Performs a simulation step on all drone given the specified actions. Each action in the actions list consists
        of a 4-item array, where each entry corresponds to the given motor throttle in the interval [0, 1]."""
        if self.controlled:
            self.control_reference()  # move reference accordingly

        ctrl = 0.1 + 0.9*np.array(actions).ravel()  # reformat actions for mujoco and constrain to [0.1, 1]
        self.do_simulation(ctrl, self.frame_skip)
        self.num_steps = self.num_steps + 1  # keep count of episode lengths
        self.total_steps += 1  # keep count of total simulation steps performed

        states = self.get_drone_states()
        obs, rewards, dones, truncated, infos = [], [], [], [], []
        for i in range(self.num_drones):
            state_i = states[i]  # load state
            terminated = self.terminated_fcn(self, state_i, actions[i], self.num_steps[i])  # decide episode termination
            reward = self.reward_fcn(self, state_i, actions[i], self.num_steps[i])  # compute reward
            # add all values to the output lists
            rewards.append(reward)
            dones.append(terminated)
            truncated.append(False)
            infos.append({})

        if self.render_mode == 'human':  # if rendering is enabled, render after each simulation step
            self.render()

        if self.random_params and self.regen_env_at_steps and self.total_steps == self.regen_env_at_steps:
            self.total_steps = 0
            self.reset_model(regen=True)  # reset model with regenerated drone parameters
            dones = np.ones(self.num_drones, dtype=bool)  # set all drones to terminated

        return self._get_obs(), rewards, dones, truncated, infos

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
        self.set_state(qpos, qvel)  # set the mujoco state
        print('Environment reset')
        return self._get_obs()

    def vector_reset(self, seeds=None, options=None):
        """reset all the drones"""
        obs = self.reset_model()
        print('Vector reset')
        infos = [{}]*self.num_drones
        return obs, infos

    def reset_at(self, index, seed=None, options=None):
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
        info = {}
        ob = self._get_obs()[index]
        return ob, info

    def _get_obs(self):
        """defaults observation function consists of just the states"""
        obs = self.get_drone_states()
        return obs

    def get_drone_states(self):
        """Returns an array of per drone states. Each state consists of the position, rpy angles, velocity,
        angular velocity vector, the acceleration vector, reference vector and the drone model parameters.
        The vectors are represented in the global coordinate frame.
        """
        states = []  # state info for every drone
        pos_idx_offset = 4 * self.pendulum  # add an index offset if pendulum is enabled
        vel_idx_offset = 3 * self.pendulum
        for i in range(self.num_drones):
            # all these observations correspond to the free joint coordinates and thus are in global coord. frame
            pos = self.data.qpos[(7 + pos_idx_offset)*i:(7 + pos_idx_offset)*i + 3]  # xyz position
            rpy = mujoco_quat2rpy(self.data.qpos[(7 + pos_idx_offset)*i + 3:(7 + pos_idx_offset)*i + 7])  # rpy angles
            vel = self.data.qvel[(6 + vel_idx_offset)*i:(6 + vel_idx_offset)*i + 3]  # xyz velocity
            ang_vel = self.data.qvel[(6 + vel_idx_offset)*i + 3:(6 + vel_idx_offset)*i + 6]  # rpy velocity (probably in different order)
            acc = self.data.sensordata[i*3:i*3+3]  # accelerometer data (given there is only one sensor on each drone)
            if self.pendulum:
                pendulum_rpy = mujoco_quat2rpy(self.data.qpos[(7 + pos_idx_offset)*i + 7:(7 + pos_idx_offset)*i + 11])
                pendulum_ang_vel = self.data.qvel[(6 + vel_idx_offset)*i + 6:(6 + vel_idx_offset)*i + 9]
                obs = np.concatenate((pos, rpy, vel, ang_vel, pendulum_rpy, pendulum_ang_vel, acc, self.reference, list(self.drone_params[i].values())))
            else:
                obs = np.concatenate((pos, rpy, vel, ang_vel, acc, self.reference, list(self.drone_params[i].values())))
            states.append(obs)  # add state to state list
        return states

    def viewer_setup(self):
        assert self.viewer is not None
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent