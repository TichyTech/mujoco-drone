import numpy as np
from gym import utils
from .mujoco_env_custom import extendedEnv
from gym.spaces import Box
from .env_gen import make_arena, mjcf_to_mjmodel
random_state = np.random.RandomState(42)


class TestEnv(extendedEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, num_drones=1, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        model = mjcf_to_mjmodel(make_arena(num_drones))
        extendedEnv.__init__(
            self,
            model,
            1,
            observation_space=observation_space,
            **kwargs
        )

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        terminated = bool(not np.isfinite(ob).all() or (np.abs(ob[1]) > 0.2))
        if self.render_mode == "human":
            self.render()
        return ob, reward, terminated, False, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        rgb = self._get_viewer("human").render_to_array(cam_id=0)

        print('qs', self.data.qpos)
        print('vs', self.data.qvel)
        print('sense', self.data.sensordata)
        print(rgb.shape)
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

    def viewer_setup(self):
        assert self.viewer is not None
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent