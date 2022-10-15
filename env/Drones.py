import mujoco
import numpy as np
from gym import utils
from .mujoco_env_custom import extendedEnv
from .env_gen import make_arena, mjcf_to_mjmodel
from gym.spaces import Dict, Box


class DronesEnv(extendedEnv, utils.EzPickle):

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
        width, height = 640, 480
        self.num_drones = num_drones
        observation_space = Dict({
            'pos': Box(low=-np.inf, high=np.inf, shape=(self.num_drones*7, ), dtype=np.float64),
            'vel': Box(low=-np.inf, high=np.inf, shape=(self.num_drones*6, ), dtype=np.float64),
            'rgbs': Box(low=0, high=255, shape=(self.num_drones, height, width, 3), dtype=np.uint8)
        })
        model = mjcf_to_mjmodel(make_arena(self.num_drones))
        extendedEnv.__init__(
            self,
            model,
            1,
            observation_space=observation_space,
            width=width,
            height=height,
            **kwargs
        )
        self.terminated = np.zeros((self.num_drones, ))

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        positions = ob['pos']
        for i in range(self.num_drones):
            self.terminated[i] = positions[i*7 + 2] > 5
        terminated = self.terminated.all()
        if self.render_mode == "human":
            self.render()
        return ob, reward, terminated, False, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.03, high=0.03
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        images = []
        # for i in range(self.num_drones):
        #     rgb = self._get_viewer("human").render_to_array(cam_id=i)
        #     images.append(rgb)
        # ncol = self.data.ncon  # number of collisions
        # print('%d collisions' % ncol)
        # for i in range(ncol):
        #     con = self.data.contact[i]
        #     print('geom1', con.geom1, mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1,))
        #     print('geom2', con.geom2, mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2,))
        return {'pos': self.data.qpos, 'vel': self.data.qvel, 'rgbs': np.array(images)}

    def viewer_setup(self):
        assert self.viewer is not None
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent