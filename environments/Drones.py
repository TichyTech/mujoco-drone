import numpy as np
from gym import utils
from .mujoco_env_custom import extendedEnv
from .env_gen import make_arena, mjcf_to_mjmodel
from gym.spaces import Dict, Box
from scipy.spatial.transform import Rotation as R
from typing import Optional, Union


class DronesEnv(extendedEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(self, num_drones=1, reference=[0, 0, 1], start_pos=None, pendulum=False, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        width, height = 640, 480
        self.num_drones = num_drones

        self.reference = reference
        if start_pos is None:
            self.start_pos = self.reference[:3]
        else:
            self.start_pos = start_pos

        # if rgb:
        #     observation_space = Dict({
        #         'pos': Box(low=-np.inf, high=np.inf, shape=(self.num_drones*7, ), dtype=np.float64),
        #         'vel': Box(low=-np.inf, high=np.inf, shape=(self.num_drones*6, ), dtype=np.float64),
        #         'rgbs': Box(low=0, high=255, shape=(self.num_drones, height, width, 3), dtype=np.uint8)
        #     })
        # else:
        #     observation_space = Dict({
        #         'pos': Box(low=-np.inf, high=np.inf, shape=(self.num_drones * 7,), dtype=np.float64),
        #         'vel': Box(low=-np.inf, high=np.inf, shape=(self.num_drones * 6,), dtype=np.float64)
        #     })

        # observation_space = Dict({
        #     'pos': Box(low=-np.inf, high=np.inf, shape=(self.num_drones * 3,), dtype=np.float64),
        #     'angles': Box(low=-np.pi, high=np.pi, shape=(self.num_drones * 3,), dtype=np.float64)
        # })
        observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_drones*6, ), dtype=np.float64)
        self.action_space = Box(low=0.5, high=1, shape=(self.num_drones * 4,), dtype=np.float64)
        model = mjcf_to_mjmodel(make_arena(self.num_drones))
        extendedEnv.__init__(
            self,
            model,
            4,
            observation_space=observation_space,
            width=width,
            height=height,
            **kwargs
        )
        self.terminated = np.zeros((self.num_drones, ))

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        terminated = np.linalg.norm(ob[:3] - self.reference) > 0.5
        # if self.render_mode == "human":
        #     self.render()
        reward = 0.1 - np.linalg.norm(ob[:3] - self.reference)
        return ob, reward, terminated, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.03, high=0.03
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        qpos[:3] = self.start_pos
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset(self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,):
        ob, _ = super().reset()
        return ob

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
        # print(self.data.sensordata)
        obs = []
        for i in range(self.num_drones):
            angle = R.from_quat(self.data.qpos[7*i +3 :7*i + 7]).as_euler('zyx')  # yaw pitch roll
            pos = self.data.qpos[7*i:7*i + 3]  # xyz positions
            obs.extend(pos)
            obs.extend(angle)
        return np.array(obs)

    def viewer_setup(self):
        assert self.viewer is not None
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent