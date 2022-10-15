from gym.envs.mujoco.mujoco_rendering import Viewer
from gym.envs.mujoco.mujoco_env import MujocoEnv
import mujoco
import numpy as np
from typing import Union, Optional
from gym.spaces import Space
from os import path


DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 640


class extended_Viewer(Viewer):

    def render_to_array(self, cam_id=-1):
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)
        width, height = self.offwidth, self.offheight
        rect = mujoco.MjrRect(left=0, bottom=0, width=width, height=height)

        cam = mujoco.MjvCamera()
        if cam_id == -1:
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        else:
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = cam_id

        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.vopt,
            self.pert,
            cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn,
        )

        mujoco.mjr_render(rect, self.scn, self.con)
        rgb_arr = np.zeros(3 * rect.width * rect.height, dtype=np.uint8)
        depth_arr = np.zeros(rect.width * rect.height, dtype=np.float32)

        mujoco.mjr_readPixels(rgb_arr, depth_arr, rect, self.con)
        rgb_img = rgb_arr.reshape(rect.height, rect.width, 3)

        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.con)
        return rgb_img


class extendedEnv(MujocoEnv):

    def __init__(
        self,
        model,
        frame_skip,
        observation_space: Space,
        render_mode: Optional[str] = None,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ):

        self.width = width
        self.height = height
        self._initialize_simulation(model)  # may use width and height

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self._viewers = {}

        self.frame_skip = frame_skip

        self.viewer = None

        assert self.metadata["render_modes"] == [
            "human",
            "rgb_array",
            "depth_array",
        ], self.metadata["render_modes"]
        assert (
            int(np.round(1.0 / self.dt)) == self.metadata["render_fps"]
        ), f'Expected value: {int(np.round(1.0 / self.dt))}, Actual value: {self.metadata["render_fps"]}'

        self.observation_space = observation_space
        self._set_action_space()

        self.render_mode = render_mode
        self.camera_name = camera_name
        self.camera_id = camera_id

    def _initialize_simulation(self, model):
        if type(model) == str:
            if model.startswith("/"):
                fullpath = model
            elif model.startswith("./"):
                fullpath = model
            else:
                fullpath = path.join(path.dirname(__file__), "assets", model)
            if not path.exists(fullpath):
                raise OSError(f"File {fullpath} does not exist")
            self.model = mujoco.MjModel.from_xml_path(fullpath)
        else:
            self.model = model
        # MjrContext will copy model.vis.global_.off* to con.off*
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)

    def _get_viewer(
        self, mode
    ) -> Union[
        "gym.envs.mujoco.mujoco_rendering.Viewer",
        "gym.envs.mujoco.mujoco_rendering.RenderContextOffscreen",
    ]:
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = extended_Viewer(self.model, self.data)
            elif mode in {"rgb_array", "depth_array"}:
                from gym.envs.mujoco.mujoco_rendering import RenderContextOffscreen

                self.viewer = RenderContextOffscreen(self.model, self.data)
            else:
                raise AttributeError(
                    f"Unexpected mode: {mode}, expected modes: {self.metadata['render_modes']}"
                )

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer