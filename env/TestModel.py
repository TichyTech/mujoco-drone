import numpy as np
from matplotlib.colors import hsv_to_rgb
from gym import utils
from .mujoco_env_custom import MujocoEnv
from gym.spaces import Box
from dm_control import mjcf
from dm_control import mujoco

random_state = np.random.RandomState(42)

def make_drone(hue):
    rgba = [0, 0, 0, 1]
    rgba[:3] = hsv_to_rgb([hue, 1, 1])

    model = mjcf.RootElement()
    model.compiler.angle = 'radian'
    model.compiler.inertiafromgeom = True
    model.compiler.exactmeshinertia = True

    model.option.timestep = 0.01

    model.default.joint.damping = 1
    model.default.joint.armature = 1

    model.default.geom.type = 'box'
    model.default.geom.rgba = rgba
    model.default.geom.conaffinity = 0
    model.default.geom.condim = 3
    model.default.geom.friction = (1, 0.5, 0.5)
    model.default.geom.margin = 0

    model.worldbody.add('geom', name='core_geom', size=(0.04, 0.04, 0.02), mass=0.1, rgba=rgba)
    model.worldbody.add('site', name='sense', pos=(0, 0, -0.01))
    model.sensor.add('accelerometer', site='sense')
    model.sensor.add('gyro', site='sense')
    model.sensor.add('magnetometer', site='sense')
    for i in range(4):
        theta = 2 * i * np.pi / 4
        hip_pos = 0.07 * np.array([np.cos(theta), np.sin(theta), 0])
        model.worldbody.add('geom', size=(0.05, 0.005, 0.005), pos=hip_pos, euler=[0, 0, theta], rgba=rgba, mass=0.025)
        model.worldbody.add('site', name='motor%d' % i, type='cylinder', pos=1.6*hip_pos + np.array([0, 0, 0.0075]), size=(0.01, 0.0025), rgba=rgba)
        model.worldbody.add('geom', type='cylinder', pos=1.6 * hip_pos + np.array([0, 0, 0.01]), size=(0.05, 0.0025), rgba=rgba, mass=0.025)
        model.actuator.add('motor', site='motor%d' % i, gear=(0, 0, 1, 0, 0, 0.1*(-1)**i), ctrllimited=True, ctrlrange=(0, 1))
    return model

arena = mjcf.RootElement()
chequered = arena.asset.add('texture', type='2d', builtin='checker', width=300,
                            height=300, rgb1=[.2, .3, .4], rgb2=[.3, .4, .5])
grid = arena.asset.add('material', name='grid', texture=chequered,
                       texrepeat=[5, 5], reflectance=.2)
arena.worldbody.add('geom', type='plane', size=[10, 10, .1], material=grid)
arena.worldbody.add('light', pos=[0, 0, 3], cutoff=100, dir=[0, 0, -1.3])

total = 1
drones = [make_drone(i/total) for i in range(total)]
height = .15
grid = 0.5
xpos, ypos, zpos = np.meshgrid([-grid, 0, grid], [0, grid], [height])
for i, model in enumerate(drones):
  spawn_pos = (xpos.flat[i], ypos.flat[i], zpos.flat[i])
  spawn_site = arena.worldbody.add('site', pos=spawn_pos, group=3)
  spawn_site.attach(model).add('freejoint')

xml_string = arena.to_xml_string()
assets = arena.get_assets()
model = mujoco.MjModel.from_xml_string(xml_string)


class TestEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        MujocoEnv.__init__(
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
        print('qs', self.data.qpos)
        print('vs', self.data.qvel)
        print('sense', self.data.sensordata)
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

    def viewer_setup(self):
        assert self.viewer is not None
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent