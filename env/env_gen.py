import numpy as np
from matplotlib.colors import hsv_to_rgb
from dm_control import mjcf
from dm_control import mujoco


def make_drone(id=0, hue=1):
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
    model.worldbody.add('camera', name='dronecam%d' % id, pos=(0.02, 0, 0.01))
    for i in range(4):
        theta = 2 * i * np.pi / 4
        hip_pos = 0.07 * np.array([np.cos(theta), np.sin(theta), 0])
        model.worldbody.add('geom', size=(0.05, 0.005, 0.005), pos=hip_pos, euler=[0, 0, theta], rgba=rgba, mass=0.025)
        model.worldbody.add('site', name='motor%d' % i, type='cylinder', pos=1.6*hip_pos + np.array([0, 0, 0.0075]), size=(0.01, 0.0025), rgba=rgba)
        model.worldbody.add('geom', type='cylinder', pos=1.6 * hip_pos + np.array([0, 0, 0.01]), size=(0.05, 0.0025), rgba=rgba, mass=0.025)
        model.actuator.add('motor', site='motor%d' % i, gear=(0, 0, 1, 0, 0, 0.1*(-1)**i), ctrllimited=True, ctrlrange=(0, 1))
    return model


def make_arena(num_drones=1):
    arena = mjcf.RootElement()
    chequered = arena.asset.add('texture', type='2d', builtin='checker', width=300,
                                height=300, rgb1=[.2, .3, .4], rgb2=[.3, .4, .5])
    grid = arena.asset.add('material', name='grid', texture=chequered,
                           texrepeat=[5, 5], reflectance=.2)
    arena.worldbody.add('geom', type='plane', size=[10, 10, .1], material=grid)
    arena.worldbody.add('light', pos=[0, 0, 3], cutoff=100, dir=[0, 0, -1.3])

    drones = [make_drone(i, i/num_drones) for i in range(num_drones)]
    height = .15
    grid = 0.5
    xpos, ypos, zpos = np.meshgrid([-grid, 0, grid], [0, grid], [height])
    for i, model in enumerate(drones):
        spawn_pos = (xpos.flat[i], ypos.flat[i], zpos.flat[i])
        spawn_site = arena.worldbody.add('site', pos=spawn_pos, group=3)
        spawn_site.attach(model).add('freejoint')

    return arena


def mjcf_to_mjmodel(mjcf_model):
    xml_string = mjcf_model.to_xml_string()
    # assets = arena.get_assets()
    model = mujoco.MjModel.from_xml_string(xml_string)
    return model
