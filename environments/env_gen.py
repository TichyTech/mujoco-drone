import numpy as np
from matplotlib.colors import hsv_to_rgb
from dm_control import mjcf
from dm_control import mujoco
from scipy.spatial.transform import Rotation as R


def make_drone(id=0, hue=1, pendulum=False):
    rgba = [0, 0, 0, 1]
    rgba[:3] = hsv_to_rgb([hue, 1, 1])

    model = mjcf.RootElement(model='drone_%d' % id)

    model.default.geom.type = 'box'
    model.default.geom.rgba = rgba
    model.default.geom.conaffinity = 0
    model.default.geom.contype = 1
    model.default.geom.condim = 3
    model.default.geom.friction = (1, 0.5, 0.5)
    model.default.geom.margin = 0

    model.default.joint.damping = 0.01
    model.default.joint.armature = 0.0

    body_size = 0.02
    arm_size = 0.15
    pendulum_length = 0.1 + 0.05*np.random.rand()
    pole_mass = pendulum_length
    weight_mass = 0.15 + 0.1*np.random.rand()

    core = model.worldbody.add('body', name='core_body_%d' % id, pos=(0, 0, 0))
    core.add('geom', name='core_geom_%d' % id, size=(2*body_size, 2*body_size, body_size), mass=0.9, rgba=rgba)
    core.add('geom', pos=[body_size, 0, 0], name='front_%d' % id, size=(3*body_size, 0.3*body_size, 0.3*body_size), mass=0, rgba=rgba)
    core.add('site', name='sense', pos=(0, 0, -body_size/2))
    model.sensor.add('accelerometer', name='acc_%d' % id, site='sense')
    model.sensor.add('gyro', name='gyro_%d' % id, site='sense')
    model.sensor.add('magnetometer', name='mag_%d' % id, site='sense')
    core.add('camera', name='dronecam_%d' % id, pos=(body_size, 0, body_size/2), euler=[0, np.pi, 0])  # TODO: check angle
    for i in range(4):
        theta = i * np.pi / 2 + np.pi/4
        arm_pos = (1.4*body_size + 0.5*arm_size) * np.array([np.cos(theta), np.sin(theta), 0])
        core.add('geom', name='arm_%d' % i, size=(arm_size, 0.005, 0.005), pos=arm_pos, euler=[0, 0, theta], mass=0.025)
        core.add('site', name='motorsite_%d' % i, type='cylinder', pos=2.2 * arm_pos + np.array([0, 0, 0.0075]), size=(0.01, 0.0025))
        core.add('geom', name='prop_%d' % i, type='cylinder', pos=2.2 * arm_pos + np.array([0, 0, 0.01]), size=(0.05, 0.0025), mass=0.025)
        model.actuator.add('motor', name='motor_%d' % i, site='motorsite_%d' % i, gear=(0, 0, 6, 0, 0, 0.6*(-1)**i), ctrllimited=True, ctrlrange=(0, 1))
    if pendulum:
        pend = core.add('body', name='pendulum', pos=(0, 0, 0))
        pend.add('joint', type='ball', pos=(0, 0, -body_size))
        pend.add('geom', size=(0.005, pendulum_length), pos=(0, 0, -pendulum_length), type='cylinder', mass=pole_mass)
        pend.add('geom', pos=(0, 0, -2*pendulum_length), type='sphere', size=[0.05], mass=weight_mass)
    return model


def make_arena(num_drones=1, pendulum=False, reference=None):
    arena = mjcf.RootElement(model='arena')

    arena.size.nconmax = 1000  # set maximum number of collisions
    arena.size.njmax = 2000  # increase the maximum number of constraints

    arena.option.timestep = 0.005
    arena.option.density = 1.2  # enable density and viscosity of air to compute drag forces
    arena.option.viscosity = 0.00002
    arena.option.wind = (0, 0, 0)  # wind direction and speed

    arena.compiler.angle = 'radian'
    arena.compiler.inertiafromgeom = True
    arena.compiler.exactmeshinertia = True


    chequered = arena.asset.add('texture', name='checker', type='2d', builtin='checker', width=300,
                                height=300, rgb1=[.2, .3, .4], rgb2=[.3, .4, .5])
    grid = arena.asset.add('material', name='grid', texture=chequered,
                           texrepeat=[5, 5], reflectance=.2)
    arena.asset.add('texture', name='skybox', type='skybox', builtin='gradient',
                    rgb1=(.4, .6, .8), rgb2=(0, 0, 0), width=800, height=800, mark="random", markrgb=(1, 1, 1))
    arena.worldbody.add('geom', name='floor', type='plane', size=[10, 10, 0.5], material=grid)
    arena.worldbody.add('light', name='light', pos=[0, 0, 6], cutoff=100, dir=[0, 0, -1.3])

    if reference is not None:  # draw reference
        arena.worldbody.add('geom', pos=reference[:3], type='sphere', size=[0.1], rgba=(1, 1, 1, 1), contype=1, conaffinity=0)
        pos = reference[:3] + 0.075*np.array([np.cos(reference[3]), np.sin(reference[3]), 0])
        arena.worldbody.add('geom', pos=pos, euler=[0, 0, reference[3]], size=[0.15, 0.01, 0.01],
                            type='box', rgba=(1, 0, 0, 1), contype=1, conaffinity=0)

    drones = [make_drone(i, i/num_drones, pendulum) for i in range(num_drones)]
    height = .15
    margin = 0.5
    sz = np.ceil(np.sqrt(num_drones)).astype(np.long)
    steps = (np.arange(sz) - (sz-1)/2) * margin
    xpos, ypos, zpos = np.meshgrid(steps, steps, [height])
    for i, model in enumerate(drones):
        spawn_pos = (xpos.flat[i], ypos.flat[i], zpos.flat[i])
        spawn_site = arena.worldbody.add('site', name='spawnsite_%d' % i, pos=spawn_pos, group=3)
        spawn_site.attach(model).add('freejoint')
    return arena


def mjcf_to_mjmodel(mjcf_model):
    xml_string = mjcf_model.to_xml_string(precision=5)
    # print(xml_string)
    # assets = arena.get_assets()
    model = mujoco.MjModel.from_xml_string(xml_string)
    return model
