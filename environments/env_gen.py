import numpy as np
from matplotlib.colors import hsv_to_rgb
from dm_control import mjcf
from dm_control import mujoco
from scipy.spatial.transform import Rotation as R


def make_drone(id=0, hue=1, params=None):
    rgba = [0, 0, 0, 1]
    rgba[:3] = hsv_to_rgb([hue, 1, 1])
    rgba_transparent = [0, 0, 0, 0.6]
    rgba_transparent[:3] = rgba[:3]

    model = mjcf.RootElement(model='drone_%d' % id)

    model.default.geom.type = 'box'
    model.default.geom.rgba = rgba
    model.default.geom.conaffinity = 0
    model.default.geom.contype = 1
    model.default.geom.condim = 3
    model.default.geom.friction = (1, 0.5, 0.5)
    model.default.geom.margin = 0

    model.default.joint.damping = 0.1
    model.default.joint.armature = 0.0

    mass = params.get('mass', 0.9)
    motor_tau = params.get('motor_tau', 0.015)  # motor acts as a low pass filter with crossover frequency 1/motor_tau
    motor_force = params.get('motor_force', 7)
    arm_len = params.get('arm_len', 0.15)
    pendulum = params.get('pendulum', False)
    pendulum_length = params.get('pendulum_len', 0.2)
    weight_mass = params.get('weight_mass', 0.2)
    pole_mass = 0.2*pendulum_length

    half_body_size = 0.05

    # hopefully, this is a reasonable mass distribution
    body_mass = 0.56*mass
    arm_mass = 0.07*mass
    motor_mass = 0.04*mass

    core = model.worldbody.add('body', name='core_body_%d' % id, pos=(0, 0, 0))
    core.add('geom', name='core_geom_%d' % id, size=(half_body_size, half_body_size, half_body_size/3), mass=body_mass, rgba=rgba)
    core.add('geom', pos=[half_body_size + half_body_size/3, 0, 0], name='front_%d' % id, size=(half_body_size/3, 0.15*half_body_size, 0.15*half_body_size), mass=0, rgba=rgba)
    core.add('site', name='sense', pos=(0, 0, -half_body_size/4))
    model.sensor.add('accelerometer', name='acc_%d' % id, site='sense')
    # model.sensor.add('gyro', name='gyro_%d' % id, site='sense')
    # model.sensor.add('magnetometer', name='mag_%d' % id, site='sense')
    # core.add('camera', name='dronecam_%d' % id, pos=(half_body_size/2, 0, half_body_size/4), euler=[0, -np.pi/2, 0])
    for i in range(4):
        theta = i * np.pi / 2 + np.pi/4
        arm_pos = (np.sqrt(2)*half_body_size + 0.5*arm_len) * np.array([np.cos(theta), np.sin(theta), 0])
        rot_pos = (np.sqrt(2)*half_body_size + arm_len) * np.array([np.cos(theta), np.sin(theta), 0])
        prop_radius = arm_len/1.5
        core.add('geom', name='arm_%d' % i, size=(arm_len/2, arm_len/20, arm_len/20), pos=arm_pos, euler=[0, 0, theta], mass=arm_mass, rgba=[0.3, 0.3, 0.3, 1])
        core.add('site', name='motorsite_%d' % i, type='cylinder', pos=rot_pos, size=(0.015, arm_len/20), rgba=[0.3, 0.3, 0.3, 3])
        core.add('geom', name='motor_%d' % i, type='cylinder', pos=rot_pos + np.array([0, 0, 0.015]), size=(0.01, 0.01), rgba=[0.1, 0.1, 0.1, 1], mass=motor_mass)
        core.add('geom', name='prop_%d' % i, type='cylinder', pos=rot_pos + np.array([0, 0, 0.025]), size=(prop_radius, 0.0025), mass=0, rgba=rgba_transparent)
        model.actuator.add('general', name='motor_%d' % i, site='motorsite_%d' % i, gear=(0, 0, motor_force, 0, 0, motor_force/1000*(-1)**i),
                           ctrllimited=True, ctrlrange=(0, 1), dyntype='filter',
                           dynprm=[motor_tau, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    if pendulum:
        pend = core.add('body', name='pendulum', pos=(0, 0, -half_body_size/3))
        pend.add('joint', type='ball', pos=(0, 0, 0))
        pend.add('geom', size=(0.005, pendulum_length), pos=(0, 0, -pendulum_length), type='cylinder', mass=pole_mass, rgba=[0.3, 0.3, 0.3, 1])
        pend.add('geom', pos=(0, 0, -2*pendulum_length), type='box', size=[0.1*np.cbrt(weight_mass), 0.1*np.cbrt(weight_mass), 0.1*np.cbrt(weight_mass)], mass=weight_mass)
    return model


def make_arena(drone_params=[None], reference=None, frequency=1000):
    arena = mjcf.RootElement(model='arena')

    arena.size.nconmax = 1000  # set maximum number of collisions
    arena.size.njmax = 2000  # increase the maximum number of constraints

    arena.option.timestep = 1/frequency  # timestep for physics
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
    arena.worldbody.add('geom', name='floor', type='plane', size=[20, 20, 0.1], material=grid)
    arena.worldbody.add('light', directional='true', name='light', pos=[0, 0, 10], dir=[0, 0, -1.3])

    #  visualize origin
    # arrow_sz = [0.005, 0.5]
    # sphere_sz = 0.1
    # arrow_dist = sphere_sz + arrow_sz[1] / 2
    # x_pos = arrow_dist * np.array([1, 0, 0])
    # y_pos = arrow_dist * np.array([0, 1, 0])
    # origin_body = arena.worldbody.add('body', name='origin', pos=[0, 0, 0], mocap='true')
    # origin_body.add('geom', type='sphere', size=[sphere_sz], rgba=(1, 1, 1, 1), contype=1, conaffinity=0)
    # origin_body.add('geom', pos=x_pos, euler=[np.pi / 2, 0 - np.pi / 2, 0], size=arrow_sz,
    #              type='cylinder', rgba=(1, 0, 0, 1), contype=1, conaffinity=0)
    # origin_body.add('geom', pos=y_pos, euler=[np.pi / 2, 0, 0], size=arrow_sz,
    #              type='cylinder', rgba=(0, 1, 0, 1), contype=1, conaffinity=0)
    # origin_body.add('geom', pos=[0, 0, arrow_dist], size=arrow_sz,
    #              type='cylinder', rgba=(0, 0, 1, 1), contype=1, conaffinity=0)

    if reference is not None:  # draw reference
        arrow_sz = [0.005, 0.15]
        sphere_sz = 0.05
        ref_yaw = reference[3]
        arrow_dist = sphere_sz + arrow_sz[1]/2
        ref_body = arena.worldbody.add('body', name='reference', pos=reference[:3], euler=[0, 0, ref_yaw], mocap='true')
        ref_body.add('geom', type='sphere', size=[sphere_sz], rgba=(1, 1, 1, 1), contype=1, conaffinity=0)
        ref_body.add('geom', pos=[arrow_dist, 0, 0], euler=[np.pi/2, - np.pi/2, 0], size=arrow_sz,
                            type='cylinder', rgba=(1, 0, 0, 1), contype=1, conaffinity=0)
        ref_body.add('geom', pos=[0, arrow_dist, 0], euler=[np.pi/2, 0, 0], size=arrow_sz,
                     type='cylinder', rgba=(0, 1, 0, 1), contype=1, conaffinity=0)
        ref_body.add('geom', pos=[0, 0, arrow_dist], size=arrow_sz,
                     type='cylinder', rgba=(0, 0, 1, 1), contype=1, conaffinity=0)

    num_drones = len(drone_params)
    drones = [make_drone(i, i/num_drones, drone_params[i]) for i in range(num_drones)]
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
