from environments.BaseDroneEnv import BaseDroneEnv, base_config
from models.Analytic.AttitudeController import AttittudeController
from models.Analytic.PositionController import PositionController
from environments.transformation import mujoco_quat2DCM, mujoco_rpy2quat, mujoco_DCM2quat
import numpy as np


num_drones = 1
ref = [0, 0, 10, 0]
start = [0, 0, 10, 0]

config = base_config
config['reference'] = ref
config['start_pos'] = start
config['num_drones'] = num_drones
config['pendulum'] = True
config['random_params'] = False
config['random_start_pos'] = True
config['controlled'] = False
config['mocaps'] = 3
config['state_difficulty'] = 1
config['window_title'] = 'attitude_test'
env = BaseDroneEnv(base_config)
obs, _ = env.reset()

masses = np.array([env.drone_params[i]['mass'] for i in range(num_drones)])
if config['pendulum']:
    masses = masses + [env.drone_params[i]['weight_mass'] for i in range(num_drones)]
    masses = masses + 0.2*np.array([env.drone_params[i]['pendulum_len'] for i in range(num_drones)])

forces = np.array([env.drone_params[i]['motor_force'] for i in range(num_drones)])
print(masses, forces)
attc = AttittudeController(num_drones, masses, forces)
posc = PositionController(num_drones)

for i in range(1200):
    rpy = np.array(obs)[:, 3:6].T
    xyz = np.array(obs)[:, :3].T  # ref - pos
    h = rpy[2]
    # print('state ', xyz, rpy)
    pos_action = posc.compute_control(np.array(env.reference[:3]), xyz)
    print('pos_action ', pos_action)
    rpyz = attc.tilts2rpy(pos_action, np.ones(num_drones)*env.reference[3])
    print('rpyz ', rpyz)
    action = attc.compute_control(rpyz, rpy)
    print('actions', action)
    obs, reward, terminated, truncated, info = env.vector_step(np.clip(action - 0.1, a_min=0, a_max=1))

    state = obs[0]
    params = state[-6:]
    pendulum_rp = state[12:14]
    drone_R = mujoco_quat2DCM(mujoco_rpy2quat(state[3:6]))  # drone rotation in world frame
    pendulum_R = mujoco_quat2DCM(mujoco_rpy2quat(np.append(pendulum_rp, 0)))  # pendulum rotation in drone frame

    from scipy.spatial.transform import Rotation as Rt
    quat1 = Rt.from_euler('XY', pendulum_rp).as_quat()
    mujoco_quat1 = np.append(quat1[3], quat1[:3])
    quat2 = Rt.from_euler('yx', pendulum_rp[::-1]).as_quat()
    mujoco_quat2 = np.append(quat2[3], quat2[:3])

    pendulum_R = mujoco_quat2DCM(mujoco_quat1)

    print('pendrp', pendulum_rp)
    print('pendrot', pendulum_R)

    pendulum_end = np.array([[0], [0], [-params[4]]])  # pendulum end in pendulum frame
    R = drone_R @ pendulum_R
    pendulum_pos = state[:3] + (R @ pendulum_end)[:, 0]
    env.move_mocap_to(np.concatenate((state[:3], mujoco_DCM2quat(drone_R))), 1)
    env.move_mocap_to(np.concatenate((pendulum_pos[:3], mujoco_DCM2quat(R))), 2)

    env.set_state(env.data.qpos, env.data.qvel)  # call forward
    env.render()
env.close()