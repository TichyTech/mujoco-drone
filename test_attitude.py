from environments.BaseDroneEnv import BaseDroneEnv, base_config
from models.Analytic.AttitudeController import AttittudeController
from models.Analytic.PositionController import PositionController

import numpy as np

num_drones = 64
ref = [2, 0, 10, 0.2]
start = [0, 0, 10, 0]

config = base_config
config['reference'] = ref
config['start_pos'] = start
config['num_drones'] = num_drones
env = BaseDroneEnv(base_config)
obs, _ = env.reset()

masses = np.array([env.drone_params[i]['body_mass'] for i in range(num_drones)]) + 0.45
forces = np.array([env.drone_params[i]['motor_force'] for i in range(num_drones)])
attc = AttittudeController(num_drones, masses, forces)
posc = PositionController(num_drones)

for _ in range(1200):
    rpy = np.array(obs)[:, 3:6].T
    xyz = np.array(obs)[:, :3].T  # ref - pos
    h = rpy[2]
    print('state ', xyz, rpy)
    pos_action = posc.compute_control(np.array(ref[:3]), xyz)
    print('pos_action ', pos_action)
    rpyz = attc.tilts2rpy(pos_action, np.ones(num_drones)*ref[3])
    print('rpyz ', rpyz)
    action = attc.compute_control(rpyz, rpy)
    print(action)
    obs, reward, terminated, truncated, info = env.vector_step(action)
    env.render()
env.close()