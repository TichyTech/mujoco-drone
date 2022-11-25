import environments as envs
import numpy as np

camera_width, camera_hight = 240, 240
num_drones = 1

env = envs.Drones.DronesEnv(num_drones=num_drones, render_mode="human")
observation = env.reset()

for _ in range(2000):
    action = np.ones((4*num_drones, ))*0.7
    observation, reward, terminated, info = env.step(action)
    env.render()
env.close()