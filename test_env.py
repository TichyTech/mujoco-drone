from environments.SimpleDrone import SimpleDrone
import numpy as np

camera_width, camera_hight = 240, 240
num_drones = 1

env = SimpleDrone(num_drones=num_drones, render_mode="human")
observation = env.reset()

for _ in range(200):
    action = np.ones((4*num_drones, ))*0.7
    observation, reward, terminated, info = env.step(action)
    env.render()
env.close()