import gym
import env as envs
import numpy as np

camera_width, camera_hight = 240, 240
num_drones = 64

env = gym.make("Drones", num_drones=num_drones, render_mode="human")
observation, info = env.reset(seed=42)

for _ in range(2000):
    action = np.ones((4*num_drones, ))*0.75
    observation, reward, terminated, truncated, info = env.step(action)
    # print(observation)
    if terminated or truncated:
        observation, info = env.reset()
env.close()