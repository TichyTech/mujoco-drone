# About:

This is code for my [Master thesis](https://dspace.cvut.cz/handle/10467/108688) in Cybernetics and Robotics course at [Faculty of Electrical Engineering](https://fel.cvut.cz/cs) at [Czech Technical University in Prague](https://www.cvut.cz/en/).

The main goal was to teach a quadcopter with hanging load how to fly using an artificial neural network via reinforcement learning. This is all done inside a physics simulator [MuJoCo](https://mujoco.readthedocs.io/), since the training algorithm needs a lot of samples to converge, which is impractical to achieve on a real system. [Rllib](https://docs.ray.io/en/latest/rllib/index.html) is used as a training backend for efficient parallelized training of a [PyTorch](https://pytorch.org/) Neural Network policy.

The resulting neural network policy is able to stabilize the quadcopter on a reference position from any initial state.

 ![SimplePolicySimpleModel](https://github.com/user-attachments/assets/a30c0b10-486a-403f-82b0-1ea8fb5c54c3)

Or even follow a moving reference point.

 ![MLPcut](https://github.com/user-attachments/assets/a82e0308-c507-4f06-9cf9-5e546e8bb46f)

Further experiments have also dealt with robustness of the controller under changing system parameters such as weight of the quadcopter or weight of the load.

 ![Adapt](https://github.com/user-attachments/assets/2e6e5d97-2a4a-4a32-83dc-278f4a9d5126)

And finally I used an LSTM network to estimate the load state (Visualized by green ball).

 ![StateEst](https://github.com/user-attachments/assets/22845642-7fb9-4592-b4ea-61dc3d7d98d6)

## Instructions for simple mujoco environment:
 - install the modules using the following commands:
```
pip install gym[mujoco]
pip install numpy matplotlib mujoco dm_control
```
 - run ```python test_env.py```

## Instructions for vectorized rllib training:
 - ideally prepare a conda env as ```conda create -n rllib python=3.8```
 - install the modules using the following commands:
```
pip install ray[rllib]==2.1 
pip install gym[mujoco]==0.26.2 
pip install numpy matplotlib dm_control
```
 - install pytorch with gpu support: <https://pytorch.org/get-started/locally/>
 - run ```python train_PPO_MLP.py``` to start training
 - run ```tensorboard --logdir=~/ray_results``` to monitor the training progress
