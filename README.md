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
 - run ```python rllib_train.py```