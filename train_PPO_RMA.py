from ray.rllib.algorithms.ppo import PPOConfig
from environments.BaseDroneEnv import BaseDroneEnv, base_config, distance_time_energy_reward
from models.PPO.RMA.RMA_model import RMA_model
from copy import copy
from training import train
import os
import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from environments.transformation import mujoco_quat2DCM, mujoco_rpy2quat
from gymnasium.spaces import Box
from ray.rllib.models.torch.torch_action_dist import TorchBeta
import torch


def custom_reward(env, state, action, num_steps):
    # penalize distance weighted by time steps and action magnitude
    ref = env.reference
    heading_err = np.linalg.norm(state[5] - ref[3])
    heading_err = ((heading_err + np.pi) % (2 * np.pi) - np.pi)**2
    tilt_mag = (np.array(state[3:5]) ** 2).sum()
    pos_err = ((state[:3] - ref[:3]) ** 2).sum()
    ctrl_effort = (np.array(action) ** 2).sum()
    rot_energy = (np.array(state[6:9])**2).sum()
    trans_energy = (np.array(state[3:6])**2).sum()
    too_far = pos_err > env.max_distance**2 - 1
    # reward = 1 - (1 + num_steps/100)*(pos_err + 0*heading_err) - 0*ctrl_effort - 100*too_far*(pos_err-env.max_distance**2)
    # reward += 10*(pos_err < 0.1)*(0.1 - pos_err)
    # reward = 3 - 2*pos_err - 0.2*heading_err - 0.5*rot_energy - 0.2*trans_energy - 0.2*ctrl_effort - 50*too_far*(pos_err - env.max_distance**2 + 1)**2
    # reward += -0.2*tilt_mag
    reward = (3 -2*pos_err -heading_err - 0.05*ctrl_effort)/10
    return reward


class MyCallbacks(DefaultCallbacks):
    def on_learn_on_batch(self, *, policy, train_batch, result, **kwargs):
        obs = train_batch['obs'].cpu().numpy()
        ob_mins = np.min(obs, axis=0)
        ob_maxes = np.max(obs, axis=0)
        ob_means = np.mean(obs, axis=0)
        ob_vars = np.var(obs, axis=0)
        for i in range(len(ob_mins)):  # add custom metrics to log on train
            result['min_obs%d' % i] = ob_mins[i]
            result['max_obs%d' % i] = ob_maxes[i]
            result['mean_obs%d' % i] = ob_means[i]
            result['var_obs%d' % i] = ob_vars[i]

        actions = train_batch['actions'].cpu().numpy()
        act_mins = np.min(actions, axis=0)
        act_maxes = np.max(actions, axis=0)
        act_means = np.mean(actions, axis=0)
        act_vars = np.var(actions, axis=0)
        for i in range(len(act_mins)):
            result['min_act%d' % i] = act_mins[i]
            result['max_act%d' % i] = act_maxes[i]
            result['mean_act%d' % i] = act_means[i]
            result['var_act%d' % i] = act_vars[i]


class CustomObsDroneEnv(BaseDroneEnv):
    """apply custom post-processing to the drone state before passing it to the policy model"""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_states = 12
        self.num_params = 0
        num_obs = self.num_states + self.num_params
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float64)

    def _get_obs(self):
        drone_states = super()._get_obs()
        out_obs = []
        for state in drone_states:
            xyz = state[:3]
            rpy = state[3:6]
            yaw = rpy[2]
            vel = state[6:9]
            ang_vel = state[9:12]
            acc = state[12:15]
            ref = state[15:19]
            params = state[19:]
            heading_diff = np.array((self.reference[3] - yaw + np.pi) % (2 * np.pi) - np.pi)[None]  # yaw signed difference
            DCM = mujoco_quat2DCM(mujoco_rpy2quat(np.append(rpy[:2], 0)))  # compute DCM from roll, pitch, 0
            z_vec = DCM[:, 2]  # this is the direction of the z axis of the drone in the global frame rotated by yaw
            glob_ref_err = np.array(self.reference[:3] - xyz)[None].T
            R = mujoco_quat2DCM(mujoco_rpy2quat(rpy)).T
            loc_ref_err = R @ glob_ref_err  # reference direction in local frame
            glob_vel = np.array(vel)[None].T
            loc_vel = R @ glob_vel  # velocity in local frame
            glob_ang_vel = np.array(ang_vel)[None].T
            loc_ang_vel = R @ glob_ang_vel
            # obs_i = np.concatenate([loc_ref_err.squeeze(), z_vec, heading_diff, loc_vel.squeeze(), loc_ang_vel.squeeze(), params[:3]])
            # obs_i = np.concatenate([loc_ref_err.squeeze(), rpy[:2], heading_diff, loc_vel.squeeze(), loc_ang_vel.squeeze(), params[:3]])
            obs_i = np.concatenate([loc_ref_err.squeeze(), rpy[:2], heading_diff, loc_vel.squeeze(), loc_ang_vel.squeeze()])
            out_obs.append(obs_i)
        return out_obs


class MyBetaDist(TorchBeta):

    def __init__(self, inputs, model):
        """inputs should be positive."""
        super().__init__(inputs, model, low=0, high=1)

    def deterministic_sample(self):
        self.last_sample = self._squash(self.dist.mean)
        return self.last_sample

    def entropy(self):
        return super().entropy().sum(-1)

    def kl(self, other):
        return super().kl(other).sum(-1)


ModelCatalog.register_custom_action_dist("my_beta_dist", MyBetaDist)

# checkpoint settings
model_dir = 'models/PPO/RMA/'
checkpoint_to_load = 'checkpoints/checkpoint_000150'
load_checkpoint = False

# training configuration
num_epochs = 500
train_vis = 1  # visualize one training environment
train_drones = 64  # number of drones per env
num_processes = 8  # number parallel envs used for training
rollout_length = 64  # length of individual episodes used in training
train_batch_size = num_processes * train_drones * rollout_length  # total length of the training data batch

# environment configuration
eval_env_config = copy(base_config)
eval_env_config['window_title'] = 'evaluation'
eval_env_config['num_drones'] = 1
eval_env_config['controlled'] = True
eval_env_config['max_distance'] = 3
eval_env_config['reward_fcn'] = custom_reward

train_env_config = copy(base_config)
train_env_config['reward_fcn'] = custom_reward
train_env_config['num_drones'] = train_drones  # set number of drones used per environment for training in parallel
train_env_config['window_title'] = 'training'
train_env_config['regen_env_at_steps'] = 512  # regenerate simulation after 2000 timesteps
train_env_config['train_vis'] = train_vis

# model configuration
ModelCatalog.register_custom_model("RMA_model", RMA_model)
model_config = {
    "custom_model": "RMA_model",
    "custom_model_config": {'num_states': 12,
                            'num_params': 0,
                            'num_actions': 4,
                            'param_embed_dim': 12},
    "custom_action_dist": "my_beta_dist"
}

# PPO configuration
algo_config = PPOConfig() \
    .training(gamma=0.95, lr=0.001, sgd_minibatch_size=train_batch_size // 4,
              train_batch_size=train_batch_size, model=model_config, num_sgd_iter=20) \
    .resources(num_gpus=1) \
    .rollouts(num_rollout_workers=num_processes, rollout_fragment_length=rollout_length)\
    .framework(framework='torch') \
    .environment(env=CustomObsDroneEnv, env_config=train_env_config, normalize_actions=False)\
    .exploration(explore=True, exploration_config={"type": "StochasticSampling", "random_timesteps": 0})\
    .evaluation(evaluation_duration='auto', evaluation_interval=1, evaluation_parallel_to_training=True,
                evaluation_config={'env_config': eval_env_config, 'explore': False}, evaluation_num_workers=1)\
    .debugging(seed=42)\
    .callbacks(MyCallbacks)


if __name__ == '__main__':
    algo = algo_config.build()
    if load_checkpoint and os.path.exists(model_dir + checkpoint_to_load):
        algo.restore(model_dir + checkpoint_to_load)
        print('checkpoint from {} loaded'.format(model_dir + checkpoint_to_load))

    # eval_env = VecDrone(eval_env_config)  # create an environment for evaluation
    train(algo, num_epochs, model_dir)
    algo.stop()