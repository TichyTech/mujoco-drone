import torch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import numpy as np
from ray.tune.logger import UnifiedLogger


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

    def on_train_result(self, *, algorithm: "Algorithm", result: dict, **kwargs, ) -> None:
        """Called at the end of Algorithm.train().

        Args:
            algorithm: Current Algorithm instance.
            result: Dict of results returned from Algorithm.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """

        policy = algorithm.get_policy('default_policy')

        for i, (n, w) in enumerate(policy.model.named_parameters()):
            result['weights_norm_l%d_%s' % (i, n)] = torch.norm(w).item()
            if w.grad is not None:
                result['grad_norm_l%d_%s' % (i, n)] = torch.norm(w.grad).item()
            else:
                result['grad_norm_l%d_%s' % (i, n)] = 0

        # z = policy.model.z.cpu().detach().numpy()
        # for i in range(policy.model.param_embed_dim):
        #     result['z_var_%d' % i] = np.var(z[:, i])
        #     result['z_mean_%d' % i] = np.mean(z[:, i])


def custom_logger_creator(logdir):
    def logger(config):
        return UnifiedLogger(config, logdir, loggers=None)
    return logger
