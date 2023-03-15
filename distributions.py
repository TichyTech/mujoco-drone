import torch
from ray.rllib.models.torch.torch_action_dist import TorchBeta, TorchDistributionWrapper
import numpy as np


class MyBetaDist(TorchBeta, TorchDistributionWrapper):

    def __init__(self, inputs, model):
        """inputs should be positive."""
        TorchDistributionWrapper.__init__(self, inputs, model)
        # Stabilize input parameters (possibly coming from a linear layer).
        self.inputs = torch.clamp(self.inputs, -50, 50)
        self.inputs = torch.log(torch.exp(self.inputs) + 1.0) + 1.0
        self.low = 0
        self.high = 1
        alpha, beta = torch.chunk(self.inputs, 2, dim=-1)
        self.dist = torch.distributions.Beta(concentration1=alpha, concentration0=beta)\

    def logp(self, x):
        x = torch.clamp(x, 1e-2, 1 - 1e-2)
        logps = torch.sum(self.dist.log_prob(x), dim=-1)
        return logps

    def deterministic_sample(self):
        self.last_sample = self._squash(self.dist.mean)
        return self.last_sample

    def entropy(self):
        return super().entropy().sum(-1)

    def kl(self, other):
        return super().kl(other).sum(-1)

    def _squash(self, raw_values):
        return raw_values

    def _unsquash(self, values):
        return values


class MySquashedGaussian(TorchDistributionWrapper):
    """A tanh-squashed Gaussian distribution defined by: mean, std, low, high.

    The distribution will never return low or high exactly, but
    `low`+SMALL_NUMBER or `high`-SMALL_NUMBER respectively.
    """

    def __init__(
        self,
        inputs,
        model,
    ):
        super().__init__(inputs, model)
        mean, log_std = torch.chunk(self.inputs, 2, dim=-1)
        log_std = torch.clamp(log_std, -5, 5)
        std = torch.exp(log_std)
        self.dist = torch.distributions.normal.Normal(mean, std)
        self.low = 0
        self.high = 1
        self.mean = mean
        self.std = std

    def deterministic_sample(self):
        self.last_sample = self._squash(self.dist.mean)
        return self.last_sample

    def sample(self):
        normal_sample = self.dist.rsample()
        self.last_sample = self._squash(normal_sample)
        return self.last_sample

    def logp(self, x):
        # Unsquash values (from [low,high] to ]-inf,inf[)
        unsquashed_values = self._unsquash(x)
        # Get log prob of unsquashed values from our Normal.
        log_prob_gaussian = self.dist.log_prob(unsquashed_values)
        # For safety reasons, clamp somehow, only then sum up.
        log_prob_gaussian = torch.clamp(log_prob_gaussian, -100, 100)
        log_prob_gaussian = torch.sum(log_prob_gaussian, dim=-1)
        # Get log-prob for squashed Gaussian.
        unsquashed_values_tanhd = torch.tanh(unsquashed_values)
        log_prob = log_prob_gaussian - torch.sum(
            torch.log(1 - unsquashed_values_tanhd**2 + 1e-4), dim=-1
        )
        return log_prob

    def sample_logp(self):
        z = self.dist.rsample()
        actions = self._squash(z)
        return actions, torch.sum(
            self.dist.log_prob(z) - torch.log(1 - actions * actions + 1e-4),
            dim=-1,
        )

    def entropy(self):
        return super().entropy().sum(-1)

    def kl(self, other):
        return super().kl(other).sum(-1)

    def _squash(self, raw_values):
        # Returned values are within [low, high] (including `low` and `high`).
        squashed = torch.sigmoid(raw_values)
        return torch.clamp(squashed, self.low, self.high)

    def _unsquash(self, values):
        normed_values = values * 2.0 - 1.0
        save_normed_values = torch.clamp(normed_values, -1.0 + 1e-4, 1.0 - 1e-4)
        unsquashed = torch.atanh(save_normed_values)
        return unsquashed

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return np.prod(action_space.shape, dtype=np.int32) * 2
