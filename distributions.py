import torch
from ray.rllib.models.torch.torch_action_dist import TorchBeta, TorchDistributionWrapper


class MyBetaDist(TorchBeta, TorchDistributionWrapper):

    def __init__(self, inputs, model):
        """inputs should be positive."""
        TorchDistributionWrapper.__init__(self, inputs, model)
        # Stabilize input parameters (possibly coming from a linear layer).
        self.inputs = torch.clamp(self.inputs, -100, 100)
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
