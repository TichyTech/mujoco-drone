import logging
import gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement

torch, nn = try_import_torch()
logger = logging.getLogger(__name__)


class RMA_model(TorchModelV2, nn.Module):
    """Generic fully connected network."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.param_encoder = nn.Sequential(
            SlimFC(in_size=6, out_size=64, initializer=normc_initializer(1), activation_fn='tanh'),
            SlimFC(in_size=64, out_size=64, initializer=normc_initializer(1), activation_fn='tanh'),
            SlimFC(in_size=64, out_size=12, initializer=normc_initializer(1), activation_fn='tanh'),
        )

        self._hidden_layers = nn.Sequential(
            SlimFC(in_size=28, out_size=256, initializer=normc_initializer(1), activation_fn='tanh'),
            SlimFC(in_size=256, out_size=256, initializer=normc_initializer(1), activation_fn='tanh'),
        )

        self._logits = SlimFC(in_size=256, out_size=num_outputs, initializer=normc_initializer(0.01), activation_fn=None)

        self._value_branch = SlimFC(in_size=256, out_size=1, initializer=normc_initializer(0.01), activation_fn=None)

        self.view_requirements = {
            "obs": ViewRequirement(shift=0, space=self.obs_space),
            "prev_actions": ViewRequirement(shift=-1, data_col='actions', space=self.action_space)
        }

        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        # are we training the net?
        if isinstance(input_dict, SampleBatch):
            is_training = bool(input_dict.is_training)
        else:
            is_training = bool(input_dict.get("is_training", False))
        # separate inputs
        obs = input_dict["obs"].float()
        prev_actions = input_dict["prev_actions"].float()
        obs = obs.reshape(obs.shape[0], -1)
        prev_actions = prev_actions.reshape(prev_actions.shape[0], -1)
        flat_in = torch.cat((obs[:, :12], prev_actions), dim=-1)
        drone_params = obs[:, 12:]
        # set network training mode
        self.param_encoder.train(mode=is_training)
        self._hidden_layers.train(mode=is_training)
        # forward pass
        z = self.param_encoder(drone_params)
        self._features = self._hidden_layers(torch.cat((flat_in, z), dim=-1))
        logits = self._logits(self._features)
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        return self._value_branch(self._features).squeeze(1)