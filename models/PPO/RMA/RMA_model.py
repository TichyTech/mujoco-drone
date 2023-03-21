import logging
import gymnasium

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement

torch, nn = try_import_torch()
logger = logging.getLogger(__name__)

torch.random.manual_seed(42)


class RMA_model(TorchModelV2, nn.Module):
    """Generic fully connected network."""

    def __init__(
        self,
        obs_space: gymnasium.spaces.Space,
        action_space: gymnasium.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        custom_config = model_config['custom_model_config']
        self.num_states = custom_config['num_states']
        self.num_params = custom_config['num_params']
        self.num_actions = custom_config['num_actions']
        self.param_embed_dim = custom_config['param_embed_dim']
        assert obs_space.shape[0] == self.num_states + self.num_params  # assert we are using correct environment/model
        assert action_space.shape[0] == self.num_actions

        if self.num_params > 0:
            self.param_encoder = nn.Sequential(
                SlimFC(in_size=self.num_params, out_size=32, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
                SlimFC(in_size=32, out_size=self.param_embed_dim, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            )

        if self.num_params > 0:
            hidden_in_dim = self.num_states + self.num_actions + self.param_embed_dim
        else:
            hidden_in_dim = self.num_states + self.num_actions
        self._hidden_layers = nn.Sequential(
            # nn.BatchNorm1d(hidden_in_dim),
            SlimFC(in_size=hidden_in_dim, out_size=256, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=256, out_size=128, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=128, out_size=128, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=128, out_size=96, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            nn.BatchNorm1d(96)
        )

        self._logits = nn.Sequential(
            SlimFC(in_size=96, out_size=64, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=64, out_size=64, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=64, out_size=num_outputs, initializer=nn.init.xavier_normal_, activation_fn=None)
        )

        self._value_branch = nn.Sequential(
            SlimFC(in_size=96, out_size=128, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=128, out_size=128, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=128, out_size=1, initializer=normc_initializer(0.01), activation_fn=None)
        )

        # self._value_branch = SlimFC(in_size=64, out_size=1, initializer=normc_initializer(0.01), activation_fn=None)

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
        flat_in = torch.cat((obs[:, :self.num_states], prev_actions), dim=-1)
        # flat_in = obs[:, :self.num_states]  # flattened observations
        # set network training mode
        self._hidden_layers.train(mode=is_training)
        # forward pass
        if self.num_params > 0:
            drone_params = obs[:, self.num_states:self.num_states + self.num_params]  # flattened drone parameters
            self.param_encoder.train(mode=is_training)
            z = self.param_encoder(drone_params)  # encode drone parameters
            self._features = self._hidden_layers(torch.cat((flat_in, z), dim=-1))
        else:
            self._features = self._hidden_layers(flat_in)
        logits = self._logits(self._features)
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        return self._value_branch(self._features).squeeze(1)

    def custom_loss(self, policy_loss, loss_inputs):
        # weight decay loss - gets added to ppo loss
        wd = 1e-4
        for p in self.parameters():
            policy_loss[0] += wd*torch.norm(p)**2
        return policy_loss
