import logging
import gymnasium

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.policy.rnn_sequencing import add_time_dimension

torch, nn = try_import_torch()
logger = logging.getLogger(__name__)

torch.random.manual_seed(42)


class DSN_LSTM_model(RecurrentNetwork, nn.Module):
    """Decomposed state network. First decompose state into x-y-z components (in local drone frame) and use three
    parallel networks to process the decomposed states. Then concatenate outputs and use as an input to a mixer network.
    This approach aims at exploiting the pairwise independence of x-y-z dynamics to decrease the NN complexity."""

    def __init__(
        self,
        obs_space: gymnasium.spaces.Space,
        action_space: gymnasium.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        custom_config = model_config['custom_model_config']
        self.num_states = custom_config['num_states']
        self.num_params = custom_config['num_params']
        self.num_actions = custom_config['num_actions']
        self.param_embed_dim = custom_config['param_embed_dim']
        assert obs_space.shape[0] == self.num_states + self.num_params  # assert we are using correct environment/model
        assert action_space.shape[0] == self.num_actions

        self.x_hidden = nn.Sequential(
            SlimFC(in_size=4, out_size=64, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=64, out_size=64, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=64, out_size=32, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
        )

        self.y_hidden = nn.Sequential(
            SlimFC(in_size=4, out_size=64, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=64, out_size=64, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=64, out_size=32, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
        )

        self.z_hidden = nn.Sequential(
            SlimFC(in_size=4, out_size=32, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=32, out_size=32, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=32, out_size=16, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
        )

        self.LSTM_x = nn.LSTM(32, 32, batch_first=True)
        self.LSTM_y = nn.LSTM(32, 32, batch_first=True)
        self.LSTM_z = nn.LSTM(16, 16, batch_first=True)

        self.mixer = nn.Sequential(
            SlimFC(in_size=16*5 + 4, out_size=64, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=64, out_size=num_outputs, initializer=nn.init.xavier_normal_, activation_fn=None)
        )

        self._value_branch = nn.Sequential(
            SlimFC(in_size=16*5, out_size=128, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=128, out_size=128, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=128, out_size=1, initializer=normc_initializer(0.01), activation_fn=None)
        )

        self.bn_x = nn.BatchNorm1d(32)
        self.bn_y = nn.BatchNorm1d(32)
        self.bn_z = nn.BatchNorm1d(16)

        self.view_requirements = {
            "obs": ViewRequirement(shift=0, space=self.obs_space),
            "prev_actions": ViewRequirement(shift=-1, data_col='actions', space=self.action_space)
        }

        # Holds the current "base" output (before logits layer).
        self._features = None

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ):
        if isinstance(input_dict, SampleBatch):
            is_training = bool(input_dict.is_training)
        else:
            is_training = bool(input_dict.get("is_training", False))

        self.time_major = self.model_config.get("_time_major", False)

        obs = add_time_dimension(
            input_dict["obs"],
            seq_lens=seq_lens,
            framework="torch",
            time_major=self.time_major,
        )

        actions = add_time_dimension(
            input_dict["prev_actions"],
            seq_lens=seq_lens,
            framework="torch",
            time_major=self.time_major,
        )

        output, new_state = self.forward_rnn(obs, actions, state, seq_lens, is_training)
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state

    def forward_rnn(self, obs, actions, state, seq_lens, is_training):
        self.bn_x.train(mode=is_training)
        self.bn_y.train(mode=is_training)
        self.bn_z.train(mode=is_training)
        # separate observations into xyz
        xyz_obs = obs[:, :, :12].view(obs.shape[0], obs.shape[1], 4, 3)
        x_obs = xyz_obs[:, :, :, 0]
        y_obs = xyz_obs[:, :, :, 1]
        z_obs = xyz_obs[:, :, :, 2]
        # perform forward pass on xyz
        x_f = self.bn_x(self.x_hidden(x_obs).transpose(1, 2)).transpose(1, 2)
        y_f = self.bn_y(self.y_hidden(y_obs).transpose(1, 2)).transpose(1, 2)
        z_f = self.bn_z(self.z_hidden(z_obs).transpose(1, 2)).transpose(1, 2)
        self._features = torch.cat([x_f, y_f, z_f], dim=-1)
        x_f, [h_x, c_x] = self.LSTM_x(x_f, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)])
        y_f, [h_y, c_y] = self.LSTM_y(y_f, [torch.unsqueeze(state[2], 0), torch.unsqueeze(state[3], 0)])
        z_f, [h_z, c_z] = self.LSTM_z(z_f, [torch.unsqueeze(state[4], 0), torch.unsqueeze(state[5], 0)])
        f = torch.cat([x_f, y_f, z_f], dim=-1) + self._features
        f = torch.cat([f, actions], dim=-1)
        logits = self.mixer(f)
        return logits, [h_x.squeeze(0), c_x.squeeze(0), h_y.squeeze(0), c_y.squeeze(0), h_z.squeeze(0), c_z.squeeze(0)]

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        return self._value_branch(self._features).reshape([-1])

    def get_initial_state(self):
        init_state = [
            torch.zeros(32),
            torch.zeros(32),
            torch.zeros(32),
            torch.zeros(32),
            torch.zeros(16),
            torch.zeros(16),
        ]
        return init_state
