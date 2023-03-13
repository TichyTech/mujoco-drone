from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
import gymnasium
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.policy.rnn_sequencing import add_time_dimension

torch, nn = try_import_torch()


class CustomLSTM(RecurrentNetwork, nn.Module):

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
        # self.param_embed_dim = custom_config['param_embed_dim']

        self._features = None
        input_size = self.num_states + self.num_actions

        self.MLP1 = nn.Sequential(
            SlimFC(in_size=input_size, out_size=256, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=256, out_size=128, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=128, out_size=64, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=64, out_size=64, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
        )
        self.LSTM = nn.LSTM(64, 64, batch_first=True)
        self._logits = nn.Sequential(
            SlimFC(in_size=64, out_size=64, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=64, out_size=num_outputs, initializer=nn.init.xavier_normal_, activation_fn=None),
        )
        self._value_branch = nn.Sequential(
            SlimFC(in_size=64, out_size=256, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=256, out_size=1, initializer=normc_initializer(0.01), activation_fn=None),
        )

        self.bn = nn.BatchNorm1d(64)

        self.view_requirements = {
            "obs": ViewRequirement(shift=0, space=self.obs_space),
            "prev_actions": ViewRequirement(shift=-1, data_col='actions', space=self.action_space)
        }

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
        inputs = torch.cat([input_dict["obs"], input_dict["prev_actions"]], dim=-1)

        inputs = add_time_dimension(
            inputs,
            seq_lens=seq_lens,
            framework="torch",
            time_major=self.time_major,
        )

        output, new_state = self.forward_rnn(inputs, state, seq_lens, is_training)
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state

    def forward_rnn(self, inputs, state, seq_lens, is_training):
        self.bn.train(mode=is_training)

        self._features = self.MLP1(inputs)
        self._features = self.bn(self._features.transpose(1, 2)).transpose(1, 2)  # apply batch norm
        f, [h, c] = self.LSTM(self._features, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)])
        logits = self._logits(f + self._features)
        return logits, [h.squeeze(0), c.squeeze(0)]

    def value_function(self):
        return self._value_branch(self._features).reshape([-1])

    def get_initial_state(self):
        init_state = [
            torch.zeros(64),
            torch.zeros(64),
        ]
        return init_state

    def custom_loss(self, policy_loss, loss_inputs):
        # weight decay loss - gets added to ppo loss
        wd = 1e-5
        for p in self.parameters():
            policy_loss[0] += wd*torch.norm(p)**2
        return policy_loss