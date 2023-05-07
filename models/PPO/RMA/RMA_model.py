import logging
import gymnasium

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
import torch
from torch import nn

logger = logging.getLogger(__name__)


torch.random.manual_seed(42)


class RMA_full(TorchModelV2, nn.Module):
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
        self.train_adaptation = custom_config['train_adaptation']
        self.seq_len = custom_config['adapt_seq_len']

        if self.train_adaptation and self.seq_len > 1:  # if we are training the adaptation module, we need a history of actions
            self.view_requirements["obs_history"] = ViewRequirement(shift="%d:0" % (-self.seq_len + 1), data_col='obs', space=self.obs_space, batch_repeat_value=1)
            self.view_requirements["action_history"] = ViewRequirement(shift="%d:-1" % (-self.seq_len), data_col='actions', space=self.action_space, batch_repeat_value=1)
        else:
            self.view_requirements["obs_history"] = ViewRequirement(shift=0, data_col='obs', space=self.obs_space)
            self.view_requirements["action_history"] = ViewRequirement(shift=-1, data_col='actions', space=self.action_space)

        self.param_encoder = nn.Sequential(
            SlimFC(in_size=self.num_params, out_size=32, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=32, out_size=self.param_embed_dim, initializer=nn.init.xavier_normal_, activation_fn=None),
        )

        self.adaptation_module = TimeCNN2(self.num_states + self.num_actions, self.param_embed_dim, self.seq_len)

        hidden_in_dim = self.num_states + self.num_actions + self.param_embed_dim
        self._hidden_layers = nn.Sequential(
            SlimFC(in_size=hidden_in_dim, out_size=256, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=256, out_size=128, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            nn.BatchNorm1d(128)
        )

        self._logits = nn.Sequential(
            SlimFC(in_size=128, out_size=128, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=128, out_size=num_outputs, initializer=nn.init.xavier_normal_, activation_fn=None)
        )

        self._value_branch = nn.Sequential(
            SlimFC(in_size=128, out_size=128, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=128, out_size=128, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=128, out_size=1, initializer=normc_initializer(0.01), activation_fn=None)
        )

        self.adaptation_loss = nn.MSELoss()

        self._features = None
        self.z = None
        self.z_hat = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs_history = input_dict["obs_history"].float()
        action_history = input_dict["action_history"].float()
        if isinstance(input_dict, SampleBatch):
            is_training = bool(input_dict.is_training)
        else:
            is_training = bool(input_dict.get("is_training", False))

        self._hidden_layers.train(mode=is_training)

        if self.train_adaptation and self.seq_len > 1:
            s_a_history = torch.cat((obs_history[:, :, :self.num_states], action_history), dim=-1)  # concatenate state and action histories
            e = obs_history[:, -1, -self.num_params:]  # current drone parameters
            flat_in = s_a_history[:, -1]  # take current state and last actions only
        else:
            s_a_history = torch.cat((obs_history[:, :self.num_states], action_history), dim=-1)  # concatenate state and action histories
            e = obs_history[:, -self.num_params:]  # current drone parameters
            flat_in = s_a_history

        if self.train_adaptation:
            self.z_hat = self.adaptation_module(s_a_history)
            with torch.no_grad():
                self.z = self.param_encoder(e)  # encode drone parameters
                self._features = self._hidden_layers(torch.cat((flat_in, self.z_hat), dim=-1))
                logits = self._logits(self._features)
        else:
            self.z = self.param_encoder(e)  # encode drone parameters
            self._features = self._hidden_layers(torch.cat((flat_in, self.z), dim=-1))
            logits = self._logits(self._features)
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if self.train_adaptation:  # if we are training adaptation, keep the values, but we do not need the gradients
            with torch.no_grad():
                return self._value_branch(self._features).squeeze(1)
        return self._value_branch(self._features).squeeze(1)

    def custom_loss(self, policy_loss, loss_inputs):
        # weight decay loss - gets added to ppo loss
        wd = 1e-5
        if self.train_adaptation:  # adaptation loss
            l = self.adaptation_loss(self.z_hat, self.z)
            for p in self.adaptation_module.parameters():
                l += wd * torch.norm(p) ** 2
            self.al = l.cpu().detach().numpy()
            return [l]
        for p in self.parameters():
            policy_loss[0] += wd*torch.norm(p)**2
        return policy_loss

    def metrics(self):
        if self.train_adaptation:
            return {
                "adaptation_loss": self.al,
            }


class TimeCNN(nn.Module):
    def __init__(self, in_feature_dim, param_embed_dim, seq_len=50):
        nn.Module.__init__(self)

        self.inMLP = nn.Sequential(
            SlimFC(in_size=in_feature_dim, out_size=32, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=32, out_size=32, initializer=nn.init.xavier_normal_, activation_fn='tanh')
        )

        self.tCNN = nn.Sequential(
            nn.Conv1d(32, 32, 5, 2),
            nn.Conv1d(32, 16, 5)
        )

        test_x = self.tCNN(torch.zeros((1, 32, seq_len))).flatten(1)

        self.outMLP = nn.Sequential(
            SlimFC(in_size=test_x.size(1), out_size=32, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=32, out_size=param_embed_dim, initializer=nn.init.xavier_normal_, activation_fn=None)
        )

    def forward(self, x):
        y = self.inMLP(x)
        y = self.tCNN(y.transpose(1, 2))
        y = self.outMLP(y.flatten(1))
        return y


class TimeCNN2(nn.Module):
    def __init__(self, in_feature_dim, param_embed_dim, seq_len=50):
        nn.Module.__init__(self)

        self.inMLP = nn.Sequential(
            SlimFC(in_size=in_feature_dim, out_size=32, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=32, out_size=32, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=32, out_size=32, initializer=nn.init.xavier_normal_, activation_fn='tanh')
        )

        self.tCNN = nn.Sequential(
            nn.Conv1d(32, 32, 5, 2),
            nn.Conv1d(32, 16, 5)
        )

        test_x = self.tCNN(torch.zeros((1, 32, seq_len))).flatten(1)

        self.outMLP = nn.Sequential(
            SlimFC(in_size=test_x.size(1), out_size=64, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=64, out_size=param_embed_dim, initializer=nn.init.xavier_normal_, activation_fn=None)
        )

    def forward(self, x):
        y = self.inMLP(x)
        y = self.tCNN(y.transpose(1, 2))
        y = self.outMLP(y.flatten(1))
        return y


# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


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
        self.z = None

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
        if self.num_actions > 0:
            flat_in = torch.cat((obs[:, :self.num_states], prev_actions), dim=-1)
        else:
            flat_in = obs[:, :self.num_states]
        # flat_in = obs[:, :self.num_states]  # flattened observations
        # set network training mode
        self._hidden_layers.train(mode=is_training)
        # forward pass
        if self.num_params > 0:
            drone_params = obs[:, self.num_states:self.num_states + self.num_params]  # flattened drone parameters
            self.param_encoder.train(mode=is_training)
            self.z = self.param_encoder(drone_params)  # encode drone parameters
            self._features = self._hidden_layers(torch.cat((flat_in, self.z), dim=-1))
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


class RMA_model_smaller(RMA_model):

    def __init__(self,
        obs_space: gymnasium.spaces.Space,
        action_space: gymnasium.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
                 ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

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
            SlimFC(in_size=hidden_in_dim, out_size=256, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=256, out_size=128, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            nn.BatchNorm1d(128)
        )

        self._logits = nn.Sequential(
            SlimFC(in_size=128, out_size=128, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=128, out_size=num_outputs, initializer=nn.init.xavier_normal_, activation_fn=None)
        )

        self._value_branch = nn.Sequential(
            SlimFC(in_size=128, out_size=128, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=128, out_size=128, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=128, out_size=1, initializer=normc_initializer(0.01), activation_fn=None)
        )


class ResBlock(nn.Module):
    def __init__(self, width, depth):
        nn.Module.__init__(self)
        layers = [SlimFC(in_size=width, out_size=width, initializer=nn.init.xavier_normal_, activation_fn='tanh') for _ in range(depth)]
        self.hidden = nn.Sequential(*layers)

    def forward(self, x):
        return self.hidden(x) + x


class RMA_model_smaller2(RMA_model):
    def __init__(self,
        obs_space: gymnasium.spaces.Space,
        action_space: gymnasium.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
                 ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

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
            SlimFC(in_size=hidden_in_dim, out_size=512, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=512, out_size=256, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            nn.BatchNorm1d(256)
        )

        self._logits = nn.Sequential(
            SlimFC(in_size=256, out_size=num_outputs, initializer=nn.init.xavier_normal_, activation_fn=None)
        )

        self._value_branch = nn.Sequential(
            ResBlock(256, 1),
            SlimFC(in_size=256, out_size=128, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            ResBlock(128, 2),
            SlimFC(in_size=128, out_size=1, initializer=normc_initializer(0.01), activation_fn=None)
        )


class RMA_model_smaller2(RMA_model):
    def __init__(self,
        obs_space: gymnasium.spaces.Space,
        action_space: gymnasium.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
                 ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

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
            SlimFC(in_size=hidden_in_dim, out_size=512, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=512, out_size=256, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            nn.BatchNorm1d(256)
        )

        self._logits = nn.Sequential(
            SlimFC(in_size=256, out_size=num_outputs, initializer=nn.init.xavier_normal_, activation_fn=None)
        )

        self._value_branch = nn.Sequential(
            ResBlock(256, 1),
            SlimFC(in_size=256, out_size=128, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            ResBlock(128, 2),
            SlimFC(in_size=128, out_size=1, initializer=normc_initializer(0.01), activation_fn=None)
        )


