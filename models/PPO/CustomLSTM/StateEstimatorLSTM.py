from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
import gymnasium
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.policy.rnn_sequencing import add_time_dimension

import torch
from torch import nn


class LSTMestimator(RecurrentNetwork, nn.Module):

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
        self.num_actions = custom_config['num_actions']
        self.use_estimate = custom_config['use_estimate']
        self.train_estimator = custom_config['train_estimator']

        input_size = 2*15 + 4  # o_{t-1}, o_t, a_{t-1} without pendulum info

        self.estimation_module = LSTMestimatorModule2(input_size)

        self._hidden = nn.Sequential(
            SlimFC(in_size=23, out_size=256, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=256, out_size=128, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
        )

        self._logits = nn.Sequential(
            SlimFC(in_size=128, out_size=num_outputs, initializer=nn.init.xavier_normal_, activation_fn=None),
        )

        self._value_branch = nn.Sequential(
            SlimFC(in_size=128, out_size=128, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=128, out_size=128, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=128, out_size=1, initializer=normc_initializer(0.01), activation_fn=None),
        )

        self.view_requirements = {
            "obs": ViewRequirement(shift=0, data_col='obs', space=self.obs_space),
            "prev_o": ViewRequirement(shift="-1:0", data_col='obs', space=self.obs_space),
            "prev_a": ViewRequirement(shift=-1, data_col='actions', space=self.action_space)
        }

        self._features = None
        self.regression_loss = nn.MSELoss()
        self.el = 0

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

        is_training = is_training or self.training

        self.time_major = self.model_config.get("_time_major", False)

        # print(input_dict["obs"].size(), seq_lens)
        obs = input_dict["prev_o"][:, :, :15]
        inputs = torch.cat([obs.flatten(1), input_dict["prev_a"]], dim=-1)

        inputs = add_time_dimension(
            inputs,
            seq_lens=seq_lens,
            framework="torch",
            time_major=self.time_major,
        )

        self.gt_pendulum_states = add_time_dimension(
            input_dict["prev_o"][:, 1, 15:],
            seq_lens=seq_lens,
            framework="torch",
            time_major=self.time_major,
        )

        output, new_state = self.forward_rnn(inputs, state, seq_lens, is_training)
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state

    def forward_rnn(self, inputs, state, seq_lens, is_training):
        self._hidden.train(mode=is_training)
        self.estimation_module.train(mode=is_training)

        if self.train_estimator:
            self.pendulum_state_estimates, state_out = self.estimation_module(inputs, state)
        else:
            with torch.no_grad():
                self.pendulum_state_estimates, state_out = self.estimation_module(inputs, state)

        if self.use_estimate:
            obs = torch.cat((inputs[:, :, -19:], self.pendulum_state_estimates), dim=-1)  # complete states
        else:
            obs = torch.cat((inputs[:, :, -19:], self.gt_pendulum_states), dim=-1)  # complete states

        self._features = self._hidden(obs)
        logits = self._logits(self._features)

        return logits, state_out

    def value_function(self):
        return self._value_branch(self._features).reshape([-1])

    def get_initial_state(self):
        init_state = [
            torch.zeros(self.estimation_module.lstm_hidden_size),
            torch.zeros(self.estimation_module.lstm_hidden_size),
        ]
        return init_state

    def custom_loss(self, policy_loss, loss_inputs):
        # weight decay loss - gets added to ppo loss
        wd = 1e-5
        if self.train_estimator:  # train estimator only
            l = self.regression_loss(self.pendulum_state_estimates, self.gt_pendulum_states)
            for p in self.estimation_module.parameters():
                l += wd * torch.norm(p) ** 2
            self.el = l.cpu().detach().numpy()
            return [l]
        for p in self.parameters():
            policy_loss[0] += wd*torch.norm(p)**2
        return policy_loss

    def metrics(self):
        if self.train_estimator:
            return {
                "estimation_loss": self.el,
            }


class LSTMestimatorModule(nn.Module):

    def __init__(self, input_size):
        nn.Module.__init__(self)

        self.lstm_hidden_size = 16

        self.MLP1 = nn.Sequential(
            SlimFC(in_size=input_size, out_size=self.lstm_hidden_size, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
        )
        # self.bn = nn.BatchNorm1d(self.lstm_hidden_size)
        self.LSTM = nn.LSTM(self.lstm_hidden_size, self.lstm_hidden_size, batch_first=True)
        self.MLP2 = nn.Sequential(
            SlimFC(in_size=self.lstm_hidden_size, out_size=4, initializer=nn.init.xavier_normal_, activation_fn=None),
        )

    def forward(self, inputs, state):
        y = self.MLP1(inputs)
        # y = self.bn(y.transpose(1, 2)).transpose(1, 2)  # apply batch norm
        f, [h, c] = self.LSTM(y, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)])
        estimates = self.MLP2(f + y)
        return estimates, [h.squeeze(0), c.squeeze(0)]


class LSTMestimatorModule2(nn.Module):

    def __init__(self, input_size):
        nn.Module.__init__(self)

        self.lstm_hidden_size = 32

        self.MLP1 = nn.Sequential(
            SlimFC(in_size=input_size, out_size=32, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=32, out_size=self.lstm_hidden_size, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
        )
        # self.bn = nn.BatchNorm1d(self.lstm_hidden_size)
        self.LSTM = nn.LSTM(self.lstm_hidden_size, self.lstm_hidden_size, batch_first=True)
        self.MLP2 = nn.Sequential(
            SlimFC(in_size=self.lstm_hidden_size, out_size=32, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=32, out_size=4, initializer=nn.init.xavier_normal_, activation_fn=None),
        )

    def forward(self, inputs, state):
        y = self.MLP1(inputs)
        # y = self.bn(y.transpose(1, 2)).transpose(1, 2)  # apply batch norm
        f, [h, c] = self.LSTM(y, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)])
        estimates = self.MLP2(f + y)
        return estimates, [h.squeeze(0), c.squeeze(0)]


class CNNestimator(TorchModelV2, nn.Module):
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
        self.num_actions = custom_config['num_actions']
        self.use_estimate = custom_config['use_estimate']
        self.train_estimator = custom_config['train_estimator']
        self.seq_len = model_config['max_seq_len']
        input_size = self.num_states + self.num_actions

        self.estimation_module = TimeCNN(input_size - 4, 4, self.seq_len)

        self._hidden = nn.Sequential(
            SlimFC(in_size=input_size, out_size=256, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=256, out_size=128, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
        )

        self._logits = nn.Sequential(
            SlimFC(in_size=128, out_size=num_outputs, initializer=nn.init.xavier_normal_, activation_fn=None),
        )

        self._value_branch = nn.Sequential(
            SlimFC(in_size=128, out_size=128, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=128, out_size=128, initializer=nn.init.xavier_normal_, activation_fn='tanh'),
            SlimFC(in_size=128, out_size=1, initializer=normc_initializer(0.01), activation_fn=None),
        )

        if (self.train_estimator or self.use_estimate) and self.seq_len > 1:  # if we are training the adaptation module, we need a history of actions
            self.view_requirements["obs_history"] = ViewRequirement(shift="%d:0" % (-self.seq_len + 1), data_col='obs', space=self.obs_space, batch_repeat_value=1)
            self.view_requirements["action_history"] = ViewRequirement(shift="%d:-1" % (-self.seq_len), data_col='actions', space=self.action_space, batch_repeat_value=1)
        else:
            self.view_requirements["obs_history"] = ViewRequirement(shift=0, data_col='obs', space=self.obs_space)
            self.view_requirements["action_history"] = ViewRequirement(shift=-1, data_col='actions', space=self.action_space)

        self._features = None
        self.regression_loss = nn.MSELoss()
        self.el = 0

    def forward(self, input_dict, state, seq_lens):
        obs_history = input_dict["obs_history"].float()
        action_history = input_dict["action_history"].float()
        if isinstance(input_dict, SampleBatch):
            is_training = bool(input_dict.is_training)
        else:
            is_training = bool(input_dict.get("is_training", False))

        self._hidden.train(mode=is_training)
        self.estimation_module.train(mode=is_training)

        if (self.train_estimator or self.use_estimate) and self.seq_len > 1:
            o_a_history = torch.cat((obs_history[:, :, :self.num_states-4], action_history), dim=-1)  # concatenate obs and action histories
            self.gt_pendulum_state = obs_history[:, -1, self.num_states-4:]
            flat_in = o_a_history[:, -1]  # take current state and last actions only
        else:
            o_a_history = torch.cat((obs_history[:, :self.num_states-4], action_history), dim=-1)  # concatenate obs and action histories
            self.gt_pendulum_state = obs_history[:, self.num_states-4:]
            flat_in = o_a_history  # take current state and last actions only

        if self.train_estimator:
            self.pendulum_state_estimate = self.estimation_module(o_a_history)
        elif self.use_estimate:
            with torch.no_grad():
                self.pendulum_state_estimate = self.estimation_module(o_a_history)

        if self.use_estimate:
            obs = torch.cat((flat_in, self.pendulum_state_estimate), dim=-1)  # complete states
        else:
            obs = torch.cat((flat_in, self.gt_pendulum_state), dim=-1)  # complete states

        self._features = self._hidden(obs)
        logits = self._logits(self._features)
        return logits, state

    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if self.train_estimator:  # if we are training adaptation, keep the values, but we do not need the gradients
            with torch.no_grad():
                return self._value_branch(self._features).squeeze(1)
        return self._value_branch(self._features).squeeze(1)

    def custom_loss(self, policy_loss, loss_inputs):
        # weight decay loss - gets added to ppo loss
        wd = 1e-5
        if self.train_estimator:  # train estimator only
            l = self.regression_loss(self.pendulum_state_estimate, self.gt_pendulum_state)
            for p in self.estimation_module.parameters():
                l += wd * torch.norm(p) ** 2
            self.el = l.cpu().detach().numpy()
            return [l]
        for p in self.parameters():
            policy_loss[0] += wd*torch.norm(p)**2
        return policy_loss

    def metrics(self):
        if self.train_estimator:
            return {
                "estimation_loss": self.el,
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

