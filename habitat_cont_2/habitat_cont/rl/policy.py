#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

import torch
import torch.nn as nn

from habitat_cont_2.habitat_cont.rl.ddppo_utils import CategoricalNet, DualCategoricalNet, GaussianNet, MultiGaussianNet, BetaNet, Flatten
from habitat_cont_2.habitat_cont.rl.rnn_state_encoder import RNNStateEncoder
from habitat_cont_2.habitat_cont.rl.simple_cnn import SimpleCNN


class Policy(nn.Module):
    def __init__(self, net, dim_actions, action_distribution='categorical'):
        super().__init__()
        self.net = net

        if action_distribution == 'categorical':
            self.action_distribution = CategoricalNet(
                self.net.output_size, dim_actions
            )
        elif action_distribution == 'dual_categorical':
            self.action_distribution = DualCategoricalNet(
                self.net.output_size, dim_actions, dim_actions*2-1
            )
        elif action_distribution == 'beta':
            self.action_distribution = BetaNet(
                self.net.output_size, dim_actions
            )
        elif action_distribution == 'gaussian':
            self.action_distribution = GaussianNet(
                self.net.output_size, dim_actions
            )
        elif action_distribution == 'multi_gaussian':
            self.action_distribution = MultiGaussianNet(
                self.net.output_size, dim_actions
            )
        else:
            raise ValueError("Invalid action distribution specified")

        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class PointNavBaselinePolicy(Policy):
    def __init__(self, observation_space, action_space, hidden_size=512):
        super().__init__(
            PointNavBaselineNet(
                observation_space=observation_space, hidden_size=hidden_size
            ),
            action_space.n,
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class PointNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size):
        super().__init__()

        self._n_input_goal = observation_space.spaces["pointgoal_with_gps_compass"
        ].shape[0]

        self._hidden_size = hidden_size

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self._hidden_size) + self._n_input_goal,
            self._hidden_size,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        target_encoding = observations["pointgoal_with_gps_compass"
        ]

        x = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            x = [perception_embed] + x

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states
