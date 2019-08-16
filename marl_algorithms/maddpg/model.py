import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNetwork(nn.Module):
    """
    Deep Critic-Network applied on joint observations & actions

    based on https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/networks.py
    """

    def __init__(self, input_dim, hidden_dim=64, dropout_p=0.0, nonlin=F.relu, norm_in=True):
        """
        Initialize parameters and build model.
        :param input_dim: dimension of network input
        :param hidden_dim: dimension of hidden layers
        :param dropout_p: dropout probability
        :param nonlin: nonlinearity to use
        :param norm_in: normalise input first
        """
        super(CriticNetwork, self).__init__()
        # normalize inputs
        if norm_in:
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.nonlin = nonlin
        self.drop_layer = nn.Dropout(p=dropout_p)

    def forward(self, x):
        """
        Compute forward pass over QNetwork
        :param x: network input (concatenated observations and actions)
        :return: forward pass result
        """
        h1 = self.nonlin(self.fc1(self.in_fn(x)))
        h1 = self.drop_layer(h1)
        h2 = self.nonlin(self.fc2(h1))
        h2 = self.drop_layer(h2)
        out = self.fc3(h2)
        return out


class ActorNetwork(nn.Module):
    """
    Deep Actor/Policy-Network choosing action based on individual observation
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=64,
        dropout_p=0.0,
        discrete_actions=True,
        nonlin=F.relu,
        norm_in=True,
    ):
        """
        Initialize parameters and build model.
        :param input_dim: dimension of network input
        :param output_dim: dimension of last layer
        :param dropout_p: dropout probability
        :param discrete_actions: flag indicating if actions are discrete
        :param nonlin: nonlinearity to use
        :param norm_in: normalise input first
        """
        super(ActorNetwork, self).__init__()
        # normalize inputs
        if norm_in:
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.nonlin = nonlin
        self.drop_layer = nn.Dropout(p=dropout_p)
        if discrete_actions:
            # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x
        else:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = torch.tanh

    def forward(self, observation):
        """
        Compute forward pass over policy network
        :param observation: representation for input observations
        :return: forward pass result
        """
        h1 = self.nonlin(self.fc1(self.in_fn(observation)))
        h1 = self.drop_layer(h1)
        h2 = self.nonlin(self.fc2(h1))
        h2 = self.drop_layer(h2)
        out = self.out_fn(self.fc3(h2))
        return out
