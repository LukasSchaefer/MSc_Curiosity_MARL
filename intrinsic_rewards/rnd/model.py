import torch
import torch.nn as nn


class RNDNetwork(nn.Module):
    """Random Network Distillation (RND) network"""

    def __init__(self, state_size, action_size, hidden_dim=128, state_rep_size=64):
        """
        Initialize parameters and build model.
        :param state_size: dimension of each state
        :param action_size: dimension of each action
        :param hidden_dim: hidden dimension of networks
        :param state_rep_size: dimension of internal state feature representation
        """
        super(RNDNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.state_rep_size = state_rep_size

        # state representation
        self.state_rep = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_rep_size),
        )

    def forward(self, state):
        """
        Compute forward pass over RND network
        :param state: state
        :return: state representation
        """
        return self.state_rep(state)
