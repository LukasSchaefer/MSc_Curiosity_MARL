import torch
import torch.nn as nn
import torch.nn.functional as F


class ICMNetwork(nn.Module):
    """Intrinsic curiosity module (ICM) network"""

    def __init__(
        self, state_size, action_size, hidden_dim=128, state_rep_size=64, discrete_actions=False
    ):
        """
        Initialize parameters and build model.
        :param state_size: dimension of each state
        :param action_size: dimension of each action
        :param hidden_dim: hidden dimension of networks
        :param state_rep_size: dimension of internal state feature representation
        :param discrete_actions: flag if discrete actions are used (one-hot encoded)
        """
        super(ICMNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.state_rep_size = state_rep_size
        self.discrete_actions = discrete_actions

        # state representation
        self.state_rep = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_rep_size),
        )

        # inverse model
        self.inverse_model = nn.Sequential(
            nn.Linear(state_rep_size * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size),
        )

        # forward model
        self.forward_model = nn.Sequential(
            nn.Linear(state_rep_size + action_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_rep_size),
        )

    def forward(self, state, next_state, action):
        """
        Compute forward pass over ICM network
        :param state: current state
        :param next_state: reached state
        :param action: applied action
        :return: predicted_action, predicted_next_state_rep, next_state_rep
        """
        # compute state representations
        state_rep = self.state_rep(state)
        next_state_rep = self.state_rep(next_state)

        # inverse model output
        inverse_input = torch.cat([state_rep, next_state_rep], 1)
        predicted_action = self.inverse_model(inverse_input)

        # forward model output
        forward_input = torch.cat([state_rep, action], 1)
        predicted_next_state_rep = self.forward_model(forward_input)

        return predicted_action, predicted_next_state_rep, next_state_rep
