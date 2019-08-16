import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    """Deep Q-Network"""

    def __init__(
        self, state_size, action_size, hidden_dim, dropout_p=0.0, nonlin=F.relu, norm_in=True
    ):
        """
        Initialize parameters and build model.
        :param state_size: Dimension of each state
        :param action_size: Dimension of each action
        :param hidden_dim: dimension of hidden layers
        :param dropout_p: dropout probability
        :param nonlin: nonlinearity to use
        :param norm_in: normalise input first
        """
        super(QNetwork, self).__init__()
        # normalize inputs
        if norm_in:
            self.in_fn = nn.BatchNorm1d(state_size)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_size)
        self.nonlin = nonlin
        self.drop_layer = nn.Dropout(p=dropout_p)

    def forward(self, state):
        """
        Compute forward pass over QNetwork
        :param state: state representation for input state
        :return: forward pass result
        """
        x = self.nonlin(self.fc1(self.in_fn(state)))
        x = self.drop_layer(x)
        x = self.nonlin(self.fc2(x))
        x = self.drop_layer(x)
        return self.fc3(x)
