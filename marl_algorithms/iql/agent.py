import sys
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F

from .model import QNetwork
from ..marl_utils import hard_update, soft_update, onehot_from_logits

sys.path.append("...")

from intrinsic_rewards.icm.icm import ICM
from intrinsic_rewards.rnd.rnd import RND
from intrinsic_rewards.count_based_bonus.hashing_bonus import HashingBonus

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Agent:
    """
    Class for individual IQL agent
    """

    def __init__(self, observation_size, action_size, params, discrete_actions=True):
        """
        Initialise parameters for agent
        :param observation_size: dimensions of observations
        :param action_size: dimensions of actions
        :param params: parsed arglist parameter list
        :param discrete_actions: flag whether action space is discrete
        """
        self.observation_size = observation_size
        self.action_size = action_size
        self.discrete_actions = discrete_actions
        self.params = params

        self.epsilon = params.exploration_bonus

        # create Q-Learning networks
        self.model = QNetwork(observation_size, action_size, params.hidden_dim, params.dropout_p)

        if params.seed is not None:
            random.seed(params.seed)
            np.random.seed(params.seed)
            torch.manual_seed(params.seed)
            torch.cuda.manual_seed(params.seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        self.target_model = QNetwork(
            observation_size, action_size, params.hidden_dim, params.dropout_p
        )

        if params.seed is not None:
            random.seed(params.seed)
            np.random.seed(params.seed)
            torch.manual_seed(params.seed)
            torch.cuda.manual_seed(params.seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        # create individual curiosity model if set
        if params.curiosity is not None and not params.joint_curiosity:
            if params.curiosity == "icm":
                self.curiosity_model = ICM(
                    observation_size,
                    action_size,
                    params.curiosity_hidden_dim,
                    params.curiosity_state_rep_size,
                    params.curiosity_lr,
                    params.eta,
                    discrete_actions,
                )
            elif params.curiosity == "rnd":
                self.curiosity_model = RND(
                    observation_size,
                    action_size,
                    params.curiosity_hidden_dim,
                    params.curiosity_state_rep_size,
                    params.curiosity_lr,
                    params.eta,
                )
            elif params.curiosity == "count":
                self.curiosity_model = HashingBonus(
                    observation_size,
                    action_size,
                    params.batch_size,
                    params.eta,
                    params.count_key_dim,
                    params.count_decay,
                )
            else:
                raise ValueError("Unknown curiosity: " + params.curiosity)

        hard_update(self.target_model, self.model)

        # create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=params.lr)

        if params.seed is not None:
            random.seed(params.seed)
            np.random.seed(params.seed)
            torch.manual_seed(params.seed)
            torch.cuda.manual_seed(params.seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        self.t_step = 0

    def step(self, obs, explore=False, available_actions=None):
        """
        Take a step forward in environment for a minibatch of observations
        :param obs (PyTorch Variable): Observations for this agent
        :param explore (boolean): Whether or not to add exploration noise
        :param available_actions: binary vector (n_agents, n_actions) where each list contains
                                binary values indicating whether action is applicable
        :return: action (PyTorch Variable) Actions for this agent
        """
        qvals = self.model(obs)
        self.t_step += 1

        if available_actions is not None:
            assert self.discrete_actions
            available_mask = torch.ByteTensor(list(map(lambda a: a == 1, available_actions)))
            negative_tensor = torch.ones(qvals.shape) * -1e9
            negative_tensor[:, available_mask] = qvals[:, available_mask]
            qvals = negative_tensor
        if explore:
            action = onehot_from_logits(qvals, self.epsilon)
        else:
            action = onehot_from_logits(qvals)

        self.epsilon *= self.params.decay_factor

        return action
