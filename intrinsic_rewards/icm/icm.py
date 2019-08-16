import random

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import sys

from ..intrinsic_reward import IntrinsicReward
from .model import ICMNetwork


class ICM(IntrinsicReward):
    """
    Intrinsic curiosity module (ICM) class

    Paper:
    Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017).
    Curiosity-driven exploration by self-supervised prediction.
    In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (pp. 16-17).

    Link: http://openaccess.thecvf.com/content_cvpr_2017_workshops/w5/html/Pathak_Curiosity-Driven_Exploration_by_CVPR_2017_paper.html
    """

    def __init__(
        self,
        state_size,
        action_size,
        hidden_dim=128,
        state_rep_size=64,
        learning_rate=1e-5,
        eta=2,
        discrete_actions=False,
    ):
        """
        Initialise parameters for MARL training
        :param state_size: dimension of state input
        :param action_size: dimension of action input
        :param hidden_dim: hidden dimension of networks
        :param state_rep_size: dimension of state representation in network
        :param learning_rate: learning rate for ICM parameter optimisation
        :param eta: curiosity loss weighting factor
        :param discrete_actions: flag if discrete actions are used (one-hot encoded)
        """
        super(ICM, self).__init__(state_size, action_size, eta)
        self.hidden_dim = hidden_dim
        self.state_rep_size = state_rep_size
        self.learning_rate = learning_rate
        self.discrete_actions = discrete_actions

        self.model_dev = "cpu"

        self.model = ICMNetwork(
            state_size, action_size, hidden_dim, state_rep_size, discrete_actions
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.forward_loss = None
        self.inverse_loss = None

    def _prediction(self, state, action, next_state, use_cuda):
        """
        Compute prediction
        :param state: (batch of) current state(s)
        :param action: (batch of) applied action(s)
        :param next_state: (batch of) next/reached state(s)
        :param use_cuda: use CUDA tensors
        :return: (batch of) forward loss, inverse_loss
        """
        if use_cuda:
            fn = lambda x: x.cuda()
            device = "gpu"
        else:
            fn = lambda x: x.cpu()
            device = "cpu"
        if not self.model_dev == device:
            self.model = fn(self.model)
            self.model_dev = device

        predicted_action, predicted_next_state_rep, next_state_rep = self.model(
            state, next_state, action
        )
        if self.discrete_actions and predicted_action.shape == action.shape:
            # discrete one-hot encoded action
            action_targets = action.max(1)[1]
            inverse_loss = F.cross_entropy(predicted_action, action_targets, reduction="none")
        else:
            inverse_loss = ((predicted_action - action) ** 2).sum(-1)
        forward_loss = 0.5 * ((next_state_rep - predicted_next_state_rep) ** 2).sum(-1)
        return forward_loss.mean(-1), inverse_loss.mean(-1)

    def compute_intrinsic_reward(self, state, action, next_state, use_cuda, train=False):
        """
        Compute intrinsic reward for given input
        :param state: (batch of) current state(s)
        :param action: (batch of) applied action(s)
        :param next_state: (batch of) next/reached state(s)
        :param use_cuda: use CUDA tensors
        :param train: flag if model should be trained
        :return: (batch of) intrinsic reward(s)
        """
        forward_loss, inverse_loss = self._prediction(state, action, next_state, use_cuda)

        self.forward_loss = forward_loss
        self.inverse_loss = inverse_loss

        if train:
            self.optimizer.zero_grad()
            loss = forward_loss + inverse_loss
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

        return self.eta * forward_loss

    def get_losses(self):
        """
        Get losses of last computation if existing
        :return: list of (batch of) loss(es)
        """
        if self.forward_loss is not None:
            return [self.forward_loss, self.inverse_loss]
        else:
            return []
