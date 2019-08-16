import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys

sys.path.append(".")

from ..intrinsic_reward import IntrinsicReward
from .model import RNDNetwork


class RND(IntrinsicReward):
    """
    Random Network Distillation (RND) class

    Paper:
    Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2018).
    Exploration by random network distillation.
    arXiv preprint arXiv:1810.12894.

    Link: https://arxiv.org/abs/1810.12894
    """

    def __init__(
        self, state_size, action_size, hidden_dim=128, state_rep_size=64, learning_rate=1e-5, eta=2
    ):
        """
        Initialise parameters for MARL training
        :param state_size: dimension of state input
        :param action_size: dimension of action input
        :param hidden_dim: hidden dimension of networks
        :param state_rep_size: dimension of state representation in network
        :param learning_rate: learning rate for ICM parameter optimisation
        :param eta: curiosity loss weighting factor
        """
        super(RND, self).__init__(state_size, action_size, eta)
        self.hidden_dim = hidden_dim
        self.state_rep_size = state_rep_size
        self.learning_rate = learning_rate

        self.predictor_dev = "cpu"
        self.target_dev = "cpu"

        # create models
        self.predictor_model = RNDNetwork(state_size, action_size, hidden_dim, state_rep_size)
        self.target_model = RNDNetwork(state_size, action_size, hidden_dim, state_rep_size)

        for param in self.target_model.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.predictor_model.parameters(), lr=learning_rate)
        self.loss = None

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
        if use_cuda:
            fn = lambda x: x.cuda()
            device = "gpu"
        else:
            fn = lambda x: x.cpu()
            device = "cpu"
        if not self.predictor_dev == device:
            self.predictor_model = fn(self.predictor_model)
            self.predictor_dev = device
        if not self.target_dev == device:
            self.target_model = fn(self.target_model)
            self.target_dev = device

        target_feature = self.target_model(next_state)
        predict_feature = self.predictor_model(next_state)

        forward_loss = ((target_feature - predict_feature) ** 2).sum(-1).mean()
        self.loss = forward_loss

        if train:
            self.optimizer.zero_grad()
            self.loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.predictor_model.parameters(), 0.5)
            self.optimizer.step()

        return self.eta * forward_loss

    def get_losses(self):
        """
        Get losses of last computation if existing
        :return: list of (batch of) loss(es)
        """
        if self.loss is not None:
            return [self.loss]
        else:
            return []
