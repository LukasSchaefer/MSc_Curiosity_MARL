import random
import numpy as np
import torch


class IntrinsicReward:
    """
    Abstract class for intrinsic rewards as exploration bonuses
    """

    def __init__(self, state_size, action_size, eta=2, discrete_actions=False):
        """
        Initialise parameters for MARL training
        :param state_size: dimension of state input
        :param action_size: dimension of action input
        :param eta: curiosity loss weighting factor
        :param discrete_actions: flag if discrete actions are used (one-hot encoded)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.eta = eta
        self.discrete_actions = discrete_actions

    def compute_intrinsic_reward(self, state, action, next_state, use_cuda, train=False):
        """
        Compute intrinsic reward for given input
        :param state: (batch of) current state(s)
        :param action: (batch of) applied action(s)
        :param next_state: (batch of) next/reached state(s)
        :param use_cuda: flag if CUDA tensors should be used
        :param train: flag if model should be trained
        :return: (batch of) intrinsic reward(s)
        """
        raise NotImplementedError

    def get_losses(self):
        """
        Get losses of last computation if existing
        :return: list of (batch of) loss(es)
        """
        return []
