import sys

import numpy as np
import torch

sys.path.append(".")

from ..intrinsic_reward import IntrinsicReward


class HashingBonus(IntrinsicReward):
    """
    Hash-based count bonus for exploration class

    Paper:
    Tang, H., Houthooft, R., Foote, D., Stooke, A., Chen, O. X., Duan, Y., ... & Abbeel, P. (2017).
    # Exploration: A study of count-based exploration for deep reinforcement learning.
    In Advances in neural information processing systems (pp. 2753-2762).

    Link: http://papers.nips.cc/paper/6868-exploration-a-study-of-count-based-exploration-for-deep-reinforcement-learning
    Open-source: https://github.com/openai/EPG/blob/master/epg/exploration.py
    """

    def __init__(
        self,
        state_size,
        action_size,
        batch_size,
        eta=2,
        dim_key=128,
        decay_factor=1.0,
        decay_steps=5000,
        bucket_sizes=None,
    ):
        """
        Initialise parameters for MARL training
        :param state_size: dimension of state input
        :param action_size: dimension of action input
        :param batch_size: dimension of observation batches
        :param eta: curiosity loss weighting factor
        :param dim_key: dimensonality of keys to use
        :param bucket_sizes: use specific bucket sizes instead of predefined ones
        """
        super(HashingBonus, self).__init__(state_size, action_size, eta)
        self.batch_size = batch_size
        self.dim_key = dim_key
        self.decay_factor = decay_factor
        self.decay_interest = 0.0
        self.decay_steps = decay_steps
        # Hashing function: SimHash
        if bucket_sizes is None:
            # Large prime numbers
            bucket_sizes = [999931, 999953, 999959, 999961, 999979, 999983]
        mods_list = []
        for bucket_size in bucket_sizes:
            mod = 1
            mods = []
            for _ in range(dim_key):
                mods.append(mod)
                mod = (mod * 2) % bucket_size
            mods_list.append(mods)
        self.bucket_sizes = np.asarray(bucket_sizes)
        self.mods_list = np.asarray(mods_list).T
        self.tables = np.zeros((len(bucket_sizes), np.max(bucket_sizes)))
        self.projection_matrix = np.random.normal(size=(state_size, dim_key))
        self.last_reward = None

    def compute_keys(self, obss):
        binaries = np.sign(np.asarray(obss).dot(self.projection_matrix))
        keys = np.cast["int"](binaries.dot(self.mods_list)) % self.bucket_sizes
        return keys

    def inc_hash(self, obss):
        keys = self.compute_keys(obss)
        for idx in range(len(self.bucket_sizes)):
            np.add.at(self.tables[idx], keys[:, idx], 1)

    def query_hash(self, obss):
        keys = self.compute_keys(obss)
        all_counts = []
        for idx in range(len(self.bucket_sizes)):
            all_counts.append(self.tables[idx, keys[:, idx]])
        return np.asarray(all_counts).min(axis=0)

    def fit_before_process_samples(self, obs):
        if len(obs.shape) == 1:
            obss = [obs]
        else:
            obss = obs
        before_counts = self.query_hash(obss)
        self.inc_hash(obss)

    def predict(self, obs):
        counts = self.query_hash(obs)
        prediction = 1.0 / np.maximum(1.0, self.decay_factor * np.sqrt(counts))
        return prediction

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
            state = state.cpu()
        if train:
            self.fit_before_process_samples(state)
        reward = torch.from_numpy(self.predict(state)).float()
        if use_cuda:
            reward = reward.to("cuda:0")
        self.last_reward = reward.mean(-1)
        return self.eta * reward

    def get_losses(self):
        """
        Get losses of last computation if existing
        :return: list of (batch of) loss(es)
        """
        if self.last_reward is not None:
            return [self.last_reward]
        else:
            return []
