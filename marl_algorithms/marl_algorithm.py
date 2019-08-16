import random
import numpy as np
import torch

import sys

sys.path.append("..")

from intrinsic_rewards.icm.icm import ICM
from intrinsic_rewards.rnd.rnd import RND
from intrinsic_rewards.count_based_bonus.hashing_bonus import HashingBonus


class MarlAlgorithm:
    """
    abstract class for MARL algorithm
    """

    def __init__(self, n_agents, observation_sizes, action_sizes, discrete_actions, params):
        """
        Initialise parameters for MARL training
        :param n_agents: number of agents
        :param observation_sizes: dimension of observation for each agent
        :param action_sizes: dimension of action for each agent
        :param discrete_actions: flag if actions are discrete
        :param params: parsed arglist parameter list
        """
        self.n_agents = n_agents
        self.observation_sizes = observation_sizes
        self.action_sizes = action_sizes
        self.params = params
        self.discrete_actions = discrete_actions
        self.batch_size = params.batch_size
        self.gamma = params.gamma
        self.tau = params.tau
        self.learning_rate = params.lr
        self.exploration_bonus = params.exploration_bonus
        self.decay_factor = params.decay_factor
        self.dropout_p = params.dropout_p
        self.curiosity = params.curiosity
        self.joint_curiosity = params.joint_curiosity
        self.curiosity_hidden_dim = params.curiosity_hidden_dim
        self.curiosity_state_rep_size = params.curiosity_state_rep_size
        self.count_key_dim = params.count_key_dim
        self.count_decay = params.count_decay
        self.curiosity_learning_rate = params.curiosity_lr
        self.eta = params.eta
        self.seed = params.seed

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        if self.joint_curiosity and self.curiosity is not None:
            if self.curiosity == "icm":
                self.curiosity_model = ICM(
                    sum(observation_sizes),
                    sum(action_sizes),
                    self.curiosity_hidden_dim,
                    self.curiosity_state_rep_size,
                    self.curiosity_learning_rate,
                    self.eta,
                    self.discrete_actions,
                )
            elif self.curiosity == "rnd":
                self.curiosity_model = RND(
                    sum(observation_sizes),
                    sum(action_sizes),
                    self.curiosity_hidden_dim,
                    self.curiosity_state_rep_size,
                    self.curiosity_learning_rate,
                    self.eta,
                )
            elif self.curiosity == "count":
                self.curiosity_model = HashingBonus(
                    sum(observation_sizes),
                    sum(action_sizes),
                    self.batch_size,
                    self.eta,
                    self.count_key_dim,
                    self.count_decay,
                )
            else:
                raise ValueError("Unknown curiosity: " + self.curiosity)
        else:
            self.curiosity_model = None

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        self.t_steps = 0

    def reset(self, episode):
        """
        Reset algorithm for new episode
        :param episode: new episode number
        """
        raise NotImplementedError

    def step(self, observations, explore=False, available_actions=None):
        """
        Take a step forward in environment with all agents
        :param observations: list of observations for each agent
        :param explore: flag whether or not to add exploration noise
        :param available_actions: binary vector (n_agents, n_actions) where each list contains
                                  binary values indicating whether action is applicable
        :return: list of actions for each agent
        """
        raise NotImplementedError

    def update(self, memory, use_cuda=False):
        """
        Train agent models based on memory samples
        :param memory: replay buffer memory to sample experience from
        :param use_cuda: flag if cuda/ gpus should be used
        :return: tuple of loss lists
        """
        raise NotImplementedError

    def load_model_networks(self, directory, extension="_final"):
        """
        Load model networks of all agents
        :param directory: path to directory where to load models from
        """
        raise NotImplementedError

    def get_curiosities(self, obs, act, next_obs):
        """
        Compute curiosities for all agents
        :param obs: current observation
        :param act: current action
        :param next_obs: next observation
        :return: list of curiosity for each agent
        """
        act_expanded = []
        for a in act:
            act_expanded.append(np.expand_dims(a, axis=0))
        act = act_expanded

        if self.curiosity is None:
            return [None] * self.n_agents

        if self.joint_curiosity:
            joint_obs = np.concatenate(obs, axis=1)
            joint_act = np.concatenate(act, axis=1)
            joint_next_obs = np.concatenate(next_obs, axis=1)
            curiosity = self.curiosity_model.compute_intrinsic_reward(
                torch.from_numpy(joint_obs).float(),
                torch.from_numpy(joint_act).float(),
                torch.from_numpy(joint_next_obs).float(),
                False,
            )
            return [curiosity] * self.n_agents

        curiosities = []
        for i, agent in enumerate(self.agents):
            curiosity = agent.curiosity_model.compute_intrinsic_reward(
                torch.from_numpy(obs[i]).float(),
                torch.from_numpy(act[i]).float(),
                torch.from_numpy(next_obs[i]).float(),
                False,
            )
            curiosities.append(curiosity)
        return curiosities
