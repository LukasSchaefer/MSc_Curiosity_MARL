import sys
import random
import numpy as np

import torch
from torch.autograd import Variable
from torch.optim import Adam

from .model import CriticNetwork, ActorNetwork
from ..marl_utils import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise

sys.path.append("...")

from intrinsic_rewards.icm.icm import ICM
from intrinsic_rewards.rnd.rnd import RND
from intrinsic_rewards.count_based_bonus.hashing_bonus import HashingBonus


class Agent(object):
    """
    Class for MADDPG agents (policy, critic, target policy, target
    critic, exploration noise)

    based on https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/agents.py
    """

    def __init__(
        self, observation_size, action_size, num_in_critic, params, discrete_actions=True
    ):
        """
        :param observation_size: dimensions of observations
        :param action_size: dimensions of actions
        :param num_in_critic: number of dimensions for critic input
        :param params: parsed arglist parameter list
        :param discrete_actions: flag whether action space is discrete
        """
        self.observation_size = observation_size
        self.action_size = action_size
        self.num_in_critic = num_in_critic

        self.actor = ActorNetwork(
            observation_size, action_size, params.hidden_dim, params.dropout_p, discrete_actions
        )

        if params.seed is not None:
            random.seed(params.seed)
            np.random.seed(params.seed)
            torch.manual_seed(params.seed)
            torch.cuda.manual_seed(params.seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        self.critic = CriticNetwork(num_in_critic, params.hidden_dim, params.dropout_p)

        if params.seed is not None:
            random.seed(params.seed)
            np.random.seed(params.seed)
            torch.manual_seed(params.seed)
            torch.cuda.manual_seed(params.seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        self.target_actor = ActorNetwork(
            observation_size, action_size, params.hidden_dim, params.dropout_p, discrete_actions
        )

        if params.seed is not None:
            random.seed(params.seed)
            np.random.seed(params.seed)
            torch.manual_seed(params.seed)
            torch.cuda.manual_seed(params.seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        self.target_critic = CriticNetwork(num_in_critic, params.hidden_dim, params.dropout_p)

        if params.seed is not None:
            random.seed(params.seed)
            np.random.seed(params.seed)
            torch.manual_seed(params.seed)
            torch.cuda.manual_seed(params.seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.actor.parameters(), lr=params.lr)

        if params.seed is not None:
            random.seed(params.seed)
            np.random.seed(params.seed)
            torch.manual_seed(params.seed)
            torch.cuda.manual_seed(params.seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        self.critic_optimizer = Adam(self.critic.parameters(), lr=params.lr)

        if params.seed is not None:
            random.seed(params.seed)
            np.random.seed(params.seed)
            torch.manual_seed(params.seed)
            torch.cuda.manual_seed(params.seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        if not discrete_actions:
            self.exploration = OUNoise(action_size)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_actions = discrete_actions

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

        if params.seed is not None:
            random.seed(params.seed)
            np.random.seed(params.seed)
            torch.manual_seed(params.seed)
            torch.cuda.manual_seed(params.seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        self.t_step = 0

    def reset_noise(self):
        if not self.discrete_actions:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_actions:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def get_exploration_scale(self):
        if self.discrete_actions:
            return self.exploration
        else:
            return self.exploration.scale

    def step(self, obs, explore=False, available_actions=None):
        """
        Take a step forward in environment for a minibatch of observations
        :param obs (PyTorch Variable): Observations for this agent
        :param explore (boolean): Whether or not to add exploration noise
        :param available_actions: binary vector (n_agents, n_actions) where each list contains
                                binary values indicating whether action is applicable
        :return: action (PyTorch Variable) Actions for this agent
        """
        action = self.actor(obs)
        self.t_step += 1

        if available_actions is not None:
            assert self.discrete_actions
            available_mask = torch.ByteTensor(list(map(lambda a: a == 1, available_actions)))
            negative_tensor = torch.ones(action.shape) * -1e9
            negative_tensor[:, available_mask] = action[:, available_mask]
            action = negative_tensor
        if self.discrete_actions:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(torch.Tensor(self.exploration.noise()), requires_grad=False)
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {
            "policy": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_policy": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_params(self, params):
        self.actor.load_state_dict(params["policy"])
        self.critic.load_state_dict(params["critic"])
        self.target_actor.load_state_dict(params["target_policy"])
        self.target_critic.load_state_dict(params["target_critic"])
        self.policy_optimizer.load_state_dict(params["policy_optimizer"])
        self.critic_optimizer.load_state_dict(params["critic_optimizer"])
