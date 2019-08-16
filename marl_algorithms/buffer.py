import numpy as np
from torch import Tensor
from torch.autograd import Variable


class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel env support

    taken from https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/buffer.py
    """

    def __init__(self, max_steps, num_agents, obs_dims, ac_dims, no_rewards=False):
        """
        Create buffer
        :param max_steps (int): maximum number of timepoints to store in buffer
        :param num_agents (int): number of agents in environment
        :param obs_dims (list of ints): number of obervation dimensions for each
                                        agent
        :param ac_dims (list of ints): number of action dimensions for each agent
        :param no_rewards (bool): flag if all rewards are 0 --> no normalisation!
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.no_rewards = no_rewards
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        for odim, adim in zip(obs_dims, ac_dims):
            self.obs_buffs.append(np.zeros((max_steps, odim)))
            self.ac_buffs.append(np.zeros((max_steps, adim)))
            self.rew_buffs.append(np.zeros(max_steps))
            self.next_obs_buffs.append(np.zeros((max_steps, odim)))
            self.done_buffs.append(np.zeros(max_steps))

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    def push(self, observations, actions, rewards, next_observations, dones):
        """
        Add entry to buffer
        :param observations: current observations
        :param actions: applied actions
        :param rewards: received rewards
        :param next_observations: observations of reached states
        :param dones: terminal flags
        """
        nentries = 1
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i  # num of indices to roll over
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i], rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i], rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i], rollover)
                self.next_obs_buffs[agent_i] = np.roll(
                    self.next_obs_buffs[agent_i], rollover, axis=0
                )
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i], rollover)
            self.curr_i = 0
            self.filled_i = self.max_steps
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][self.curr_i : self.curr_i + nentries] = np.vstack(
                observations[agent_i]
            )
            # actions are already batched by agent, so they are indexed differently
            self.ac_buffs[agent_i][self.curr_i : self.curr_i + nentries] = actions[agent_i]
            self.rew_buffs[agent_i][self.curr_i : self.curr_i + nentries] = rewards[agent_i]
            self.next_obs_buffs[agent_i][self.curr_i : self.curr_i + nentries] = np.vstack(
                next_observations[agent_i]
            )
            self.done_buffs[agent_i][self.curr_i : self.curr_i + nentries] = dones[agent_i]
        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, N, to_gpu=False, norm_rews=True):
        """
        Sample replay experience tuples (obs, actions, rewards, next_obs, dones)
        :param N: number of samples to generate
        :param to_gpu: flag whether tensors should be cast for GPU support
        :param norm_rews: flag whether rewards should be normalised
        """
        inds = np.random.choice(np.arange(self.filled_i), size=N, replace=False)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if not self.no_rewards and norm_rews:
            ret_rews = [
                cast(
                    (self.rew_buffs[i][inds] - self.rew_buffs[i][: self.filled_i].mean())
                    / self.rew_buffs[i][: self.filled_i].std()
                )
                for i in range(self.num_agents)
            ]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        return (
            [cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
            [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
            ret_rews,
            [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
            [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)],
        )
