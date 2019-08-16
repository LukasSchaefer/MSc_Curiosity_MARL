import os

import numpy as np

import torch

from ..marl_algorithm import MarlAlgorithm
from ..marl_utils import soft_update
from .agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MSELoss = torch.nn.MSELoss()


class IQL(MarlAlgorithm):
    """
    (Deep) Independent Q-Learning (IQL) class

    Original IQL paper:
    Tan, M. (1993).
    Multi-agent reinforcement learning: Independent vs. cooperative agents.
    In Proceedings of the tenth international conference on machine learning (pp. 330-337).

    Link: http://web.mit.edu/16.412j/www/html/Advanced%20lectures/2004/Multi-AgentReinforcementLearningIndependentVersusCooperativeAgents.pdf

    Deep Q-Learning (DQN) paper:
    Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Petersen, S. (2015).
    Human-level control through deep reinforcement learning.
    Nature, 518(7540), 529.

    Link: https://www.nature.com/articles/nature14236?wm=book_wap_0005
    """

    def __init__(self, n_agents, observation_sizes, action_sizes, discrete_actions, params):
        """
        Initialise parameters for IQL training
        :param n_agents: number of agents
        :param observation_sizes: dimension of observation for each agent
        :param action_sizes: dimension of action for each agent
        :param discrete_actions: flag if actions are discrete
        :param params: parsed arglist parameter list
        """
        super(IQL, self).__init__(
            n_agents, observation_sizes, action_sizes, discrete_actions, params
        )

        self.discrete_actions = True

        self.model_dev = "cpu"  # device for model
        self.trgt_model_dev = "cpu"  # device for target model

        self.agents = [
            Agent(observation_sizes[i], action_sizes[i], params, discrete_actions)
            for i in range(n_agents)
        ]

    def reset(self, episode):
        """
        Reset algorithm for new episode
        :param episode: new episode number
        """
        self.prep_rollouts(device="cpu")

    def prep_rollouts(self, device="cpu"):
        """
        Prepare networks for rollout steps and use given device
        :param device: device to cast networks to
        """
        for a in self.agents:
            a.model.eval()
        if device == "gpu":
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.model_dev == device:
            for a in self.agents:
                a.model = fn(a.model)
            self.model_dev = device

    def step(self, observations, explore=False, available_actions=None):
        """
        Take a step forward in environment with all agents
        :param observations: list of observations for each agent
        :param explore: flag whether or not to add exploration noise
        :param available_actions: binary vector (n_agents, n_actions) where each list contains
                                  binary values indicating whether action is applicable
        :return: list of actions for each agent
        """
        if available_actions is None:
            return [a.step(obs, explore)[0] for a, obs in zip(self.agents, observations)]
        else:
            return [
                a.step(obs, explore, available_actions[i])[0]
                for i, (a, obs) in enumerate(zip(self.agents, observations))
            ]
        self.t_steps += 1

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_model, a.model, self.params.tau)

    def prep_training(self, device="gpu"):
        """
        Prepare networks for training and use given device
        :param device: device to cast networks to
        """
        for a in self.agents:
            a.model.train()
            a.target_model.train()
        if device == "gpu":
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.model_dev == device:
            for a in self.agents:
                a.model = fn(a.model)
            self.model_dev = device
        if not self.trgt_model_dev == device:
            for a in self.agents:
                a.target_model = fn(a.target_model)
            self.trgt_model_dev = device

    def update_agent(self, sample, agent_i, use_cuda):
        """
        Update parameters of agent model based on sample from replay buffer
        :param sample: tuple of (observations, actions, rewards, next
                        observations, and episode end masks) sampled randomly from
                        the replay buffer. Each is a list with entries
                        corresponding to each agent
        :param agent_i: index of agent to update
        :param use_cuda: flag if cuda/ gpus should be used
        :return: q loss, intrinsic loss
        """
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]

        curr_agent.optimizer.zero_grad()

        # get Q-targets for next states
        q_next_states = curr_agent.target_model(next_obs[agent_i])
        target_next_states = q_next_states.max(-1)[0]

        # compute intrinsic rewards + optimise curiosity model
        if self.curiosity is not None:
            train = True
            if not self.joint_curiosity:
                intrinsic_reward = curr_agent.curiosity_model.compute_intrinsic_reward(
                    obs[agent_i], acs[agent_i], next_obs[agent_i], use_cuda, train
                )
                # ALTERNATIVE: Use actor actions to also optimise actor with intrinsic reward
                # intrinsic_reward = self.curiosity_model.compute_intrinsic_reward(
                #     own_states, action, own_next_states
                # )
                intrinsic_losses = curr_agent.curiosity_model.get_losses()
            else:
                intrinsic_reward = self.curiosity_model.compute_intrinsic_reward(
                    torch.cat(obs, dim=1),
                    torch.cat(acs, dim=1),
                    torch.cat(next_obs, dim=1),
                    use_cuda,
                    train,
                )
                # ALTERNATIVE: Use actor actions to also optimise actor with intrinsic reward
                # intrinsic_reward = self.curiosity_model.compute_intrinsic_reward(
                #     own_states, action, own_next_states
                # )
                intrinsic_losses = self.curiosity_model.get_losses()
        else:
            intrinsic_losses = []
            intrinsic_reward = torch.zeros(self.params.batch_size, 1)
            if use_cuda:
                intrinsic_reward = intrinsic_reward.to("cuda:0")

        # compute Q-targets for current states
        target_states = (
            rews[agent_i].view(-1, 1)
            + intrinsic_reward.view(-1, 1)
            + self.gamma * target_next_states.view(-1, 1) * (1 - dones[agent_i].view(-1, 1))
        )

        # local Q-values
        all_q_states = curr_agent.model(obs[agent_i])
        q_states = torch.sum(all_q_states * acs[agent_i], dim=1).view(-1, 1)
        qloss = MSELoss(q_states, target_states.detach())
        qloss.backward()
        torch.nn.utils.clip_grad_norm_(curr_agent.model.parameters(), 0.5)
        curr_agent.optimizer.step()

        return qloss, intrinsic_losses

    def update(self, memory, use_cuda=False):
        """
        Train agent models based on memory samples
        :param memory: replay buffer memory to sample experience from
        :param use_cuda: flag if cuda/ gpus should be used
        :return: qnetwork losses, intrinsic losses
        """
        q_losses = []
        i_losses = []
        if use_cuda:
            self.prep_training(device="gpu")
        else:
            self.prep_training(device="cpu")
        for a_i in range(self.n_agents):
            sample = memory.sample(self.params.batch_size, to_gpu=use_cuda)
            q_loss, i_loss = self.update_agent(sample, a_i, use_cuda)
            q_losses.append(q_loss)
            i_losses.append(i_loss)
        self.update_all_targets()
        self.prep_rollouts(device="cpu")

        return q_losses, i_losses

    def load_model_networks(self, directory, extension="_final"):
        """
        Load model networks of all agents
        :param directory: path to directory where to load models from
        """
        for i, agent in enumerate(self.agents):
            name = "iql_agent%d_params" % i
            name += extension
            agent.model.load_state_dict(
                torch.load(os.path.join(directory, name), map_location=device)
            )
