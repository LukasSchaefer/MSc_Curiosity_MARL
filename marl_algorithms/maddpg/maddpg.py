import os
import sys

import torch
import torch.nn.functional as F

from ..marl_algorithm import MarlAlgorithm
from .agent import Agent

from ..marl_utils import soft_update, onehot_from_logits, gumbel_softmax

MSELoss = torch.nn.MSELoss()


class MADDPG(MarlAlgorithm):
    """
    Class for Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

    Paper:
    Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, O. P., & Mordatch, I. (2017).
    Multi-agent actor-critic for mixed cooperative-competitive environments.
    In Advances in Neural Information Processing Systems (pp. 6379-6390).

    Link: http://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments
    Open-source Tensorflow implementation: https://github.com/openai/maddpg

    This implementation is based on Shariq Iqbal Pytorch implementation: https://github.com/shariqiqbal2810/maddpg-pytorch
    """

    def __init__(self, n_agents, observation_sizes, action_sizes, discrete_actions, params):
        """
        Create MADDPG algorithm instance
        :param n_agents: number of agents
        :param observation_sizes: dimension of observation for each agent
        :param action_sizes: dimension of action for each agent
        :param discrete_actions: flag if actions are discrete
        :param params: parsed arglist parameter list
        """
        super(MADDPG, self).__init__(
            n_agents, observation_sizes, action_sizes, discrete_actions, params
        )

        self.pol_dev = "cpu"  # device for policies
        self.critic_dev = "cpu"  # device for critics
        self.trgt_pol_dev = "cpu"  # device for target policies
        self.trgt_critic_dev = "cpu"  # device for target critics

        self.agents = [
            Agent(
                observation_sizes[i],
                action_sizes[i],
                sum(observation_sizes) + sum(action_sizes),
                params,
                discrete_actions,
            )
            for i in range(n_agents)
        ]

    @property
    def policies(self):
        """
        Get policies
        :return: list of agent policy networks
        """
        return [a.actor for a in self.agents]

    @property
    def target_policies(self):
        """
        Get target policies
        :return: list of agent policy networks
        """
        return [a.target_actor for a in self.agents]

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        :param scale: scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        """
        Reset agent exploration noise
        """
        for a in self.agents:
            a.reset_noise()

    def reset(self, episode):
        """
        Reset for new episode
        :param episode: new episode number
        """
        self.prep_rollouts(device="cpu")
        explr_pct_remaining = (
            max(0, self.params.n_exploration_eps - episode) / self.params.n_exploration_eps
        )
        self.scale_noise(
            self.params.final_noise_scale
            + (self.params.init_noise_scale - self.params.final_noise_scale) * explr_pct_remaining
        )
        self.reset_noise()

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

    def update_agent(self, sample, agent_i, use_cuda):
        """
        Update parameters of agent model based on sample from replay buffer
        :param sample: tuple of (observations, actions, rewards, next
                        observations, and episode end masks) sampled randomly from
                        the replay buffer. Each is a list with entries
                        corresponding to each agent
        :param agent_i: index of agent to update
        :param use_cuda: flag if cuda/ gpus should be used
        :return: losses (critic loss, actor loss, intrinsic loss)
        """
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]

        curr_agent.critic_optimizer.zero_grad()
        if self.discrete_actions:  # one-hot encode action
            all_trgt_acs = [
                onehot_from_logits(pi(nobs)) for pi, nobs in zip(self.target_policies, next_obs)
            ]
        else:
            all_trgt_acs = [pi(nobs) for pi, nobs in zip(self.target_policies, next_obs)]
        trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)

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

        target_value = (
            rews[agent_i].view(-1, 1)
            + intrinsic_reward.view(-1, 1)
            + self.gamma * curr_agent.target_critic(trgt_vf_in) * (1 - dones[agent_i].view(-1, 1))
        )

        vf_in = torch.cat((*obs, *acs), dim=1)

        actual_value = curr_agent.critic(vf_in)
        vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss.backward()
        torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        curr_agent.policy_optimizer.zero_grad()

        if self.discrete_actions:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = curr_agent.actor(obs[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = curr_agent.actor(obs[agent_i])
            curr_pol_vf_in = curr_pol_out

        all_pol_acs = []
        for i, pi, ob in zip(range(self.n_agents), self.policies, obs):
            if i == agent_i:
                all_pol_acs.append(curr_pol_vf_in)
            elif self.discrete_actions:
                all_pol_acs.append(onehot_from_logits(pi(ob)))
            else:
                all_pol_acs.append(pi(ob))
        vf_in = torch.cat((*obs, *all_pol_acs), dim=1)

        pol_loss = -curr_agent.critic(vf_in).mean()
        pol_loss += (curr_pol_out ** 2).mean() * 1e-3
        pol_loss.backward()

        torch.nn.utils.clip_grad_norm_(curr_agent.actor.parameters(), 0.5)
        curr_agent.policy_optimizer.step()

        return vf_loss, pol_loss, intrinsic_losses

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_actor, a.actor, self.tau)

    def prep_training(self, device="gpu"):
        """
        Prepare networks for training and use given device
        :param device: device to cast networks to
        """
        for a in self.agents:
            a.actor.train()
            a.critic.train()
            a.target_actor.train()
            a.target_critic.train()
        if device == "gpu":
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.actor = fn(a.actor)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_actor = fn(a.target_actor)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

    def update(self, memory, use_cuda=False):
        """
        Train agent models based on memory samples
        :param memory: replay buffer memory to sample experience from
        :param use_cuda: flag if cuda/ gpus should be used
        :return: critic losses, actor losses, intrinsic losses
        """
        c_losses = []
        a_losses = []
        i_losses = []
        if use_cuda:
            self.prep_training(device="gpu")
        else:
            self.prep_training(device="cpu")
        for a_i in range(self.n_agents):
            sample = memory.sample(self.params.batch_size, to_gpu=use_cuda)
            c_loss, a_loss, i_loss = self.update_agent(sample, a_i, use_cuda)
            c_losses.append(c_loss)
            a_losses.append(a_loss)
            i_losses.append(i_loss)
        self.update_all_targets()
        self.prep_rollouts(device="cpu")

        return c_losses, a_losses, i_losses

    def prep_rollouts(self, device="cpu"):
        """
        Prepare networks for rollout steps and use given device
        :param device: device to cast networks to
        """
        for a in self.agents:
            a.actor.eval()
        if device == "gpu":
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.actor = fn(a.actor)
            self.pol_dev = device

    def load_model_networks(self, directory, extension="_final"):
        """
        Load model networks of all agents
        :param directory: path to directory where to load models from
        """
        for i, agent in enumerate(self.agents):
            name_actor = "maddpg_agent%d_actor_params" % i
            name_actor += extension
            name_critic = "maddpg_agent%d_critic_params" % i
            name_critic += extension
            agent.actor.load_state_dict(
                torch.load(os.path.join(directory, name_actor), map_location="cpu")
            )
            agent.critic.load_state_dict(
                torch.load(os.path.join(directory, name_critic), map_location="cpu")
            )
