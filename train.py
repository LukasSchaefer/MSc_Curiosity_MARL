import argparse
import time
import random

import numpy as np
import torch
from torch.autograd import Variable

from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete

from marl_algorithms.maddpg.maddpg import MADDPG
from marl_algorithms.iql.iql import IQL
from marl_algorithms.buffer import ReplayBuffer

from utilities.model_saver import ModelSaver
from utilities.logger import Logger
from utilities.plotter import Plotter
from utilities.frame_saver import FrameSaver

USE_CUDA = torch.cuda.is_available()
GOAL_EPSILON = 0.01


class Train:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            "Reinforcement Learning experiments for multiagent environments"
        )
        self.parse_args()
        self.arglist = self.parser.parse_args()

    def parse_default_args(self):
        """
        Parse default arguments for MARL training script
        """
        # algorithm
        self.parser.add_argument(
            "--alg", type=str, default="maddpg", help="name of the algorithm to use"
        )
        self.parser.add_argument("--hidden_dim", default=128, type=int)

        # curiosity
        self.parser.add_argument(
            "--curiosity", type=str, default=None, help="name of curiosity to use"
        )
        self.parser.add_argument(
            "--joint_curiosity",
            action="store_true",
            default=False,
            help="flag if curiosity should be applied jointly for all agents",
        )
        self.parser.add_argument(
            "--curiosity_hidden_dim",
            type=int,
            default=64,
            help="curiosity internal state representation size",
        )
        self.parser.add_argument(
            "--curiosity_state_rep_size",
            type=int,
            default=64,
            help="curiosity internal state representation size",
        )
        self.parser.add_argument(
            "--count_key_dim",
            type=int,
            default=32,
            help="key dimensionality of hash-count-based curiosity",
        )
        self.parser.add_argument(
            "--count_decay", type=float, default=1, help="factor for count decay speed"
        )
        self.parser.add_argument(
            "--eta", type=int, default=5, help="curiosity loss weighting factor"
        )
        self.parser.add_argument(
            "--curiosity_lr",
            type=float,
            default=5e-6,
            help="learning rate for curiosity optimizer",
        )

        # training length
        self.parser.add_argument(
            "--num_episodes", type=int, default=25000, help="number of episodes"
        )
        self.parser.add_argument(
            "--max_episode_len", type=int, default=25, help="maximum episode length"
        )

        # core training parameters
        self.parser.add_argument(
            "--n_training_threads", default=6, type=int, help="number of training threads"
        )
        self.parser.add_argument(
            "--no_rewards",
            action="store_true",
            default=False,
            help="flag if no rewards should be used",
        )
        self.parser.add_argument(
            "--sparse_rewards",
            action="store_true",
            default=False,
            help="flag if sparse rewards should be used",
        )
        self.parser.add_argument(
            "--sparse_freq", type=int, default=25, help="number of steps before sparse rewards"
        )
        self.parser.add_argument("--gamma", type=float, default=0.9, help="discount factor")
        self.parser.add_argument(
            "--tau", type=float, default=0.01, help="tau as stepsize for target network updates"
        )
        self.parser.add_argument(
            "--lr", type=float, default=0.01, help="learning rate for Adam optimizer"
        )
        self.parser.add_argument(
            "--dropout_p", type=float, default=0.0, help="Dropout probability"
        )
        self.parser.add_argument(
            "--seed", type=int, default=None, help="random seed used throughout training"
        )
        self.parser.add_argument(
            "--steps_per_update", type=int, default=100, help="number of steps before updates"
        )

        self.parser.add_argument(
            "--buffer_capacity", type=int, default=int(1e6), help="Replay buffer capacity"
        )
        self.parser.add_argument(
            "--batch_size",
            type=int,
            default=1024,
            help="number of episodes to optimize at the same time",
        )

        # exploration settings
        self.parser.add_argument(
            "--no_exploration",
            action="store_true",
            default=False,
            help="flag if no exploration should be used",
        )
        self.parser.add_argument(
            "--decay_factor", type=float, default=0.99999, help="exploration decay factor"
        )
        self.parser.add_argument(
            "--exploration_bonus", type=float, default=1.0, help="exploration bonus value"
        )
        self.parser.add_argument("--n_exploration_eps", default=25000, type=int)
        self.parser.add_argument("--init_noise_scale", default=0.3, type=float)
        self.parser.add_argument("--final_noise_scale", default=0.0, type=float)

        # visualisation
        self.parser.add_argument("--display", action="store_true", default=False)
        self.parser.add_argument("--save_frames", action="store_true", default=False)
        self.parser.add_argument(
            "--plot", action="store_true", default=False, help="plot reward and exploration bonus"
        )
        self.parser.add_argument(
            "--eval_frequency", default=100, type=int, help="frequency of evaluation episodes"
        )
        self.parser.add_argument(
            "--eval_episodes", default=5, type=int, help="number of evaluation episodes"
        )
        self.parser.add_argument(
            "--dump_losses",
            action="store_true",
            default=False,
            help="dump losses after computation",
        )

        # run name for store path
        self.parser.add_argument(
            "--run", type=str, default="default", help="run name for stored paths"
        )

        # model storing
        self.parser.add_argument(
            "--save_models_dir",
            type=str,
            default="models",
            help="path where models should be saved",
        )
        self.parser.add_argument("--save_interval", default=1000, type=int)
        self.parser.add_argument(
            "--load_models",
            type=str,
            default=None,
            help="path where models should be loaded from if set",
        )
        self.parser.add_argument(
            "--load_models_extension",
            type=str,
            default="final",
            help="name extension for models to load",
        )

    def parse_args(self):
        """
        parse own arguments
        """
        self.parse_default_args()

    def extract_sizes(self, spaces):
        """
        Extract space dimensions
        :param spaces: list of Gym spaces
        :return: list of ints with sizes for each agent
        """
        sizes = []
        for space in spaces:
            if isinstance(space, Box):
                size = sum(space.shape)
            elif isinstance(space, Dict):
                size = sum(self.extract_sizes(space.values()))
            elif isinstance(space, Discrete) or isinstance(space, MultiBinary):
                size = space.n
            elif isinstance(space, MultiDiscrete):
                size = sum(space.nvec)
            else:
                raise ValueError("Unknown class of space: ", type(space))
            sizes.append(size)
        return sizes

    def create_environment(self):
        """
        Create environment instance
        :return: environment (gym interface), env_name, task_name, n_agents, observation_sizes,
                 action_sizes, discrete_actions
        """
        raise NotImplementedError()

    def reset_environment(self):
        """
        Reset environment for new episode
        :return: observation (as torch tensor)
        """
        raise NotImplementedError

    def select_actions(self, obs, explore=True):
        """
        Select actions for agents
        :param obs: joint observation
        :param explore: flag if exploration should be used
        :return: action_tensor, action_list
        """
        raise NotImplementedError()

    def environment_step(self, actions):
        """
        Take step in the environment
        :param actions: actions to apply for each agent
        :return: reward, done, next_obs (as Pytorch tensors)
        """
        raise NotImplementedError()

    def environment_render(self):
        """
        Render visualisation of environment
        """
        raise NotImplementedError()

    def eval(self, ep, n_agents):
        """
        Execute evaluation episode without exploration
        :param ep: episode number
        :param n_agents: number of agents in task
        :return: episode_rewards, episode_length, done
        """
        obs = self.reset_environment()
        self.alg.reset(ep)

        episode_rewards = np.array([0.0] * n_agents)
        episode_length = 0
        done = False

        while not done and episode_length < self.arglist.max_episode_len:
            torch_obs = [
                Variable(torch.Tensor(obs[i]), requires_grad=False) for i in range(n_agents)
            ]

            actions, _ = self.select_actions(torch_obs, False)
            rewards, dones, next_obs = self.environment_step(actions)

            episode_rewards += rewards

            obs = next_obs
            episode_length += 1
            done = all(dones)

        return episode_rewards, episode_length, done

    def train(self):
        """
        Abstract training flow
        """
        # set random seeds before model creation
        if self.arglist.seed is not None:
            random.seed(self.arglist.seed)
            np.random.seed(self.arglist.seed)
            torch.manual_seed(self.arglist.seed)
            torch.cuda.manual_seed(self.arglist.seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        # use number of threads if no GPUs are available
        if not USE_CUDA:
            torch.set_num_threads(self.arglist.n_training_threads)

        env, env_name, task_name, n_agents, observation_sizes, action_sizes, discrete_actions = (
            self.create_environment()
        )
        self.env = env
        self.n_agents = n_agents

        steps = self.arglist.num_episodes * self.arglist.max_episode_len
        # steps-th root of GOAL_EPSILON
        decay_epsilon = GOAL_EPSILON ** (1 / float(steps))
        self.arglist.decay_factor = decay_epsilon
        print(
            "Epsilon is decaying with factor %.7f to %.3f over %d steps."
            % (decay_epsilon, GOAL_EPSILON, steps)
        )

        print("Observation sizes: ", observation_sizes)
        print("Action sizes: ", action_sizes)

        # Create curiosity instances
        if self.arglist.curiosity is None:
            print("No curiosity is to be used!")
        elif self.arglist.curiosity == "icm":
            print("Training uses Intrinsic Curiosity Module (ICM)!")
        elif self.arglist.curiosity == "rnd":
            print("Training uses Random Network Distillation (RND)!")
        elif self.arglist.curiosity == "count":
            print("Training uses hash-based counting exploration bonus!")
        else:
            raise ValueError("Unknown curiosity: " + self.arglist.curiosity)

        # create algorithm trainer
        if self.arglist.alg == "maddpg":
            self.alg = MADDPG(
                n_agents, observation_sizes, action_sizes, discrete_actions, self.arglist
            )
            print(
                "Training multi-agent deep deterministic policy gradient (MADDPG) on "
                + env_name
                + " environment"
            )
        elif self.arglist.alg == "iql":
            self.alg = IQL(
                n_agents, observation_sizes, action_sizes, discrete_actions, self.arglist
            )
            print("Training independent q-learning (IQL) on " + env_name + " environment")
        else:
            raise ValueError("Unknown algorithm: " + self.arglist.alg)

        self.memory = ReplayBuffer(
            self.arglist.buffer_capacity,
            n_agents,
            observation_sizes,
            action_sizes,
            self.arglist.no_rewards,
        )

        # set random seeds past model creation
        if self.arglist.seed is not None:
            random.seed(self.arglist.seed)
            np.random.seed(self.arglist.seed)
            torch.manual_seed(self.arglist.seed)
            torch.cuda.manual_seed(self.arglist.seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        if self.arglist.load_models is not None:
            print(
                "Loading models from "
                + self.arglist.load_models
                + " with extension "
                + self.arglist.load_models_extension
            )
            self.alg.load_model_networks(
                self.arglist.load_models, "_" + self.arglist.load_models_extension
            )

        self.model_saver = ModelSaver(
            self.arglist.save_models_dir, self.arglist.run, self.arglist.alg
        )
        self.logger = Logger(
            n_agents,
            self.arglist.eta,
            task_name,
            self.arglist.run,
            self.arglist.alg,
            self.arglist.curiosity,
        )
        self.plotter = Plotter(
            self.logger,
            n_agents,
            self.arglist.eval_frequency,
            task_name,
            self.arglist.run,
            self.arglist.alg,
            self.arglist.curiosity,
        )
        if self.arglist.save_frames:
            self.frame_saver = FrameSaver(
                self.arglist.eta, task_name, self.arglist.run, self.arglist.alg
            )

        print("Starting iterations...")
        start_time = time.time()
        t = 0

        for ep in range(self.arglist.num_episodes):
            obs = self.reset_environment()
            self.alg.reset(ep)

            episode_rewards = np.array([0.0] * n_agents)
            if self.arglist.sparse_rewards:
                sparse_rewards = np.array([0.0] * n_agents)
            episode_length = 0
            done = False
            interesting_episode = False

            while not done and episode_length < self.arglist.max_episode_len:
                torch_obs = [
                    Variable(torch.Tensor(obs[i]), requires_grad=False) for i in range(n_agents)
                ]

                actions, agent_actions = self.select_actions(
                    torch_obs, not self.arglist.no_exploration
                )
                rewards, dones, next_obs = self.environment_step(actions)

                episode_rewards += rewards
                if self.arglist.sparse_rewards:
                    sparse_rewards += rewards

                if self.arglist.no_rewards:
                    rewards = [0.0] * n_agents
                elif self.arglist.sparse_rewards:
                    if (episode_length + 1) % self.arglist.sparse_freq == 0:
                        rewards = list(sparse_rewards / self.arglist.sparse_freq)
                    else:
                        rewards = [0.0] * n_agents
                self.memory.push(obs, agent_actions, rewards, next_obs, dones)

                t += 1

                if (
                    len(self.memory) >= self.arglist.batch_size
                    and (t % self.arglist.steps_per_update) == 0
                ):
                    losses = self.alg.update(self.memory, USE_CUDA)
                    self.logger.log_losses(ep, losses)
                    if self.arglist.dump_losses:
                        self.logger.dump_losses(1)

                # for displaying learned policies
                if self.arglist.display:
                    self.environment_render()
                if self.arglist.save_frames:
                    self.frame_saver.add_frame(self.env.render("rgb_array")[0], ep)
                    if self.arglist.curiosity is not None:
                        curiosities = self.alg.get_curiosities(obs, agent_actions, next_obs)
                        interesting = self.frame_saver.save_interesting_frame(curiosities)
                        interesting_episode = interesting_episode or interesting

                obs = next_obs
                episode_length += 1
                done = all(dones)

            if ep % self.arglist.eval_frequency == 0:
                eval_rewards = np.zeros((self.arglist.eval_episodes, n_agents))
                for i in range(self.arglist.eval_episodes):
                    ep_rewards, _, _ = self.eval(ep, n_agents)
                    eval_rewards[i, :] = ep_rewards
                if self.arglist.alg == "maddpg":
                    self.logger.log_episode(
                        ep,
                        eval_rewards.mean(0),
                        eval_rewards.var(0),
                        self.alg.agents[0].get_exploration_scale(),
                    )
                if self.arglist.alg == "iql":
                    self.logger.log_episode(
                        ep, eval_rewards.mean(0), eval_rewards.var(0), self.alg.agents[0].epsilon
                    )
                self.logger.dump_episodes(1)
            if ep % 100 == 0 and ep > 0:
                duration = time.time() - start_time
                self.logger.dump_train_progress(ep, self.arglist.num_episodes, duration)

            if interesting_episode:
                self.frame_saver.save_episode_gif()

            if ep % (self.arglist.save_interval // 2) == 0 and ep > 0:
                # update plots
                self.plotter.update_reward_plot(self.arglist.plot)
                self.plotter.update_exploration_plot(self.arglist.plot)
                self.plotter.update_alg_loss_plot(self.arglist.plot)
                if self.arglist.curiosity is not None:
                    self.plotter.update_cur_loss_plot(self.arglist.plot)
                    self.plotter.update_intrinsic_reward_plot(self.arglist.plot)

            if ep % self.arglist.save_interval == 0 and ep > 0:
                # save plots
                print("Remove previous plots")
                self.plotter.clear_plots()
                print("Saving intermediate plots")
                self.plotter.save_reward_plot(str(ep))
                self.plotter.save_exploration_plot(str(ep))
                self.plotter.save_alg_loss_plots(str(ep))
                self.plotter.save_cur_loss_plots(str(ep))
                self.plotter.save_intrinsic_reward_plot(str(ep))
                # save models
                print("Remove previous models")
                self.model_saver.clear_models()
                print("Saving intermediate models")
                self.model_saver.save_models(self.alg, str(ep))
                # save logs
                print("Remove previous logs")
                self.logger.clear_logs()
                print("Saving intermediate logs")
                self.logger.save_episodes(extension=str(ep))
                self.logger.save_losses(extension=str(ep))
                # save parameter log
                self.logger.save_parameters(
                    env_name,
                    task_name,
                    n_agents,
                    observation_sizes,
                    action_sizes,
                    discrete_actions,
                    self.arglist,
                )

        duration = time.time() - start_time
        print("Overall duration: %.2fs" % duration)

        print("Remove previous plots")
        self.plotter.clear_plots()
        print("Saving final plots")
        self.plotter.save_reward_plot("final")
        self.plotter.save_exploration_plot("final")
        self.plotter.save_alg_loss_plots("final")
        self.plotter.save_cur_loss_plots("final")
        self.plotter.save_intrinsic_reward_plot("final")

        # save models
        print("Remove previous models")
        self.model_saver.clear_models()
        print("Saving final models")
        self.model_saver.save_models(self.alg, "final")

        # save logs
        print("Remove previous logs")
        self.logger.clear_logs()
        print("Saving final logs")
        self.logger.save_episodes(extension="final")
        self.logger.save_losses(extension="final")
        self.logger.save_duration_cuda(duration, torch.cuda.is_available())

        # save parameter log
        self.logger.save_parameters(
            env_name,
            task_name,
            n_agents,
            observation_sizes,
            action_sizes,
            discrete_actions,
            self.arglist,
        )

        env.close()

    if __name__ == "__main__":
        train = Train()
        train.train()
