import argparse
import time
import random

import numpy as np
import torch
from torch.autograd import Variable

from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete

from marl_algorithms.maddpg.maddpg import MADDPG
from marl_algorithms.iql.iql import IQL

from utilities.logger import Logger
from utilities.plotter import Plotter
from utilities.frame_saver import FrameSaver


class Eval:
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
        self.parser.add_argument(
            "--save_gifs", action="store_true", help="Save gif of episode into gifs directory"
        )
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
            default=128,
            help="key dimensionality of hash-count-based curiosity",
        )
        self.parser.add_argument(
            "--count_decay", type=float, default=0.01, help="factor for count decay speed"
        )
        self.parser.add_argument(
            "--eta", type=int, default=2, help="curiosity loss weighting factor"
        )
        self.parser.add_argument(
            "--curiosity_lr",
            type=float,
            default=1e-5,
            help="learning rate for curiosity optimizer",
        )

        self.parser.add_argument(
            "--batch_size",
            type=int,
            default=1024,
            help="number of episodes to optimize at the same time",
        )

        # training length
        self.parser.add_argument(
            "--num_episodes", type=int, default=100, help="number of episodes"
        )
        self.parser.add_argument(
            "--max_episode_len", type=int, default=25, help="maximum episode length"
        )

        # core training parameters
        self.parser.add_argument(
            "--n_training_threads", default=6, type=int, help="number of training threads"
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

        # exploration settings
        self.parser.add_argument(
            "--decay_factor", type=float, default=0.0, help="exploration decay factor"
        )
        self.parser.add_argument(
            "--exploration_bonus", type=float, default=0.0, help="eploration bonus value"
        )
        self.parser.add_argument("--n_exploration_eps", default=1, type=int)
        self.parser.add_argument("--init_noise_scale", default=0.0, type=float)
        self.parser.add_argument("--final_noise_scale", default=0.0, type=float)

        self.parser.add_argument(
            "--run", type=str, default="default", help="run name for stored paths"
        )

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

    def select_actions(self, obs):
        """
        Select actions for agents
        :param obs: joint observation
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

    def eval(self):
        """
        Abstract evaluation flow
        """
        print("EVALUATION RUN")
        print("No exploration and dropout will be used")
        self.arglist.exploration_bonus = 0.0
        self.arglist.init_noise_scale = 0.0
        self.arglist.dropout_p = 0.0
        if self.arglist.load_models is None:
            print("WARNING: Evaluation run without loading any models!")

        # set random seeds before model creation
        if self.arglist.seed is not None:
            random.seed(self.arglist.seed)
            np.random.seed(self.arglist.seed)
            torch.manual_seed(self.arglist.seed)
            torch.cuda.manual_seed(self.arglist.seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        env, env_name, task_name, n_agents, observation_sizes, action_sizes, discrete_actions = (
            self.create_environment()
        )
        self.env = env
        self.n_agents = n_agents

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
            # TODO: add count based ones
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

        self.logger = Logger(n_agents, task_name, None, self.arglist.alg, self.arglist.curiosity)
        self.plotter = Plotter(
            self.logger,
            n_agents,
            task_name,
            self.arglist.run,
            self.arglist.alg,
            self.arglist.curiosity,
        )
        if self.arglist.save_gifs:
            self.frame_saver = FrameSaver(
                self.arglist.eta, task_name, self.arglist.run, self.arglist.alg
            )

        print("Starting iterations...")
        start_time = time.time()
        t = 0
        for ep in range(self.arglist.num_episodes):
            episode_rewards = []

            obs = self.reset_environment()
            self.alg.reset(ep)

            episode_rewards = np.array([0.0] * n_agents)
            episode_length = 0
            done = False

            while not done and episode_length < self.arglist.max_episode_len:
                torch_obs = [
                    Variable(torch.Tensor(obs[i]), requires_grad=False) for i in range(n_agents)
                ]

                actions, agent_actions = self.select_actions(torch_obs)
                rewards, dones, next_obs = self.environment_step(actions)

                t += 1

                episode_rewards += rewards

                # for displaying learned policies
                self.environment_render()
                if self.arglist.save_gifs:
                    self.frame_saver.add_frame(self.env.render("rgb_array")[0], ep)

                obs = next_obs
                episode_length += 1
                done = all(dones)

            if self.arglist.alg == "maddpg":
                self.logger.log_episode(
                    ep,
                    episode_rewards,
                    [0.0] * n_agents,
                    self.alg.agents[0].get_exploration_scale(),
                )
            if self.arglist.alg == "iql":
                self.logger.log_episode(
                    ep,
                    episode_rewards,
                    [0.0] * n_agents,
                    self.alg.agents[0].epsilon,
                )
            self.logger.dump_episodes(1)

            episode_rewards = []
            episode_length = 0

            if self.arglist.save_gifs:
                self.frame_saver.save_episode_gif()

            if ep % 20 == 0 and ep > 0:
                # update plots
                self.plotter.update_reward_plot(True)
                self.plotter.update_exploration_plot(True)

        duration = time.time() - start_time
        print("Overall duration: %.2fs" % duration)

        env.close()

    if __name__ == "__main__":
        ev = Eval()
        ev.eval()
