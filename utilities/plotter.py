import os
from collections import OrderedDict

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import rc

# rc("text", usetex=True)
rc("font", family="serif", size=28)
rc("axes", labelsize=28)

import matplotlib.cm as cm


class Plotter:
    """
    Class to generate and save plots
    """

    def __init__(
        self,
        logger,
        n_agents,
        eval_frequency,
        task_name="mape",
        run_name="default",
        alg_name="maddpg",
        cur_name=None,
        plot_dir="plots",
    ):
        """
        Create Plotter object
        :param logger: logger object to extract info from
        :param n_agents: number of agents
        :param eval_frequency: how often are episodes rewards logged
        :param task_name: name of task
        :param plot_dir: directory in which plots are stored (in subdirectories)
        :param run_name: name of run used for subdirectory names
        :param alg_name: algorithm name
        :param cur_name: curiosity name
        """
        self.logger = logger
        self.n_agents = n_agents
        self.eval_frequency = eval_frequency
        self.task_name = task_name
        self.run_name = run_name
        self.alg_name = alg_name
        self.cur_name = cur_name
        self.plot_dir = plot_dir

        self.reward_figure = plt.figure(1)
        self.exploration_figure = plt.figure(2)
        if alg_name == "maddpg":
            self.alg_loss_figures = [plt.figure(3), plt.figure(4)]
        elif alg_name == "iql":
            self.alg_loss_figures = [plt.figure(3)]
        if cur_name is not None:
            if cur_name == "icm":
                self.cur_loss_figures = [plt.figure(5), plt.figure(6)]
            elif cur_name == "rnd" or cur_name == "count":
                self.cur_loss_figures = [plt.figure(5)]
            self.intrinsic_reward_figure = plt.figure(7)
        # self.colors = cm.rainbow(np.linspace(0, 1, n_agents))
        self.colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        if self.n_agents > 10:
            raise ValueError("Need more colours!")

    def plt_legend(self):
        """
        Plot transparent legend without duplicate labels
        """
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fancybox=True, framealpha=0.5, fontsize=12)

    def plt_shade(self, xvals, means, vars, index):
        """
        Plot shaded curve
        :param xvals: numpy array of x axis values
        :param means: numpy array of means
        :param vars: numpy array of variances
        :param index: index of agent's plot
        """
        stds = np.sqrt(vars)
        plt.plot(
            xvals, means, "-", c=self.colors[index], alpha=0.7, label="agent %d" % (index + 1)
        )
        plt.fill_between(
            xvals,
            means - stds,
            means + stds,
            alpha=0.2,
            edgecolor=self.colors[index],
            facecolor=self.colors[index],
            antialiased=True,
        )

    def update_reward_plot(self, display=True):
        """
        Update live plot of rewards
        :param display: flag if plot should be displayed
        """
        rewards = np.array(self.logger.rewards_means)
        rewards_vars = np.array(self.logger.rewards_vars)
        episodes = np.arange(0, rewards.shape[0] * self.eval_frequency, self.eval_frequency)
        # set current figure to reward figure
        plt.figure(self.reward_figure.number)
        plt.clf()
        # adjust axis ticks
        plt.xticks(fontsize=12, ha="center", va="top")
        plt.yticks(fontsize=12, ha="right", va="center")
        # tick parameters
        plt.tick_params(
            axis="both",
            bottom=True,
            top=True,
            left=True,
            right=True,
            direction="in",
            which="major",
            grid_color="blue",
        )
        # define grid
        plt.grid(linestyle="--", linewidth=0.5, alpha=0.15)
        plt.title(r"Cumulative Rewards", fontsize=14)
        plt.xlabel(r"Episode", fontsize=12)
        plt.ylabel(r"Rewards", fontsize=12)

        for i in range(rewards.shape[-1]):
            self.plt_shade(episodes, rewards[:, i], rewards_vars[:, i], i)
        if display:
            plt.ion()
            self.reward_figure.show()
            plt.pause(0.0001)

    def update_intrinsic_reward_plot(self, display=True):
        """
        Update live plot of rewards intrinsic rewards
        :param display: flag if plot should be displayed
        """
        intrinsic_rewards_means = np.asarray(self.logger.intrinsic_rewards_means)
        intrinsic_rewards_vars = np.asarray(self.logger.intrinsic_rewards_vars)
        episodes = np.arange(
            self.eval_frequency,
            (intrinsic_rewards_means.shape[1] + 1) * self.eval_frequency,
            self.eval_frequency,
        )
        # set current figure to intrinsic reward figure
        plt.figure(self.intrinsic_reward_figure.number)
        plt.clf()
        # adjust axis ticks
        plt.xticks(fontsize=12, ha="center", va="top")
        plt.yticks(fontsize=12, ha="right", va="center")
        # tick parameters
        plt.tick_params(
            axis="both",
            bottom=True,
            top=True,
            left=True,
            right=True,
            direction="in",
            which="major",
            grid_color="blue",
        )
        # define grid
        plt.grid(linestyle="--", linewidth=0.5, alpha=0.15)
        plt.title(r"Intrinsic Rewards", fontsize=14)
        plt.xlabel(r"Episode", fontsize=12)
        plt.ylabel(r"Rewards", fontsize=12)

        for i in range(self.n_agents):
            self.plt_shade(
                episodes, intrinsic_rewards_means[i, :], intrinsic_rewards_vars[i, :], i
            )
        self.plt_legend()
        if display:
            plt.ion()
            self.reward_figure.show()
            plt.pause(0.0001)

    def update_exploration_plot(self, display=True):
        """
        Update live plot of exploration values
        :param display: flag if plot should be displayed
        """
        exploration_values = self.logger.exploration_values
        episodes = np.arange(0, len(exploration_values) * self.eval_frequency, self.eval_frequency)
        # set current figure to reward figure
        plt.figure(self.exploration_figure.number)
        plt.clf()
        # adjust axis ticks
        plt.xticks(fontsize=12, ha="center", va="top")
        plt.yticks(fontsize=12, ha="right", va="center")
        # tick parameters
        plt.tick_params(
            axis="both",
            bottom=True,
            top=True,
            left=True,
            right=True,
            direction="in",
            which="major",
            grid_color="blue",
        )
        # define grid
        plt.grid(linestyle="--", linewidth=0.5, alpha=0.15)
        if self.alg_name == "maddpg":
            plt.title(r"Exploration Variance", fontsize=14)
        elif self.alg_name == "iql":
            plt.title(r"Exploration Epsilon", fontsize=14)
        plt.xlabel(r"Episode", fontsize=12)
        if self.alg_name == "maddpg":
            plt.ylabel(r"Variance", fontsize=12)
        elif self.alg_name == "iql":
            plt.ylabel(r"Epsilon", fontsize=12)

        plt.plot(episodes, exploration_values, "-", c=self.colors[1], alpha=0.7, lw=2)
        if display:
            plt.ion()
            self.exploration_figure.show()
            plt.pause(0.0001)

    def update_alg_loss_plot(self, display=True):
        """
        Update plot of algorithm loss(es)
        :param display: flag if plot should be displayed
        """
        # algorithm loss plot
        for f in self.alg_loss_figures:
            plt.figure(f.number)
            plt.clf()
            # adjust axis ticks
            plt.xticks(fontsize=12, ha="center", va="top")
            plt.yticks(fontsize=12, ha="right", va="center")
            # tick parameters
            plt.tick_params(
                axis="both",
                bottom=True,
                top=True,
                left=True,
                right=True,
                direction="in",
                which="major",
                grid_color="blue",
            )
            # define grid
            plt.grid(linestyle="--", linewidth=0.5, alpha=0.15)
            plt.xlabel(r"Episodes", fontsize=12)
            plt.ylabel(r"Loss", fontsize=12)

        alg_losses = self.logger.alg_losses

        if self.alg_name == "maddpg":
            plt.figure(self.alg_loss_figures[0].number)
            plt.title(r"MADDPG Critic Loss", fontsize=14)
            for i in range(self.n_agents):
                critic_losses = alg_losses[i]["critic"]
                means = np.array([l.mean for l in critic_losses])
                vars = np.array([l.variance for l in critic_losses])
                episodes = np.arange(
                    self.eval_frequency,
                    (len(means) + 1) * self.eval_frequency,
                    self.eval_frequency,
                )
                self.plt_shade(episodes, means, vars, i)
            self.plt_legend()

            plt.figure(self.alg_loss_figures[1].number)
            plt.title(r"MADDPG Actor Loss", fontsize=14)
            for i in range(self.n_agents):
                actor_losses = alg_losses[i]["actor"]
                means = np.array([l.mean for l in actor_losses])
                vars = np.array([l.variance for l in actor_losses])
                episodes = np.arange(
                    self.eval_frequency,
                    (len(means) + 1) * self.eval_frequency,
                    self.eval_frequency,
                )
                self.plt_shade(episodes, means, vars, i)
            self.plt_legend()
        elif self.alg_name == "iql":
            plt.figure(self.alg_loss_figures[0].number)
            plt.title(r"IQL Q Loss", fontsize=14)
            for i in range(self.n_agents):
                q_losses = alg_losses[i]["qnetwork"]
                means = np.array([l.mean for l in q_losses])
                vars = np.array([l.variance for l in q_losses])
                episodes = np.arange(
                    self.eval_frequency,
                    (len(means) + 1) * self.eval_frequency,
                    self.eval_frequency,
                )
                self.plt_shade(episodes, means, vars, i)
            self.plt_legend()
        if display:
            plt.ion()
            for f in self.alg_loss_figures:
                f.show()
            plt.pause(0.0001)

    def update_cur_loss_plot(self, display=True):
        """
        Update plot of curiosity loss(es)
        :param display: flag if plot should be displayed
        """
        if self.cur_name is None:
            raise ValueError("Can't plot curiosity loss if there is no curiosity!")

        # algorithm loss plot
        for f in self.cur_loss_figures:
            plt.figure(f.number)
            plt.clf()
            # adjust axis ticks
            plt.xticks(fontsize=12, ha="center", va="top")
            plt.yticks(fontsize=12, ha="right", va="center")
            # tick parameters
            plt.tick_params(
                axis="both",
                bottom=True,
                top=True,
                left=True,
                right=True,
                direction="in",
                which="major",
                grid_color="blue",
            )
            # define grid
            plt.grid(linestyle="--", linewidth=0.5, alpha=0.15)
            plt.xlabel(r"Episodes", fontsize=12)
            plt.ylabel(r"Loss", fontsize=12)

        cur_losses = self.logger.cur_losses

        if self.cur_name == "icm":
            plt.figure(self.cur_loss_figures[0].number)
            plt.title(r"ICM Forward Loss", fontsize=14)
            for i in range(self.n_agents):
                f_losses = cur_losses[i]["forward"]
                means = np.array([l.mean for l in f_losses])
                vars = np.array([l.variance for l in f_losses])
                episodes = np.arange(
                    self.eval_frequency,
                    (len(means) + 1) * self.eval_frequency,
                    self.eval_frequency,
                )
                self.plt_shade(episodes, means, vars, i)
            self.plt_legend()

            plt.figure(self.cur_loss_figures[1].number)
            plt.title(r"ICM Inverse Loss", fontsize=14)
            for i in range(self.n_agents):
                i_losses = cur_losses[i]["inverse"]
                means = np.array([l.mean for l in i_losses])
                vars = np.array([l.variance for l in i_losses])
                episodes = np.arange(
                    self.eval_frequency,
                    (len(means) + 1) * self.eval_frequency,
                    self.eval_frequency,
                )
                self.plt_shade(episodes, means, vars, i)
            self.plt_legend()
        elif self.cur_name == "rnd":
            plt.figure(self.cur_loss_figures[0].number)
            plt.title(r"RND Forward Loss", fontsize=14)
            for i in range(self.n_agents):
                f_losses = cur_losses[i]["forward"]
                means = np.array([l.mean for l in f_losses])
                vars = np.array([l.variance for l in f_losses])
                episodes = np.arange(
                    self.eval_frequency,
                    (len(means) + 1) * self.eval_frequency,
                    self.eval_frequency,
                )
                self.plt_shade(episodes, means, vars, i)
        elif self.cur_name == "count":
            plt.figure(self.cur_loss_figures[0].number)
            plt.title(r"Hash-Based Count", fontsize=14)
            for i in range(self.n_agents):
                count_losses = cur_losses[i]["count"]
                means = np.array([l.mean for l in count_losses])
                vars = np.array([l.variance for l in count_losses])
                episodes = np.arange(
                    self.eval_frequency,
                    (len(means) + 1) * self.eval_frequency,
                    self.eval_frequency,
                )
                self.plt_shade(episodes, means, vars, i)
            self.plt_legend()

        if display:
            plt.ion()
            for f in self.cur_loss_figures:
                f.show()
            plt.pause(0.0001)

    def clear_plots(self):
        """
        Remove plots stored in <plot_dir>/<alg>/<run>/<alg>_<task>_<plot_kind>_<extension>.pdf
        """
        if not os.path.isdir(self.plot_dir):
            return
        alg_dir = os.path.join(self.plot_dir, self.alg_name)
        if not os.path.isdir(alg_dir):
            return
        run_dir = os.path.join(alg_dir, self.run_name)
        if not os.path.isdir(run_dir):
            return
        for f in os.listdir(run_dir):
            f_path = os.path.join(run_dir, f)
            if not os.path.isfile(f_path):
                continue
            os.remove(f_path)

    def __save_plot(self, figure, file_name):
        """
        Save given figure - Stored in <plot_dir>/<alg>/<run>/<file_name>.pdf
        :param figure: figure to be stored
        :param file_name: name of file
        """
        if not os.path.isdir(self.plot_dir):
            os.mkdir(self.plot_dir)
        alg_dir = os.path.join(self.plot_dir, self.alg_name)
        if not os.path.isdir(alg_dir):
            os.mkdir(alg_dir)
        run_dir = os.path.join(alg_dir, self.run_name)
        if not os.path.isdir(run_dir):
            os.mkdir(run_dir)
        file_path = os.path.join(run_dir, file_name + ".pdf")
        plt.figure(figure.number)
        plt.savefig(file_path, format="pdf")

    def save_reward_plot(self, extension):
        """
        Save current reward plot - Stored in <plot_dir>/<alg>/<run>/<alg>_<task>_rewards_<extension>.pdf
        :param extension: name extension of file
        """
        file_name = self.alg_name + "_" + self.task_name + "_rewards_" + extension
        self.__save_plot(self.reward_figure, file_name)

    def save_intrinsic_reward_plot(self, extension):
        """
        Save current intrinsic reward plot - Stored in
            <plot_dir>/<alg>/<run>/<alg>_<task>_intrewards_<extension>.pdf
        :param extension: name extension of file
        """
        if self.cur_name is not None:
            file_name = (
                self.alg_name
                + "_"
                + self.task_name
                + "_"
                + self.cur_name
                + "_intrewards_"
                + extension
            )
            self.__save_plot(self.intrinsic_reward_figure, file_name)

    def save_exploration_plot(self, extension):
        """
        Save current exploration plot - Stored in <plot_dir>/<alg>/<run>/<alg>_<task>_exploration_<extension>.pdf
        :param extension: name extension of file
        """
        file_name = self.alg_name + "_" + self.task_name + "_exploration_" + extension
        self.__save_plot(self.exploration_figure, file_name)

    def save_alg_loss_plots(self, extension):
        """
        Save current algorithm loss plots - Stored in <plot_dir>/<alg>/<run>/<alg>_<losskind>_<task>_<extension>.pdf
        :param extension: name extension of file
        """
        if self.alg_name == "maddpg":
            criticloss_name = self.alg_name + "_criticloss_" + self.task_name + "_" + extension
            self.__save_plot(self.alg_loss_figures[0], criticloss_name)
            actorloss_name = self.alg_name + "_actorloss_" + self.task_name + "_" + extension
            self.__save_plot(self.alg_loss_figures[1], actorloss_name)
        elif self.alg_name == "iql":
            qloss_name = self.alg_name + "_qloss_" + self.task_name + "_" + extension
            self.__save_plot(self.alg_loss_figures[0], qloss_name)

    def save_cur_loss_plots(self, extension):
        """
        Save current curiosity loss plots - Stored in <plot_dir>/<alg>/<run>/<cur>_<losskind>_<task>_<extension>.pdf
        :param extension: name extension of file
        """
        if self.cur_name == "icm":
            forwardloss_name = self.cur_name + "_forwardloss_" + self.task_name + "_" + extension
            self.__save_plot(self.cur_loss_figures[0], forwardloss_name)
            inverseloss_name = self.cur_name + "_inverseloss_" + self.task_name + "_" + extension
            self.__save_plot(self.cur_loss_figures[1], inverseloss_name)
        elif self.cur_name == "rnd":
            forwardloss_name = self.cur_name + "_forwardloss_" + self.task_name + "_" + extension
            self.__save_plot(self.cur_loss_figures[0], forwardloss_name)
        elif self.cur_name == "count":
            count_name = self.cur_name + "_hashcount_" + self.task_name + "_" + extension
            self.__save_plot(self.cur_loss_figures[0], count_name)
