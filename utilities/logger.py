import os
import sys
from collections import namedtuple

import numpy as np


class Logger:
    """
    Class to log training information
    """

    def __init__(
        self,
        n_agents,
        eta,
        task_name="mape",
        run_name="default",
        alg_name="maddpg",
        cur_name=None,
        log_path="logs",
    ):
        """
        Create Logger instance
        :param n_agents: number of agents
        :param eta: weighting of intrinsic rewards
        :param task_name: name of task
        :param run_name: name of run iteration
        :param alg_name: name of algorithm
        :param cur_name: name of curiosity if used
        :param log_path: path where logs should be saved
        """
        self.n_agents = n_agents
        self.eta = eta
        self.task_name = task_name
        self.run_name = run_name
        self.alg_name = alg_name
        self.cur_name = cur_name
        self.log_path = log_path

        # episode info
        self.episode = namedtuple("Ep", "number rewards variances exploration_value")
        self.current_episode = 0
        self.episodes = []

        # loss info
        self.loss = namedtuple("Loss", "name episode mean variance")

        # parameters in arrays (for efficiency)
        self.rewards_means = []
        self.rewards_vars = []
        self.exploration_values = []
        self.alg_losses_list = [[] for i in range(n_agents)]
        self.cur_losses_list = [[] for i in range(n_agents)]
        # store current episode
        self.current_alg_losses_list = [[] for i in range(n_agents)]
        self.current_cur_losses_list = [[] for i in range(n_agents)]

        # alg losses
        self.alg_losses = []
        for _ in range(n_agents):
            losses = {}
            if alg_name == "maddpg":
                losses["critic"] = []
                losses["actor"] = []
            elif alg_name == "iql":
                losses["qnetwork"] = []
            self.alg_losses.append(losses)

        # cur losses
        if cur_name is not None:
            self.cur_losses = []
            self.intrinsic_rewards_means = []
            self.intrinsic_rewards_vars = []
            for _ in range(n_agents):
                losses = {}
                if cur_name == "icm":
                    losses["forward"] = []
                    losses["inverse"] = []
                elif cur_name == "rnd":
                    losses["forward"] = []
                elif cur_name == "count":
                    losses["count"] = []
                else:
                    raise ValueError("Unknown curiosity: " + cur_name)
                self.cur_losses.append(losses)
                self.intrinsic_rewards_means.append([])
                self.intrinsic_rewards_vars.append([])

    def log_episode(self, ep, rewards_means, rewards_vars, exploration_value):
        """
        Save episode information
        :param ep: episode number
        :param rewards_means: average rewards during episode (for each agent)
        :param rewards_vars: variance of rewards during episode (for each agent)
        :param exploration_value: value for exploration
        """
        ep = self.episode(ep, rewards_means, rewards_vars, exploration_value)
        self.episodes.append(ep)
        self.rewards_means.append(rewards_means)
        self.rewards_vars.append(rewards_vars)
        self.exploration_values.append(exploration_value)

        self.current_episode = ep

        n_losses = 0
        for l in self.current_alg_losses_list:
            n_losses += l.__len__()
        if n_losses == 0:
            return

        for i in range(self.n_agents):
            # move alg losses
            if self.alg_name == "maddpg":
                # critic loss
                critic_losses = np.array(
                    list(map(lambda x: x[0], self.current_alg_losses_list[i]))
                )
                c_loss_mean = critic_losses.mean()
                c_loss = self.loss("critic", ep.number, c_loss_mean, critic_losses.var())
                # actor loss
                actor_losses = np.array(list(map(lambda x: x[1], self.current_alg_losses_list[i])))
                a_loss_mean = actor_losses.mean()
                a_loss = self.loss("actor", ep.number, a_loss_mean, actor_losses.var())
                # save in lists
                self.alg_losses[i]["critic"].append(c_loss)
                self.alg_losses[i]["actor"].append(a_loss)
                self.alg_losses_list[i].append([c_loss_mean, a_loss_mean])
            elif self.alg_name == "iql":
                q_losses = np.array(self.current_alg_losses_list[i])
                q_loss_mean = q_losses.mean()
                q_loss = self.loss("qnetwork", ep.number, q_loss_mean, q_losses.var())
                self.alg_losses[i]["qnetwork"].append(q_loss)
                self.alg_losses_list[i].append(q_loss_mean)

            # move curiosity losses
            if self.cur_name == "icm":
                # forward loss
                forward_losses = np.array(
                    list(map(lambda x: x[0], self.current_cur_losses_list[i]))
                )
                f_loss_mean = forward_losses.mean()
                f_loss = self.loss("forward", ep.number, f_loss_mean, forward_losses.var())
                # inverse loss
                inverse_losses = np.array(
                    list(map(lambda x: x[1], self.current_cur_losses_list[i]))
                )
                i_loss_mean = inverse_losses.mean()
                i_loss = self.loss("inverse", ep.number, i_loss_mean, inverse_losses.var())
                # save in lists
                self.cur_losses[i]["forward"].append(f_loss)
                self.cur_losses[i]["inverse"].append(i_loss)
                self.cur_losses_list[i].append([f_loss_mean, i_loss_mean])
                self.intrinsic_rewards_means[i].append(f_loss_mean * self.eta)
                self.intrinsic_rewards_vars[i].append(forward_losses.var() * self.eta ** 2)
            elif self.cur_name == "rnd":
                forward_losses = np.array(self.current_cur_losses_list[i])
                f_loss_mean = forward_losses.mean()
                f_loss = self.loss("forward", ep.number, f_loss_mean, forward_losses.var())
                self.cur_losses[i]["forward"].append(f_loss)
                self.cur_losses_list[i].append(f_loss_mean)
                self.intrinsic_rewards_means[i].append(f_loss_mean * self.eta)
                self.intrinsic_rewards_vars[i].append(forward_losses.var() * self.eta ** 2)
            elif self.cur_name == "count":
                count = np.array(self.current_cur_losses_list[i])
                count_mean = count.mean()
                count_loss = self.loss("count", ep.number, count_mean, count.var())
                self.cur_losses[i]["count"].append(count_loss)
                self.cur_losses_list[i].append(count_mean)
                self.intrinsic_rewards_means[i].append(count_mean * self.eta)
                self.intrinsic_rewards_vars[i].append(count.var() * self.eta ** 2)

        # empty current episode lists
        self.current_alg_losses_list = [[] for i in range(self.n_agents)]
        self.current_cur_losses_list = [[] for i in range(self.n_agents)]

    def log_losses(self, ep, losses):
        """
        Save loss information
        :param ep: episode number
        :param losses: losses of algorithm + intrinsic reward if used
        """
        # extract losses
        if self.alg_name == "maddpg":
            critic_loss, actor_loss, intrinsic_loss = losses
            if len(critic_loss) > 0 and len(actor_loss) > 0:
                for i in range(self.n_agents):
                    c_loss = critic_loss[i].item()
                    a_loss = actor_loss[i].item()
                    self.current_alg_losses_list[i].append((c_loss, a_loss))
        elif self.alg_name == "iql":
            qnet_loss, intrinsic_loss = losses
            if len(qnet_loss) > 0:
                for i in range(self.n_agents):
                    q_loss = qnet_loss[i].item()
                    self.current_alg_losses_list[i].append(q_loss)

        if self.cur_name == "icm":
            if len(intrinsic_loss) > 0:
                for i in range(self.n_agents):
                    forward_loss, inverse_loss = intrinsic_loss[i]
                    self.current_cur_losses_list[i].append(
                        (forward_loss.item(), inverse_loss.item())
                    )
        elif self.cur_name == "rnd":
            if len(intrinsic_loss) > 0:
                for i in range(self.n_agents):
                    i_loss = intrinsic_loss[i][0].item()
                    self.current_cur_losses_list[i].append(i_loss)
        elif self.cur_name == "count":
            if len(intrinsic_loss) > 0:
                for i in range(self.n_agents):
                    count_loss = intrinsic_loss[i][0].item()
                    self.current_cur_losses_list[i].append(count_loss)

    def dump_episodes(self, num=None):
        """
        Output episode info
        :param num: number of last episodes to output info for (or all if None)
        """
        if num is None:
            start_idx = 0
        else:
            start_idx = -num
        print("\n\nEpisode\t\t\trewards\t\t\tvariances\t\t\t\texploration")
        for ep in self.episodes[start_idx:]:
            line = str(ep.number) + "\t\t\t"
            for reward in ep.rewards:
                line += "%.3f " % reward
            line = line[:-1] + "\t\t"
            for var in ep.variances:
                line += "%.3f " % var
            line = line[:-1] + "\t\t\t"
            line += "%.3f" % ep.exploration_value
            print(line)
        print()

    def __format_time(self, time):
        """
        format time from seconds to string
        :param time: time in seconds (float)
        :return: time_string
        """
        hours = time // 3600
        time -= hours * 3600
        minutes = time // 60
        time -= minutes * 60
        time_string = "%d:%d:%.2f" % (hours, minutes, time)
        return time_string

    def dump_train_progress(self, ep, num_episodes, duration):
        """
        Output training progress info
        :param ep: current episode number
        :param num_episodes: number of episodes to complete
        :param duration: training duration so far (in seconds)
        """
        print(
            "Training progress:\tepisodes: %d/%d\t\t\t\tduration: %s"
            % (ep + 1, num_episodes, self.__format_time(duration))
        )
        progress_percent = (ep + 1) / num_episodes
        remaining_duration = duration * (1 - progress_percent) / progress_percent

        arrow_len = 50
        arrow_progress = int(progress_percent * arrow_len)
        arrow_string = "|" + arrow_progress * "=" + ">" + (arrow_len - arrow_progress) * " " + "|"
        print(
            "%.2f%%\t%s\tremaining duration: %s\n"
            % (progress_percent * 100, arrow_string, self.__format_time(remaining_duration))
        )

    def dump_losses(self, num=None):
        """
        Output loss info
        :param num: number of last loss entries to output (or all if None)
        """
        if self.alg_name == "maddpg":
            num_entries = len(self.alg_losses[0]["critic"])
        elif self.alg_name == "iql":
            num_entries = len(self.alg_losses[0]["qnetwork"])
        start_idx = 0
        if num is not None:
            start_idx = num_entries - num

        if num_entries == 0:
            print("No loss values stored yet!")
            return

        # build header
        header = "Episode index\t\tagent_id:\t\t"
        # alg header
        if self.alg_name == "maddpg":
            header += "actor_loss\t\tcritic_loss\t\t"
        elif self.alg_name == "iql":
            header += "q_loss "
        # cur header
        if self.cur_name == "icm":
            header += "(forward_loss\tinverse_loss)"
        elif self.cur_name == "rnd":
            header += "(forward_loss)"
        elif self.cur_name == "count":
            header += "(count)"
        print(header)

        for i in range(start_idx, num_entries):
            for j in range(self.n_agents):
                alg_loss = self.alg_losses[j]
                line = ""
                if self.alg_name == "maddpg":
                    a_loss = alg_loss["actor"][i]
                    c_loss = alg_loss["critic"][i]
                    line += str(a_loss.episode) + "\t\t\t" + str(j + 1) + ":\t\t\t"
                    line += "%.5f\t\t%.5f\t\t\t" % (a_loss.mean, c_loss.mean)
                elif self.alg_name == "iql":
                    q_loss = alg_loss["qnetwork"][i]
                    line += str(q_loss.episode) + "\t\t\t" + str(j + 1) + ":\t\t\t"
                    line += "%.5f\t\t" % q_loss.mean
                if self.cur_name is not None:
                    cur_loss = self.cur_losses[j]
                    if self.cur_name == "icm":
                        f_loss = cur_loss["forward"][i]
                        i_loss = cur_loss["inverse"][i]
                        line += "(%.5f\t%.5f)" % (f_loss.mean, i_loss.mean)
                    elif self.cur_name == "rnd":
                        f_loss = cur_loss["forward"][i]
                        line += "(%.5f)" % f_loss.mean
                    elif self.cur_name == "count":
                        count_loss = cur_loss["count"][i]
                        line += "(%.5f)" % count_loss.mean
                print(line)

    def clear_logs(self):
        """
        Remove log files in log dir
        """
        if not os.path.isdir(self.log_path):
            return
        alg_dir = os.path.join(self.log_path, self.alg_name)
        if not os.path.isdir(alg_dir):
            return
        log_dir = os.path.join(alg_dir, self.run_name)
        if not os.path.isdir(log_dir):
            return
        for f in os.listdir(log_dir):
            f_path = os.path.join(log_dir, f)
            if not os.path.isfile(f_path):
                continue
            os.remove(f_path)

    def save_episodes(self, num=None, extension="final"):
        """
        Save episode information in CSV file
        :param num: number of last episodes to save (or all if None)
        :param extension: extension name of csv file
        """
        if not os.path.isdir(self.log_path):
            os.mkdir(self.log_path)
        alg_dir = os.path.join(self.log_path, self.alg_name)
        if not os.path.isdir(alg_dir):
            os.mkdir(alg_dir)
        log_dir = os.path.join(alg_dir, self.run_name)
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

        csv_name = self.alg_name + "_" + self.task_name + "_epinfo_" + extension + ".csv"
        csv_path = os.path.join(log_dir, csv_name)

        with open(csv_path, "w") as csv_file:
            # write header line
            h = "number,rewards,variances,exploration_value\n"
            csv_file.write(h)

            if num is None:
                start_idx = 0
            else:
                start_idx = -num
            for ep in self.episodes[start_idx:]:
                line = ""
                line += str(ep.number) + ","
                if len(ep.rewards) > 1:
                    line += "["
                    for r in ep.rewards:
                        line += "%.5f " % r
                    line = line[:-1] + "],"
                else:
                    line += str(ep.rewards) + ","
                if len(ep.variances) > 1:
                    line += "["
                    for v in ep.variances:
                        line += "%.5f " % v
                    line = line[:-1] + "],"
                else:
                    line += str(ep.variances) + ","
                line += str(ep.exploration_value) + "\n"
                csv_file.write(line)

    def save_losses(self, num=None, extension="final"):
        """
        Save loss information in CSV file
        :param num: number of last episodes to save (or all if None)
        :param extension: extension name of csv file
        """
        if not os.path.isdir(self.log_path):
            os.mkdir(self.log_path)
        alg_dir = os.path.join(self.log_path, self.alg_name)
        if not os.path.isdir(alg_dir):
            os.mkdir(alg_dir)
        log_dir = os.path.join(alg_dir, self.run_name)
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

        csv_name = self.alg_name + "_" + self.task_name + "_lossinfo_" + extension + ".csv"
        csv_path = os.path.join(log_dir, csv_name)

        with open(csv_path, "w") as csv_file:
            # write header line
            h = "iteration,episode,"
            for i in range(self.n_agents):
                if self.alg_name == "maddpg":
                    h += str(i + 1) + "_maddpg_actor_loss,"
                    h += str(i + 1) + "_maddpg_critic_loss,"
                elif self.alg_name == "iql":
                    h += str(i + 1) + "_iql_loss,"

                if self.cur_name == "icm":
                    h += str(i + 1) + "_icm_forward_loss,"
                    h += str(i + 1) + "_icm_inverse_loss,"
                elif self.cur_name == "rnd":
                    h += str(i + 1) + "_rnd_forward_loss,"
                elif self.cur_name == "count":
                    h += str(i + 1) + "_count_value,"
            h = h[:-1] + "\n"
            csv_file.write(h)

            if self.alg_name == "maddpg":
                num_entries = len(self.alg_losses[0]["critic"])
            elif self.alg_name == "iql":
                num_entries = len(self.alg_losses[0]["qnetwork"])
            start_idx = 0
            if num is not None:
                start_idx = num_entries - num

            for i in range(start_idx, num_entries):
                line = str(i) + ","
                for j in range(self.n_agents):
                    alg_loss = self.alg_losses[j]
                    if self.alg_name == "maddpg":
                        a_loss = alg_loss["actor"][i]
                        c_loss = alg_loss["critic"][i]
                        if j == 0:
                            line += str(a_loss.episode) + ","
                        line += "%.5f,%.5f," % (a_loss.mean, c_loss.mean)
                    elif self.alg_name == "iql":
                        q_loss = alg_loss["qnetwork"][i]
                        if j == 0:
                            line += str(q_loss.episode) + ","
                        line += "%.5f," % q_loss.mean
                    if self.cur_name is not None:
                        cur_loss = self.cur_losses[j]
                        if self.cur_name == "icm":
                            f_loss = cur_loss["forward"][i]
                            i_loss = cur_loss["inverse"][i]
                            line += "%.5f,%.5f," % (f_loss.mean, i_loss.mean)
                        elif self.cur_name == "rnd":
                            f_loss = cur_loss["forward"][i]
                            line += "%.5f," % f_loss.mean
                        elif self.cur_name == "count":
                            count_loss = cur_loss["count"][i]
                            line += "%.5f," % count_loss.mean
                line = line[:-1] + "\n"
                csv_file.write(line)

    def save_duration_cuda(self, duration, cuda):
        """
        Store mini log file with duration and if cuda was used
        :param duration: duration of run in seconds
        :param cuda: flag if cuda was used
        """
        if not os.path.isdir(self.log_path):
            os.mkdir(self.log_path)
        alg_dir = os.path.join(self.log_path, self.alg_name)
        if not os.path.isdir(alg_dir):
            os.mkdir(alg_dir)
        log_dir = os.path.join(alg_dir, self.run_name)
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

        log_name = self.alg_name + "_" + self.task_name + ".log"
        log_path = os.path.join(log_dir, log_name)

        with open(log_path, "w") as log_file:
            log_file.write("duration: %.2fs\n" % duration)
            log_file.write("cuda: %s\n" % str(cuda))

    def save_parameters(
        self, env, task, n_agents, observation_sizes, action_sizes, discrete_actions, arglist
    ):
        """
        Store mini csv file with used parameters
        :param env: environment name
        :param task: task name
        :param n_agents: number of agents
        :param observation_sizes: dimension of observation for each agent
        :param action_sizes: dimension of action for each agent
        :param discrete_actions: flag indicating if actions are discrete
        :param arglist: parsed arglist of parameters
        """
        if not os.path.isdir(self.log_path):
            os.mkdir(self.log_path)
        alg_dir = os.path.join(self.log_path, self.alg_name)
        if not os.path.isdir(alg_dir):
            os.mkdir(alg_dir)
        log_dir = os.path.join(alg_dir, self.run_name)
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

        log_name = self.alg_name + "_" + self.task_name + "_parameters.csv"
        log_path = os.path.join(log_dir, log_name)

        with open(log_path, "w") as log_file:
            log_file.write("param,value\n")
            log_file.write("env,%s\n" % env)
            log_file.write("task,%s\n" % task)
            log_file.write("n_agents,%d\n" % n_agents)
            log_file.write("observation_sizes,%s\n" % observation_sizes)
            log_file.write("action_sizes,%s\n" % action_sizes)
            log_file.write("discrete_actions,%s\n" % discrete_actions)
            for arg in vars(arglist):
                log_file.write(arg + ",%s\n" % str(getattr(arglist, arg)))
