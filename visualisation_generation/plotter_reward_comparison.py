import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from csv_util import csv_to_dict, extract_value, collect_values, combine_arrays
from plot_util import plt_axis, plt_shade, plt_legend, COLORS
from plotter_summary import parse_logs_path, parse_plots_path, parse_envs, extract_all_confs, run_dirs_by_conf, conf_to_name


def conf_list_to_names(conf_list):
    """
    Transform list of configurations to conf names
    :param conf_list: list of read configuration
    :return: list of configuration names
    """
    confs = []
    for conf in conf_list:
        c = conf.split("_")
        c_string = []
        for s in c:
            if "count" in s or "icm" in s or "rnd" in s or "iql" in s or "maddpg" in s:
                c_string.append(s.upper())
            elif "joint" in s:
                c_string.append("(joint)")
            elif "partialobservable" in s:
                c_string.append("under partial observability")
            elif "sparserewards" in s:
                c_string.append("with sparse rewards")
        confs.append(" ".join(c_string))
    return confs

def parse_args(argv):
    """
    Parse arguments
    :param argv: list of arguments
    :return: logs_dir, plots_dir, envs, filters
    """
    if len(argv) < 3:
        print("Usage: ./plotter_summary.py <path/to/logs/dir> <path/to/plots/dir> (<env_name/ALL>) ([conf1,conf2,...])")
        print("where each conf is a list of strings to match")
        print("If no environment name is given, then all found environments will be summarised.")
        sys.exit(1)

    # check given logs dir
    logs_dir = parse_logs_path(argv)
    # create plots dir if needed
    plots_dir = parse_plots_path(argv)
    # parse envs
    envs, valid_env = parse_envs(argv)

    # collect filters after optional environment
    if len(argv) > 3 and not valid_env or len(argv) > 4 and valid_env:
        filter_start = 4 if valid_env else 3
        filters_strings = argv[filter_start:]
    else:
        filters_strings = []

    filters = []
    for f_string in filters_strings:
        f_string = "['" + f_string[1:-1].replace(",","','") + "']"
        filters.append(ast.literal_eval(f_string))
    
    return logs_dir, plots_dir, envs, filters


def generate_reward_comparison_plot(episode_list, reward_list, variance_list, conf_list, path, y_min=None, y_max=None):
    """
    Generate reward plots for rewards
    :param episode_list: list of episode arrays
    :param reward_list: list of reward arrays
    :param variance_list: list of reward variance arrays
    :param conf_list: list of configurations matching with reward_list
    :param path: path to store plot at
    :param y_min: minimum y-axis value
    :param y_max: maximum y-axis value
    """
    plt.clf()
    plt_axis()
    plt.title("Average Episodic Rewards", fontsize=14)
    plt.xlabel(r"Episode", fontsize=12)
    plt.ylabel(r"Rewards", fontsize=12)
    plt.xticks(np.arange(0, 30000, 5000))

    axes = plt.gca()
    if y_min is not None and y_max is not None:
        axes.set_ylim([y_min, y_max])

    n_agents = reward_list[0].shape[-1]
    
    confs = conf_list_to_names(conf_list)
    for i, (episodes, rewards, variances, conf_name) in enumerate(zip(episode_list, reward_list, variance_list, confs)):
        for j in range(n_agents):
            plt_shade(rewards[:, :, j] * 25, episodes, variances[:, :, j] * 25**2, COLORS[i], label=conf_name)
    plt_legend()
    plt.savefig(path, format="pdf")

def generate_reward_comparisons(logs_dir, plots_dir, envs, filters_list):
    """
    Generate reward plots for rewards of all filter matching configurations
    :param logs_dir: path to logs directory
    :param plots_dir: path to plots directory where plots are stored
    :param envs: list of environments
    :param filters_list: list of filters to apply
    """
    # create env directories
    for env in envs:
        env_dir = os.path.join(plots_dir, env)
        if not os.path.isdir(env_dir):
            os.mkdir(env_dir)
    
    confs_list = []
    for filters in filters_list:
        confs = extract_all_confs(logs_dir, envs, filters, match=True)
        confs_list.append(confs)

    for env in envs:
        print(env)
        env_dir = os.path.join(plots_dir, env)
        episode_list = []
        reward_list = []
        variance_list = []
        conf_list = []
        for confs in confs_list:
            for c in confs:
                c_name = conf_to_name(c)
                dirs = run_dirs_by_conf(logs_dir, env, c)
                if not dirs:
                    # no logs for this conf/ env combination
                    continue
                
                print("\t" + str(c))

                epinfo_dicts = []
                loss_dicts = []
                for d in dirs:
                    for f in os.listdir(d):
                        log_path = os.path.join(d, f)
                        if "epinfo" in f:
                            ep_dict = csv_to_dict(log_path)
                            epinfo_dicts.append(ep_dict)
                        if "lossinfo" in f:
                            loss_dict = csv_to_dict(log_path)
                            loss_dicts.append(loss_dict)

                # collect rewards
                episodes = [extract_value(d, "number") for d in epinfo_dicts]
                rewards = [extract_value(d, "rewards") for d in epinfo_dicts]
                variances = [extract_value(d, "variances") for d in epinfo_dicts]

                episode_list.append(episodes[0])
                reward_list.append(combine_arrays(rewards))
                variance_list.append(combine_arrays(variances))
                conf_list.append(c_name)
        
        # generate plot of all rewards
        if reward_list:
            filters_name = "-".join(["_".join(filters) for filters in filters_list])
            if len(filters_name) > 100:
                filters_name = filters_name[:50] + "__" + filters_name[-50:]
            path = os.path.join(env_dir, "reward_comparison_" + filters_name + ".pdf")
            if env == "simple_speaker_listener":
                y_max = 0
                y_min = -300
            elif env == "simple_spread":
                y_max = -100
                y_min = -400
            elif env == "simple_adversary":
                y_max = 100
                y_min = -300
            elif env == "simple_tag":
                y_max = 100
                y_min = -100
            if "sparserewards" in filters_name:
                if env == "simple_speaker_listener":
                    y_min = -450
                elif env == "simple_spread":
                    y_min = -500
                elif env == "simple_adversary":
                    y_max = 100
                    y_min = -300
                elif env == "simple_tag":
                    y_max = 100
                    y_min = -150
            generate_reward_comparison_plot(
                    episode_list,
                    reward_list,
                    variance_list,
                    conf_list,
                    path,
                    y_min,
                    y_max,
            )
        else:
            print("\tNo matching configurations found!")

if __name__ == "__main__":
    logs_dir, plots_dir, envs, filters_list = parse_args(sys.argv)
    generate_reward_comparisons(logs_dir, plots_dir, envs, filters_list)
