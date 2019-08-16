import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from csv_util import csv_to_dict, extract_value, collect_values, combine_arrays
from plot_util import plt_axis, plt_shade, plt_legend, COLORS
from plotter_summary import parse_logs_path, parse_plots_path, parse_envs, extract_all_confs, run_dirs_by_conf, conf_to_name
from plotter_reward_comparison import conf_list_to_names

def parse_args(argv):
    """
    Parse arguments
    :param argv: list of arguments
    :return: logs_dir, plots_dir, envs, filters
    """
    if len(argv) < 3:
        print("Usage: ./plotter_intreward_comparison.py <path/to/logs/dir> <path/to/plots/dir> (<env_name/ALL>) ([conf1,conf2,...])")
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


def generate_intreward_comparison_plot(episode_list, reward_list, conf_list, path):
    """
    Generate reward plots for rewards
    :param episode_list: list of episode arrays
    :param reward_list: list of intrinsic reward arrays
    :param conf_list: list of configurations matching with reward_list
    :param path: path to store plot at
    """
    plt.clf()
    plt_axis()
    plt.title("Intrinsic Rewards", fontsize=14)
    plt.xlabel(r"Episode", fontsize=12)
    plt.ylabel(r"Rewards", fontsize=12)
    n_agents = reward_list[0].shape[-1]
    plt.xticks(np.arange(0, 30000, 5000))
    
    confs = conf_list_to_names(conf_list)
    for i, (episodes, rewards, conf_name) in enumerate(zip(episode_list, reward_list, confs)):
        for j in range(n_agents):
            plt_shade(rewards[:, :, j], episodes, color=COLORS[i], label=conf_name)
    plt_legend()
    plt.savefig(path, format="pdf")

def generate_intreward_comparisons(logs_dir, plots_dir, envs, filters_list):
    """
    Generate intrinsic reward plots for rewards of all filter matching configurations
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
        conf_list = []
        for confs in confs_list:
            for c in confs:
                c_name = conf_to_name(c)
                dirs = run_dirs_by_conf(logs_dir, env, c)
                if not dirs:
                    # no logs for this conf/ env combination
                    continue
                
                print("\t" + str(c))

                # determine curiosity
                cur = None
                if "count" in c:
                    cur = "count"
                elif "icm" in c:
                    cur = "icm"
                elif "rnd" in c:
                    cur = "rnd"
                if cur is None:
                    continue

                epinfo_dicts = []
                loss_dicts = []
                eta = None
                for d in dirs:
                    for f in os.listdir(d):
                        log_path = os.path.join(d, f)
                        if "epinfo" in f:
                            ep_dict = csv_to_dict(log_path)
                            epinfo_dicts.append(ep_dict)
                        if "lossinfo" in f:
                            loss_dict = csv_to_dict(log_path)
                            loss_dicts.append(loss_dict)
                        if "parameters" in f:
                            param_dict = csv_to_dict(log_path, numeric=False)
                            for par, val in zip(param_dict["param"], param_dict["value"]):
                                if par == "eta":
                                    eta = float(val)
                                    break

                # collect intrinsic rewards
                episodes = [extract_value(d, "episode") for d in loss_dicts]
                if cur == "count":
                    intrinsic_rewards = collect_values(loss_dicts, ["count_value"])
                elif cur == "icm":
                    intrinsic_rewards = collect_values(loss_dicts, ["icm_forward_loss"])
                elif cur == "rnd":
                    intrinsic_rewards = collect_values(loss_dicts, ["rnd_forward_loss"])

                episode_list.append(episodes[0])
                reward_list.append(eta * combine_arrays(intrinsic_rewards, True))
                conf_list.append(c_name)
        
        # generate plot of all rewards
        if reward_list:
            filters_name = "-".join(["_".join(filters) for filters in filters_list])
            if len(filters_name) > 100:
                filters_name = filters_name[:50] + "__" + filters_name[-50:]
            path = os.path.join(env_dir, "intreward_comparison_" + filters_name + ".pdf")
            generate_intreward_comparison_plot(episode_list, reward_list, conf_list, path)
        else:
            print("\tNo matching configurations found!")

if __name__ == "__main__":
    logs_dir, plots_dir, envs, filters_list = parse_args(sys.argv)
    generate_intreward_comparisons(logs_dir, plots_dir, envs, filters_list)
