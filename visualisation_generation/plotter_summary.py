from functools import reduce
import numpy as np
import os
import sys

from csv_util import csv_to_dict, extract_value, collect_values, combine_arrays
from plot_util import generate_reward_plot, generate_exploration_plot, generate_alg_loss_plot, generate_cur_loss_plot

SCENARIOS = [
    "simple_speaker_listener",
    "simple_spread",
    "simple_adversary",
    "simple_tag",
]

ALGS = ["maddpg", "iql"]

def parse_logs_path(argv):
    logs_path = argv[1]
    logs_dir = os.path.abspath(logs_path)
    if not os.path.isdir(logs_dir):
        print("Given logs dir %s not found!" % logs_path)
        sys.exit(1)
    return logs_dir

def parse_plots_path(argv):
    plots_path = argv[2]
    plots_dir = os.path.abspath(plots_path)
    if not os.path.isdir(plots_dir):
        os.mkdir(plots_dir)
    return plots_dir

def parse_envs(argv):
    if len(argv) > 3:
        arg = argv[3]
        valid_env = False
        if arg.upper() == "ALL":
            env = "ALL"
            valid_env = True
        else:
            # cause functional programming is cool
            valid_env = arg in SCENARIOS
            if not valid_env:
                env = "ALL"
            else:
                env = arg
    else:
        env = "ALL"
        valid_env = False
    
    if env == "ALL":
        envs = SCENARIOS
    else:
        envs = [env]
    return envs, valid_env

def parse_args(argv):
    """
    Parse arguments
    :param argv: list of arguments
    :return: logs_dir, plots_dir, envs, filters
    """
    if len(argv) < 3:
        print("Usage: ./plotter_summary.py <path/to/logs/dir> <path/to/plots/dir> (<env_name/ALL>) (filters)")
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
        filters = argv[filter_start:]
    else:
        filters = []
    
    return logs_dir, plots_dir, envs, filters

def conf_matches_filters(conf, filters, match=False):
    """
    Check if configuration matches filters
    :param conf: configuration as list of strings
    :param filters: list of strings to filter after
    :param match: flag if only exact matches with filters are valid
    """
    if match:
        return conf == filters
    else:
        return reduce(lambda z, e: z and e in conf, filters, True)

def extract_conf(dir):
    """
    Identify configurations and scenario name from log dir
    :param dir: directory name
    :return: scenario_name, list of configurations
    """
    if dir.startswith("mape_"):
        dir_name = dir[5:]
    else:
        dir_name = dir

    sce = None
    for scenario in SCENARIOS:
        if dir_name.startswith(scenario):
            sce = scenario
    if sce is None:
        raise ValueError("directory %s does not contain a valid scenario!" % dir)
    
    dir_name = dir_name[len(sce) + 1:]
    splits = dir_name.split("_")

    confs = []
    for s in splits:
        if s.startswith("seed") or s.startswith("id"):
            continue
        confs.append(s)
    return sce, confs

def extract_all_confs(logs_dir, envs, filters, match=False):
    """
    Extract all configurations matching filters and environments
    :param logs_dir: log directory to search in
    :param envs: scenario names
    :param filters: filters for configurations
    :param match: if True then only exact matches with filters are added
    :return: list of configurations
    """
    confs = []

    for alg_name in os.listdir(logs_dir):
        alg_dir = os.path.join(logs_dir, alg_name)

        for conf_dir in os.listdir(alg_dir):
            _, c = extract_conf(conf_dir)
            if alg_name in c:
                c.remove(alg_name)
            c.insert(0, alg_name)
            if conf_matches_filters(c, filters, match) and c not in confs:
                confs.append(c)

    return confs

def run_dirs_by_conf(logs_dir, env, conf):
    """
    Extract run directories by configuration
    :param logs_dir: directory of log files
    :param env: environment name
    :param conf: configuration as param list
    """
    dirs = []
    for alg_name in os.listdir(logs_dir):
        alg_dir = os.path.join(logs_dir, alg_name)

        for conf_name in os.listdir(alg_dir):
            e, c = extract_conf(conf_name)
            if alg_name in c:
                c.remove(alg_name)
            c.insert(0, alg_name)

            if c != conf or e != env:
                continue

            dirs.append(os.path.join(alg_dir, conf_name))
    return dirs

def conf_to_name(confs):
    """
    Determine name of configuration
    :param confs: list of conf strings
    """
    separator = "_"
    return separator.join(confs)

def generate_plot_summaries(logs_dir, plots_dir, envs, filters):
    """
    Generate reward plots for rewards of all filter matching configurations
    :param logs_dir: path to logs directory
    :param plots_dir: path to plots directory where plots are stored
    :param envs: list of environments
    :param filters: list of filters to apply
    """

    # create env directories
    for env in envs:
        env_dir = os.path.join(plots_dir, env)
        if not os.path.isdir(env_dir):
            os.mkdir(env_dir)
    
    confs = extract_all_confs(logs_dir, envs, filters)

    for env in envs:
        print(env)
        env_dir = os.path.join(plots_dir, env)
        for c in confs:
            c_name = conf_to_name(c)
            c_dir = os.path.join(env_dir, c_name)
            if not os.path.isdir(c_dir):
                os.mkdir(c_dir)

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
                    if cur is not None and "parameters" in f:
                        param_dict = csv_to_dict(log_path, numeric=False)
                        for par, val in zip(param_dict["param"], param_dict["value"]):
                            if par == "eta":
                                eta = float(val)
                                break

            # collect rewards and generate plot
            episodes = extract_value(epinfo_dicts[0], "number")
            rewards = [extract_value(d, "rewards") for d in epinfo_dicts]
            variances = [extract_value(d, "variances") for d in epinfo_dicts]
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
            if "sparserewards" in c_name:
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
            generate_reward_plot(
                c_dir,
                episodes,
                combine_arrays(rewards),
                combine_arrays(variances),
                y_min=y_min,
                y_max=y_max,
            )

            # collect intrinsic rewards and generate plot
            if cur == "count":
                intrinsic_rewards = collect_values(loss_dicts, ["count_value"])
            elif cur == "icm":
                intrinsic_rewards = collect_values(loss_dicts, ["icm_forward_loss"])
            elif cur == "rnd":
                intrinsic_rewards = collect_values(loss_dicts, ["rnd_forward_loss"])
            else:
                intrinsic_rewards = []

            episodes = extract_value(loss_dicts[0], "episode")
            if intrinsic_rewards:
                generate_reward_plot(
                        c_dir,
                        episodes,
                        eta * combine_arrays(intrinsic_rewards, True),
                        intrinsic=True,
                )
            
            # collect exploration values and generate plot
            alg = c[0]
            exploration_values = [extract_value(d, "exploration_value") for d in epinfo_dicts]
            generate_exploration_plot(c_dir, combine_arrays(exploration_values), alg)

            # collect alg losses and generate plot
            if alg == "maddpg":
                alg_losses = collect_values(loss_dicts, ["maddpg_actor_loss", "maddpg_critic_loss"])
            elif alg == "iql":
                alg_losses = collect_values(loss_dicts, ["iql_loss"])
            generate_alg_loss_plot(c_dir, episodes, combine_arrays(alg_losses, True), alg)

            # collect cur losses and generate plot
            if cur == "count":
                cur_losses = collect_values(loss_dicts, ["count_value"])
            elif cur == "icm":
                cur_losses = collect_values(loss_dicts, ["icm_forward_loss", "icm_inverse_loss"])
            elif cur == "rnd":
                cur_losses = collect_values(loss_dicts, ["rnd_forward_loss"])
            if cur is not None:
                generate_cur_loss_plot(c_dir, episodes, combine_arrays(cur_losses, True), cur)

if __name__ == "__main__":
    logs_dir, plots_dir, envs, filters = parse_args(sys.argv)
    generate_plot_summaries(logs_dir, plots_dir, envs, filters)
