import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from plot_util import plt_axis, COLORS
from plotter_summary import parse_logs_path, parse_plots_path, parse_envs, extract_all_confs, run_dirs_by_conf, conf_to_name
from plotter_reward_comparison import conf_list_to_names

ENV_NAMES = {
    "simple_speaker_listener": "Cooperative Communication",
    "simple_spread": "Cooperative Navigation",
    "simple_adversary": "Physical Deception",
    "simple_tag": "Predator Prey"
}

ALGS = ["maddpg", "iql"]
CURIOSITIES = ["count", "icm", "rnd"]

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

def generate_time_comparison_chart(times, env, conf_list, path):
    """
    Generate bar chart for time comparison of training
    :param times: list of time arrays (#confs, #seed)
    :param conf_list: list of configurations matching with times
    :param env: name of environment (for title)
    :param path: path to store plot at
    """
    plt.clf()
    plt.figure(figsize=(len(conf_list) * 2.5, 4))
    plt_axis()
    plt.title(r"Average Training Duration (%s)" % ENV_NAMES[env], fontsize=14)
    plt.ylabel(r"Configuration", fontsize=12)
    plt.ylabel(r"Time (s)", fontsize=12)

    x_pos = [i for i in range(len(conf_list))]
    #if len(times.shape) > 1:
    mean = times.mean(axis=1)
    std = times.std(axis=1)
    #else:
    #    mean = times
    #    std = 0

    barlist = plt.bar(x_pos, mean, yerr=std, align='center', width=0.5)

    for i in range(len(barlist)):
        barlist[i].set_color(COLORS[i % len(COLORS)])

    confs = conf_list_to_names(conf_list)
    plt.xticks(x_pos, confs)
    plt.savefig(path, format="pdf")

def get_conf_index(conf_list, conf, eval_name=None):
    if eval_name is not None:
        conf = conf + "_" + eval_name
    for i, c in enumerate(conf_list):
        if c == conf:
            return i
    raise ValueError("Configuration %s not found with evaluation_name %s in configuration list %s!" % (conf, str(eval_name), str(conf_list)))

def generate_time_comparison_table(times, env, conf_list, path, eval_name=None):
    """
    Generate LaTeX table for time comparison of training
    :param times: list of time arrays (#confs, #seed)
    :param conf_list: list of configurations matching with times
    :param env: name of environment (for title)
    :param path: path to store plot at
    """
    mean = times.mean(axis=1)
    std = times.std(axis=1)
    with open(path, 'w') as f:
        algs_format = ["c" for _ in ALGS]
        algs_format = " ".join(algs_format)
        f.write("\\begin{tabular}{l | %s}\n" % algs_format)

        # generate header line
        alg_string = " & ".join([a.upper() for a in ALGS])
        f.write("\tTraining time (in s) & %s\\\\ \\toprule\n" % alg_string)

        # write baseline line
        baseline_line = "\t\tBaseline & "
        for a in ALGS:
            i = get_conf_index(conf_list, a, eval_name)
            baseline_line += "$%.2f \\mypm %.2f$ & " % (mean[i], std[i])
        baseline_line = baseline_line[:-2] + "\\\\ \\midrule\n"
        f.write(baseline_line)

        for cur in CURIOSITIES:
            for joint in [False, True]:
                cur_name = cur.upper()
                if joint:
                    cur_name += " (joint)"
                cur_line = "\t%s & " % (cur_name)
                for a in ALGS:
                    c_list = [a, cur]
                    if joint:
                        c_list.insert(2, "joint")
                    conf = "_".join(c_list)
                    i = get_conf_index(conf_list, conf, eval_name)
                    cur_line += "$%.2f \\mypm %.2f$ & " % (mean[i], std[i])
                cur_line = cur_line[:-2] + "\\\\"
                if joint:
                    cur_line += "\\midrule"
                cur_line += "\n"
                f.write(cur_line)

        f.write("\\end{tabular}\n")
        

def generate_time_comparisons(logs_dir, plots_dir, envs, filters_list, table=False, eval_name=None):
    """
    Generate plots and tables for training times of all filter matching configurations
    :param logs_dir: path to logs directory
    :param plots_dir: path to plots directory where plots are stored
    :param envs: list of environments
    :param filters_list: list of filters to apply
    :param table: generate latex table as well
    :param eval_name: evaluation run name for specific runs (for table config matching)
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
        time_list = []
        conf_list = []
        for confs in confs_list:
            for c in confs:
                c_name = conf_to_name(c)
                dirs = run_dirs_by_conf(logs_dir, env, c)
                if not dirs:
                    # no logs for this conf/ env combination
                    continue
                
                print("\t" + str(c))

                times = []
                for d in dirs:
                    for f in os.listdir(d):
                        log_path = os.path.join(d, f)
                        if ".log" in f:
                            with open(log_path, 'r') as log:
                                for line in log.readlines():
                                    time = line[10:-2]
                                    times.append(float(time))
                                    break

                time_list.append(np.array(times))
                conf_list.append(c_name)
        
        # generate plot and tables of time comparisons
        if time_list:
            filters_name = "-".join(["_".join(filters) for filters in filters_list])
            if len(filters_name) > 100:
                filters_name = filters_name[:50] + "__" + filters_name[-50:]
            path = os.path.join(env_dir, "time_comparison_" + filters_name + ".pdf")
            generate_time_comparison_chart(np.array(time_list), env, conf_list, path)

            if table:
                table_path = os.path.join(env_dir, "time_comparison_table_" + filters_name + ".tex")
                generate_time_comparison_table(np.array(time_list), env, conf_list, table_path, eval_name)
        else:
            print("\tNo matching configurations found!")

if __name__ == "__main__":
    logs_dir, plots_dir, envs, filters_list = parse_args(sys.argv)
    generate_time_comparisons(logs_dir, plots_dir, envs, filters_list)
