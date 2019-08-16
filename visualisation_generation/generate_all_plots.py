import sys
import os

from plotter_summary import parse_args as parse_summary_args
from plotter_summary import generate_plot_summaries
from plotter_time_comparison import parse_args as parse_time_args
from plotter_time_comparison import generate_time_comparisons
from plotter_reward_comparison import parse_args as parse_reward_args
from plotter_reward_comparison import generate_reward_comparisons
from plotter_intreward_comparison import parse_args as parse_intreward_args
from plotter_intreward_comparison import generate_intreward_comparisons

ALGS = ["maddpg", "iql"]
CURIOSITIES = ["count", "icm", "rnd"]

def main(argv):
    """
    Generate all plots:
        - summary of plots
        - generate reward comparisons of
            - MADDPG and IQL as baselines
            - MADDPG/ IQL with joint & individual curiosity for each curiosity
            ONLY PARTIAL OBSERVABILITY - MADDPG/ IQL with joint curiosity under and without partial observability for each curiosity
            - MADDPG/ IQL / MADDPG joint curiosity / IQL joint curiosity for each curiosity
        - generate intrinsic reward comparisons of
            - MADDPG and IQL with joint & individual curiosity for each curiosity
        - generate time comparison of
            - MADDPG / COUNT / ICM / RND (+ all joints)
            - IQL / COUNT / ICM / RND (+ all joints)
    :param argv: argument list with only path to evaluation directory
    """
    if len(argv) != 2:
        print("Usage: ./generate_all_plots.py <path/to/evaluation/dir")
        sys.exit(1)

    eval_dir = os.path.abspath(argv[1])
    eval_name = None
    if eval_dir.split("/")[-1].split("_")[-1] != "mape":
        eval_name = "_".join(eval_dir.split("/")[-1].split("_")[1:])
    logs_dir = os.path.join(eval_dir, "logs")
    if not os.path.isdir(logs_dir):
        print("Evaluation directory does not contain a logs directory!")
        sys.exit(1)

    # generate summary of plots
    print("GENERATE PLOT SUMMARIES:\n")
    sum_dir = os.path.join(eval_dir, "plot_summary")
    if not os.path.isdir(sum_dir):
        os.mkdir(sum_dir)

    summary_args = ["./plotter_summary.py", logs_dir, sum_dir, "ALL"]
    logs_dir, plots_dir, envs, filters = parse_summary_args(summary_args)
    generate_plot_summaries(logs_dir, plots_dir, envs, filters)
    print("\n")
    print("\n")

    # generate reward comparisons
    print("GENERATE REWARD COMPARISONS:\n")
    rew_dir = os.path.join(eval_dir, "reward_comparison")
    if not os.path.isdir(rew_dir):
        os.mkdir(rew_dir)

    # reward comparison of baselines
    print("GENERATE REWARD COMPARISONS FOR BASELINES:")
    rew_base_args = ["./plotter_reward_comparison", logs_dir, rew_dir, "ALL"]
    rew_alg_args = []
    for alg in ALGS:
        if eval_name is not None:
            rew_alg_args.append([alg, eval_name])
        else:
            rew_alg_args.append([alg])
        rew_alg_args = [str(l).replace(" ", "").replace("'","") for l in rew_alg_args]
    rew_args = rew_base_args + rew_alg_args
    logs_dir, plots_dir, envs, filters = parse_reward_args(rew_args)
    generate_reward_comparisons(logs_dir, plots_dir, envs, filters)
    print("\n")

    # reward comparison of alg, cur, joint cur for each alg/ cur combination
    print("GENERATE REWARD COMPARISONS FOR ALG/ CUR COMBINATIONS:")
    for alg in ALGS:
        for cur in CURIOSITIES:
            print("%s, %s:\n" % (alg, cur))
            rew_cur_args = [[alg], [alg, cur], [alg, cur, "joint"]]
            if eval_name is not None:
                rew_cur_args = [args + [eval_name] for args in rew_cur_args]
            rew_cur_args = [str(l).replace(" ", "").replace("'","") for l in rew_cur_args]
            rew_args = rew_base_args + rew_cur_args
            logs_dir, plots_dir, envs, filters = parse_reward_args(rew_args)
            generate_reward_comparisons(logs_dir, plots_dir, envs, filters)
            print()
    print("\n")

    if eval_name == "partialobservable" or eval_name == "sparserewards":
        # reward comparison of alg, joint cur for each alg/ cur combination with baselines
        print("GENERATE REWARD COMPARISONS FOR ALG/ CUR COMBINATIONS compared with baselines:")
        for alg in ALGS:
            for cur in CURIOSITIES:
                print("%s, %s:\n" % (alg, cur))
                rew_cur_args = [[alg], [alg, eval_name], [alg, cur, "joint", eval_name]]
                rew_cur_args = [str(l).replace(" ", "").replace("'","") for l in rew_cur_args]
                rew_args = rew_base_args + rew_cur_args
                logs_dir, plots_dir, envs, filters = parse_reward_args(rew_args)
                generate_reward_comparisons(logs_dir, plots_dir, envs, filters)
                print()
        print("\n")

    # reward comparison among MADDPG and IQL (with/out cur)
    print("GENERATE REWARD COMPARISONS FOR EACH CUR:")
    for cur in CURIOSITIES:
        print(cur + ":\n")
        rew_cur_args = [["maddpg"], ["iql"], ["maddpg", cur, "joint"], ["iql", cur, "joint"]]
        if eval_name is not None:
            rew_cur_args = [args + [eval_name] for args in rew_cur_args]
        rew_cur_args = [str(l).replace(" ", "").replace("'","") for l in rew_cur_args]
        rew_args = rew_base_args + rew_cur_args
        logs_dir, plots_dir, envs, filters = parse_reward_args(rew_args)
        generate_reward_comparisons(logs_dir, plots_dir, envs, filters)
        print()
    print("\n")
    print("\n")

    # generate intrinsic reward comparisons
    print("GENERATE INTRINSIC REWARD COMPARISONS:\n")
    intrew_dir = os.path.join(eval_dir, "intreward_comparison")
    if not os.path.isdir(intrew_dir):
        os.mkdir(intrew_dir)

    # intrinsic reward comparison among MADDPG and IQL (with/out cur)
    print("GENERATE INTRINSIC REWARD COMPARISONS FOR EACH CUR:")
    intrew_base_args = ["./plotter_intreward_comparison", logs_dir, intrew_dir, "ALL"]
    for cur in CURIOSITIES:
        print(cur + ":\n")
        intrew_cur_args = []
        for alg in ALGS:
            intrew_cur_args.append([alg, cur])
            intrew_cur_args.append([alg, cur, "joint"])
        if eval_name is not None:
            intrew_cur_args = [args + [eval_name] for args in intrew_cur_args]
        intrew_cur_args = [str(l).replace(" ", "").replace("'","") for l in intrew_cur_args]
        intrew_args = intrew_base_args + intrew_cur_args
        logs_dir, plots_dir, envs, filters = parse_intreward_args(intrew_args)
        generate_intreward_comparisons(logs_dir, plots_dir, envs, filters)
        print()
    print("\n")
    print("\n")

    # generate time comparison
    print("GENERATE TIME COMPARISONS:\n")
    time_dir = os.path.join(eval_dir, "time_comparison")
    if not os.path.isdir(time_dir):
        os.mkdir(time_dir)

    time_base_args = ["./plotter_time_comparison", logs_dir, time_dir, "ALL"]
    print("GENERATE TIME COMPARISONS FOR EACH ALG:")
    for alg in ALGS:
        filter_args = [[alg]]
        for cur in CURIOSITIES:
            filter_args.append([alg, cur])
            filter_args.append([alg, cur, "joint"])
        if eval_name is not None:
            filter_args = [args + [eval_name] for args in filter_args]
        filter_args = [str(l).replace(" ", "").replace("'","") for l in filter_args]
        time_args = time_base_args + filter_args
        logs_dir, plots_dir, envs, filters = parse_time_args(time_args)
        generate_time_comparisons(logs_dir, plots_dir, envs, filters, False, eval_name)
        print()

    print("\nGENERATE TIME COMPARISONS FOR ALL WITH TABLES:")
    filter_args = []
    for alg in ALGS:
        filter_args.append([alg])
        for cur in CURIOSITIES:
            filter_args.append([alg, cur])
            filter_args.append([alg, cur, "joint"])
    if eval_name is not None:
        filter_args = [args + [eval_name] for args in filter_args]
    filter_args = [str(l).replace(" ", "").replace("'","") for l in filter_args]
    time_args = time_base_args + filter_args
    logs_dir, plots_dir, envs, filters = parse_time_args(time_args)
    generate_time_comparisons(logs_dir, plots_dir, envs, filters, True, eval_name)



if __name__ == "__main__":
    main(sys.argv)
