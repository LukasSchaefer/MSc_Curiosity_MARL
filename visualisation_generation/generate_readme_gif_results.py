import sys
import os

from plotter_time_comparison import ENV_NAMES, ALGS, CURIOSITIES
from plotter_summary import SCENARIOS
from generate_gifs import CONDITIONS

def generate_gif_row(path):
    """
    generate gif row text to write to readme
    :param path: path to gif folder
    :return: string of gif row line
    """
    line = ""
    for i in range(0, 3):
        s = os.path.join(path, "ep_%d.gif" % i)
        s += "?raw=true"
        line += '<img src="%s" width="32%%"> ' % s
    return line

def main(argv):
    """
    Generate markdown file of result gifs
    :param argv: argument list
    """
    if len(argv) != 3:
        print("Usage: ./generate_readme_gif_results.py <path/to/gifs_evaluation/dir> <path/to/results.md")
        sys.exit(1)

    gifs_dir = os.path.abspath(argv[1])
    if not os.path.isdir(gifs_dir):
        print("Invalid gifs results directory: %s" % gifs_dir)
        sys.exit(1)
    results_path = os.path.abspath(argv[2])
    if not os.path.isfile(results_path):
        open(results_path, 'w').close()
    
    results_file = open(results_path, 'w')

    # generate results md file
    results_file.write("## Results\n")
    results_file.write("In the following, we show animated GIFs that show the performance after training for various tasks and configurations.\n\n")
    
    for condition in CONDITIONS:
        full_condition = "mape"
        if condition != "":
            suffix = "_" + condition
            full_condition += "_" + condition
        else:
            suffix = ""

        cond_path = os.path.join(gifs_dir, full_condition)

        if full_condition == "mape":
            header = "Original Multi-Agent Particle Environment"
            subheader = "First, we show the performance of baselines on the original multi-agent particle environment tasks under full observability, trained with usual, continuous rewards."
        elif full_condition == "mape_partialobservable":
            header = "Multi-Agent Particle Environment under Partial Observability"
            subheader = "Now, we show the performance of baselines and some curiosity on multi-agent particle environment tasks under partial observability, trained with continuous rewards."
        elif full_condition == "mape_sparserewards":
            header = "Multi-Agent Particle Environment with Sparse Rewards"
            subheader = "Lastly, we show the performance of baselines and some curiosity on multi-agent particle environment tasks trained with sparse rewards (under full observability)"

        results_file.write("### %s\n" % header)
        results_file.write(subheader + "\n\n")

        for scenario in SCENARIOS:
            results_file.write("#### %s (`%s`)\n" % (ENV_NAMES[scenario], scenario))
            for alg in ALGS:
                results_file.write("* %s:\n" % alg.upper())
                for cur in [None, "count"]:
                    if cur is None:
                        cur_name = "baseline"
                        conf_name = alg
                    elif cur == "count":
                        cur_name = "with joint count-based curiosity"
                        conf_name = alg + "_count"
                    elif cur == "icm":
                        cur_name = "with joint ICM"
                        conf_name = alg + "_icm"
                    elif cur == "rnd":
                        cur_name = "with joint RND"
                        conf_name = alg + "_rnd"
                    results_file.write("\t* %s:\n\n" % cur_name)
                    conf_dir = os.path.join(cond_path, conf_name)
                    scenario_dir = os.path.join(conf_dir, scenario)
                    seed_dir = os.path.join(scenario_dir, "seed1")

                    gif_line = generate_gif_row(seed_dir)
                    results_file.write(gif_line + "\n\n")


    results_file.close()


if __name__ == "__main__":
    main(sys.argv)
