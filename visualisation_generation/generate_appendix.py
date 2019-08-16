import sys
import os

from plotter_time_comparison import ENV_NAMES, ALGS, CURIOSITIES
from plotter_summary import SCENARIOS

confs = ["", "partialobservable", "sparserewards"]

def generate_miniplot(f, fig_path, name, width=0.5, intend=""):
    """
    Generate latex miniplot for given figure and write to file
    :param f: file to write to
    :param fig_path: path of figure to include
    :param name: name of figure for label
    :param width: width of textwidth to use
    :param intend: intendation of every line
    """
    f.write(intend + "\\begin{minipage}{%.2f\\linewidth}\n" % width)
    f.write(intend + "\t\\centering\n")
    f.write(intend + "\t\\includegraphics[width=.95\\textwidth]{%s}\n" % fig_path)
    f.write(intend + "\t\\label{fig:%s}\n" % name)
    f.write(intend + "\\end{minipage}\n")

def main(argv):
    """
    Generates appendix
        Rewards:
        for all confs generate
        - for each task:
            - for each alg:
                - for each curiosity:
                    rewards of alg with cur with joint cur

        for partial observability only:
        - for each task:
            - for each alg:
                - for each cur:
                    rewards of alg with/out joint cur compared to baseline without partial observability

        Intrinsic rewards:
        for all confs generate
        - for each task:
            - for each curiosity:
                intrinsic rewards of all algs with individual and joint cur
        Times:
        for each task:
            time table
    :param argv: argument list
    """
    if len(argv) != 3:
        print("Usage: ./generate_appendix.py <path/to/evaluation_results/dir> <path/to/appendix/dir>")
        sys.exit(1)

    eval_res_dir = os.path.abspath(argv[1])
    if not os.path.isdir(eval_res_dir):
        print("Invalid evaluation results directory: %s" % eval_res_dir)
    appendix_dir = os.path.abspath(argv[2])
    if not os.path.isdir(appendix_dir):
        print("Invalid appendix directory: %s" % appendix_dir)
    
    eval_res_path = os.path.join(appendix_dir, "eval_res_appendix.tex")
    if os.path.isfile(eval_res_path):
        os.remove(eval_res_path)
    eval_res_file = open(eval_res_path, 'w')

    # generate reward appendix
    eval_res_file.write("\\section{Episodic Rewards}\n")
    
    for conf in confs:
        full_conf = "mape"
        if conf != "":
            suffix = "_" + conf
            full_conf += "_" + conf
        else:
            suffix = ""

        if full_conf == "mape":
            conf_name = "Multi-Agent Particle Environment"
        elif full_conf == "mape_partialobservable":
            conf_name = "Multi-Agent Particle Environment under Partial Observability"
        elif full_conf == "mape_sparserewards":
            conf_name = "Multi-Agent Particle Environment with Sparse Rewards"

        conf_path = os.path.join(eval_res_dir, full_conf)

        eval_res_file.write("\\subsection{%s}\n" % conf_name)

        first_task = True
        for task in SCENARIOS:
            task_name = ENV_NAMES[task]
            task_name_suffix = task_name.lower().replace(" ", "_") + suffix
            if not first_task:
                eval_res_file.write("\\newpage\n")
            first_task = False
            eval_res_file.write("\\subsubsection{%s}\n" % task_name)
            eval_res_file.write("\\begin{figure}[h]\n")
            for cur in CURIOSITIES:
                for alg in ALGS:
                    fig_path = os.path.join(conf_path, "reward_comparison")
                    fig_path = os.path.join(fig_path, task)
                    fig_name = "reward_comparison_"
                    filter_name = "%s-%s-%s" % (alg + suffix, alg + "_" + cur + suffix, alg + "_" + cur + "_joint" + suffix)
                    if len(filter_name) > 100:
                        filter_name = filter_name[:50] + "__" + filter_name[-50:]
                    fig_name += filter_name + ".pdf"
                    fig_path = os.path.join(fig_path, fig_name)
                    name = "appendix_reward_%s_%s_%s" % (alg, cur, task_name_suffix)
                    generate_miniplot(eval_res_file, fig_path, name, 0.5, "\t")
                    if not (cur == CURIOSITIES[-1] and alg == ALGS[-1]):
                        eval_res_file.write("\t\\hfill\n")
            caption = "Episodic rewards for MADDPG (left column) and IQL (right column) with individual and joint curiosities for %s task" % task_name.lower()
            if full_conf == "mape":           
                caption += "."
            elif full_conf == "mape_partialobservable":
                caption += " under partial observability."
            elif full_conf == "mape_sparserewards":
                caption += " with sparse rewards."
            eval_res_file.write("\t\\caption{%s}\n" % caption)
            eval_res_file.write("\t\\label{fig:appendix_reward_%s}\n" % task_name_suffix)
            eval_res_file.write("\\end{figure}\n")
            eval_res_file.write("\n\\FloatBarrier\n\\newpage\n\n")

            if full_conf == "mape_partialobservable" or full_conf == "mape_sparserewards":
                eval_res_file.write("\\begin{figure}[h]\n")
                for cur in CURIOSITIES:
                    for alg in ALGS:
                        fig_path = os.path.join(conf_path, "reward_comparison")
                        fig_path = os.path.join(fig_path, task)
                        fig_name = "reward_comparison_"
                        filter_name = "%s-%s-%s" % (alg, alg + suffix, alg + "_" + cur + "_joint" + suffix)
                        if len(filter_name) > 100:
                            filter_name = filter_name[:50] + "__" + filter_name[-50:]
                        fig_name += filter_name + ".pdf"
                        fig_path = os.path.join(fig_path, fig_name)
                        name = "appendix_reward_%s_%s_%s_vs_baseline" % (alg, cur, task_name_suffix)
                        generate_miniplot(eval_res_file, fig_path, name, 0.5, "\t")
                        if not (cur == CURIOSITIES[-1] and alg == ALGS[-1]):
                            eval_res_file.write("\t\\hfill\n")
                caption = "Episodic rewards for MADDPG (left column) and IQL (right column) with joint curiosities for %s task with and without " % task_name.lower()
                if full_conf == "mape_partialobservable":
                    caption += "partial observability."
                elif full_conf == "mape_sparserewards":
                    caption += "sparse rewards."
                eval_res_file.write("\t\\caption{%s}\n" % caption)
                eval_res_file.write("\t\\label{fig:appendix_reward_%s_vs_baseline}\n" % task_name_suffix)
                eval_res_file.write("\\end{figure}\n")
                eval_res_file.write("\n\\FloatBarrier\n\\newpage\n\n")


    # generate intrinsic reward appendix
    eval_res_file.write("\\section{Intrinsic Rewards}\n")
    
    for conf in confs:
        full_conf = "mape"
        if conf != "":
            suffix = "_" + conf
            full_conf += "_" + conf
        else:
            suffix = ""

        if full_conf == "mape":
            conf_name = "Multi-Agent Particle Environment"
        elif full_conf == "mape_partialobservable":
            conf_name = "Multi-Agent Particle Environment under Partial Observability"
        elif full_conf == "mape_sparserewards":
            conf_name = "Multi-Agent Particle Environment with Sparse Rewards"

        conf_path = os.path.join(eval_res_dir, full_conf)

        eval_res_file.write("\\subsection{%s}\n" % conf_name)

        column = 0
        eval_res_file.write("\\begin{figure}[h]\n")
        for task in SCENARIOS:
            task_name = ENV_NAMES[task]
            task_name_suffix = task_name.lower().replace(" ", "_") + suffix
            for cur in CURIOSITIES:
                fig_path = os.path.join(conf_path, "intreward_comparison")
                fig_path = os.path.join(fig_path, task)
                file_name = "intreward_comparison_"
                filter_name = ""
                for alg in ALGS:
                    filter_name += alg + "_" + cur + suffix + "-"
                    filter_name += alg + "_" + cur + "_joint" + suffix + "-"
                filter_name = filter_name[:-1]
                if len(filter_name) > 100:
                    filter_name = filter_name[:50] + "__" + filter_name[-50:]
                file_name = file_name + filter_name + ".pdf"
                fig_path = os.path.join(fig_path, file_name)
                name = "appendix_intreward_%s_ind_joint_%s" % (cur, task_name_suffix)
                generate_miniplot(eval_res_file, fig_path, name, 0.32, "\t")
                if column != 2:
                    eval_res_file.write("\t\\hfill\n")
                column = (column + 1) % 3
        caption = "Intrinsic rewards for MADDPG and IQL with individual and joint curiosities for "
        for i, task in enumerate(SCENARIOS):
            if i == 0:
                caption += ENV_NAMES[task].lower() + " (1st row)"
            elif i == 1:
                caption += ENV_NAMES[task].lower() + " (2nd row)"
            elif i == 2:
                caption += ENV_NAMES[task].lower() + " (3rd row)"
            elif i == 3:
                caption += ENV_NAMES[task].lower() + " (4th row)"
            else:
                caption += ENV_NAMES[task].lower() + " (%dth row)" % (i + 1)
            if i != len(SCENARIOS) - 2:
                caption += ", "
            else:
                caption += " and "
        caption = caption[:-2] + " task"
        if full_conf == "mape":           
            caption += "."
        elif full_conf == "mape_partialobservable":
            caption += " under partial observability."
        elif full_conf == "mape_sparserewards":
            caption += " with sparse rewards."
        eval_res_file.write("\t\\caption{%s}\n" % caption)
        eval_res_file.write("\t\\label{fig:appendix_intreward_%s}\n" % full_conf)
        eval_res_file.write("\\end{figure}\n\n")
        eval_res_file.write("\\FloatBarrier\n\\newpage\n\n")

    # generate time tables appendix
    eval_res_file.write("\\section{Training Time}\n")
    
    for conf in confs:
        full_conf = "mape"
        if conf != "":
            suffix = "_" + conf
            full_conf += "_" + conf
        else:
            suffix = ""

        if full_conf == "mape":
            conf_name = "Multi-Agent Particle Environment"
        elif full_conf == "mape_partialobservable":
            conf_name = "Multi-Agent Particle Environment under Partial Observability"
        elif full_conf == "mape_sparserewards":
            conf_name = "Multi-Agent Particle Environment with Sparse Rewards"

        conf_path = os.path.join(eval_res_dir, full_conf)

        eval_res_file.write("\\subsection{%s}\n" % conf_name)

        first_column = True
        eval_res_file.write("\\begin{table}[h]\n")
        for i, task in enumerate(SCENARIOS):
            task_name = ENV_NAMES[task]
            eval_res_file.write("\t\\begin{minipage}{.5\linewidth}\n")
            eval_res_file.write("\t\t\\centering\n")
            eval_res_file.write("\t\t\\resizebox{.9\\textwidth}{!}{\n")

            # find name of table
            table_path = os.path.join(conf_path, "time_comparison")
            table_path = os.path.join(table_path, task)
            table_name = "time_comparison_table_"
            filter_name = ""
            for alg in ALGS:
                filter_name += alg + suffix + "-"
                for cur in CURIOSITIES:
                    filter_name += alg + "_" + cur + suffix + "-"
                    filter_name += alg + "_" + cur + "_joint" + suffix + "-"
            filter_name = filter_name[:-1]
            if len(filter_name) > 100:
                filter_name = filter_name[:50] + "__" + filter_name[-50:]
            table_name += filter_name + ".tex"
            table_path = os.path.join(table_path, table_name)

            eval_res_file.write("\t\t\t\\input{%s}\n" % table_path)
            eval_res_file.write("\t\t}\n")
            eval_res_file.write("\t\\end{minipage}\n")
            if first_column:
                eval_res_file.write("\t\\hfill\n")
            elif i != len(SCENARIOS) - 1:
                eval_res_file.write("\t\\mbox{} \\vspace{1cm}\n\n")
            first_column = not first_column

        # determine caption
        caption = "Training times for MADDPG and IQL with individual and joint curiosities for "
        for i, task in enumerate(SCENARIOS):
            caption += ENV_NAMES[task].lower()
            if i == 0:
                caption += " (top left)"
            elif i == 1:
                caption += " (top right)"
            elif i == 2:
                caption += " (bottom left)"
            elif i == 3:
                caption += " (bottom right)"
            if i != len(SCENARIOS) - 2:
                caption += ", "
            else:
                caption += " and "
        caption = caption[:-2] + " task"
        if full_conf == "mape": 
            caption += "."
        elif full_conf == "mape_partialobservable":
            caption += " under partial observability."
        elif full_conf == "mape_sparserewards":
            caption += " with sparse rewards."
        eval_res_file.write("\t\\caption{%s}\n" % caption)
        eval_res_file.write("\t\\label{tab:appendix_timetable_%s}\n" % full_conf)
        eval_res_file.write("\\end{table}\n")
        eval_res_file.write("\n\\FloatBarrier\n\\newpage\n\n")

    eval_res_file.close()


if __name__ == "__main__":
    main(sys.argv)
