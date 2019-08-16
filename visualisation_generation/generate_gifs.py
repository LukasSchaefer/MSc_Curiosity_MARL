import os
import subprocess
import sys

from plotter_time_comparison import ENV_NAMES, ALGS, CURIOSITIES
from plotter_summary import SCENARIOS

CONDITIONS = ["", "partialobservable", "sparserewards"]

def main(argv):
    """
    Generate gifs for all trained configurations
    :param argv: argument list
    """
    if len(argv) != 3:
        print("Usage: ./generate_gifs.py <path/to/evaluation_results/dir> <path/to/gifs/dir>")
        sys.exit(1)

    eval_res_dir = os.path.abspath(argv[1])
    if not os.path.isdir(eval_res_dir):
        print("Invalid evaluation results directory: %s" % eval_res_dir)

    gifs_dir = os.path.abspath(argv[2])
    if not os.path.isdir(gifs_dir):
        os.mkdir(gifs_dir)
    
    for condition in CONDITIONS:
        full_condition = "mape"
        if condition != "":
            suffix = "_" + condition
            full_condition += "_" + condition
        else:
            suffix = ""

        condition_dir = os.path.join(eval_res_dir, full_condition)
        model_dir = os.path.join(condition_dir, "models")
        if not os.path.isdir(model_dir):
            print("Invalid model directory: %s" % model_dir)

        gif_condition_dir = os.path.join(gifs_dir, full_condition)
        if not os.path.isdir(gif_condition_dir):
            os.mkdir(gif_condition_dir)

        default_params = "--num_episodes=3 --save_gifs --joint_curiosity"
        if condition == "partialobservable":
            default_params += " --partial_observable"
        for alg in ALGS:
            params = default_params + " --alg=" + alg
            alg_model_dir = os.path.join(model_dir, alg)
            for cur in [None, "count", "icm", "rnd"]:
                if cur is not None:
                    params += " --curiosity=" + cur
                if cur is not None:
                    gif_conf_dir = os.path.join(gif_condition_dir, alg + "_" + cur)
                else:
                    gif_conf_dir = os.path.join(gif_condition_dir, alg)
                if not os.path.isdir(gif_conf_dir):
                    os.mkdir(gif_conf_dir)
                for scenario in SCENARIOS:
                    params += " --scenario=" + scenario
                    gif_scenario_dir = os.path.join(gif_conf_dir, scenario)
                    if not os.path.isdir(gif_scenario_dir):
                        os.mkdir(gif_scenario_dir)
                    for seed in range(1,2):
                        gif_seed_dir = os.path.join(gif_scenario_dir, "seed" + str(seed))
                        params += " --seed=" + str(seed)
                        if cur is not None:
                            conf = "mape_" + scenario + "_seed" + str(seed) + "_" + alg + "_" + cur + "_joint" + suffix
                        else:
                            conf = "mape_" + scenario + "_seed" + str(seed) + "_" + alg + suffix
                        params += " --run=" + conf

                        # search model path (with id at end)
                        model_path = None
                        for d in os.listdir(alg_model_dir):
                            if d.startswith(conf):
                                model_path = os.path.join(alg_model_dir, d)
                        if model_path is None:
                            continue
                        
                        # execute evaluation
                        process = ["python3", "../mape_eval.py"] + params.split()
                        print("Executing: %s" % " ".join(process))
                        subprocess.call(process)

                        # copy gifs
                        path = "gifs/" + alg + "/" + conf

                        process = ["cp", "-r", os.path.abspath(path), gif_seed_dir]
                        print("Executing: %s" % " ".join(process))
                        subprocess.call(process)


if __name__ == "__main__":
    main(sys.argv)
