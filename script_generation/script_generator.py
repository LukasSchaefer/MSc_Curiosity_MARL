import os
import sys


def generate_script(
    repo_dir,
    output_dir,
    path,
    id,
    env,
    map,
    scenario,
    alg,
    cur,
    joint_curiosity,
    curiosity_state_rep_size,
    count_key_dim,
    eta,
    seed,
    lr,
    curiosity_lr,
    dropout_p,
    partial_observable,
    sparse_rewards,
    no_rewards,
    no_exploration
):
    """
    Generate shell script for given parameters
    :param repo_dir: path to reposity directory
    :param repo_dir: path to output directory
    :param path: path to directory where to generate jobscripts
    :param id: job id
    :param env: environment to be used ("smac" or "mape")
    :param map: smac map to be used
    :param scenario: mape scenario to be used
    :param alg: algorithm to be used (None, "maddpg" or "iql")
    :param cur: curiosity to be used ("icm" or "rnd")
    :param joint_curiosity: flag if curiosity should be used jointly (only for maddpg)
    :param curiosity_state_rep_size: state representation size of curiosity
    :param count_key_dim: key dimensionality of hash-count-based curiosity
    :param eta: curiosity scaling factor
    :param seed: random seed to be used (None for no fixed seed)
    :param lr: learning rate for RL algorithm
    :param curiosity_lr: learning rate for curiosity approach
    :param dropout_p: dropout prob to be used in RL algorithm
    :param partial_observable: use partial observable environments
    :param sparse_rewards: use sparse rewards only
    :param no_rewards: don't use extrinsic reward
    :param no_exploration: don't use any exploration (besides curiosity)
    """
    if env == "mape":
        assert(scenario is not None)
        run_name = env + "_" + scenario
    else:
        assert(map is not None)
        run_name = env + "_" + map
    run_name += "_seed" + str(seed) + "_" + alg
    if cur is not None:
        run_name += "_" + str(cur)
    if joint_curiosity:
        run_name += "_joint"
    if partial_observable:
        run_name += "_partialobservable"
    if sparse_rewards:
        run_name += "_sparserewards"
    if no_rewards:
        run_name += "_norewards"
    if no_exploration:
        run_name += "_noexploration"
    run_name += "_id" + str(id)

    job_name = "job_" + run_name
    job_path = os.path.join(path, job_name + ".sh")
    output_path = os.path.join(output_dir, job_name + ".log")
    with open(job_path, "w") as job_file:
        parameter_chain = ""
        parameter_chain += "--alg=%s " % alg
        if cur is not None:
            parameter_chain += "--curiosity=%s " % cur
        if joint_curiosity:
            parameter_chain += "--joint_curiosity "
        if curiosity_state_rep_size is not None:
            parameter_chain += "--curiosity_state_rep_size=%s " % curiosity_state_rep_size
        if count_key_dim is not None:
            parameter_chain += "--count_key_dim=%s " % count_key_dim
        if eta is not None:
            parameter_chain += "--eta=%d " % eta
        if seed is not None:
            parameter_chain += "--seed=%d " % seed
        if lr is not None:
            parameter_chain += "--lr=%f " % lr
        if curiosity_lr is not None:
            parameter_chain += "--curiosity_lr=%f " % curiosity_lr
        if dropout_p is not None:
            parameter_chain += "--dropout_p=%f " % dropout_p
        if partial_observable:
            parameter_chain += "--partial_observable "
        if sparse_rewards:
            parameter_chain += "--sparse_rewards "
        if no_rewards:
            parameter_chain += "--no_rewards "
        if no_exploration:
            parameter_chain += "--no_exploration "

        parameter_chain += "--run=%s" % run_name

        job_file.write("cd " + repo_dir + "\n")
        pipe_output =  " > " + output_path + " 2>&1"
        if env == "mape":
            job_file.write("python3 mape_train.py --scenario=" + str(scenario) + " " + parameter_chain + pipe_output + "\n")
        elif env == "smac":
            job_file.write("python3 smac_train.py --map=" + str(map) + " " + parameter_chain + pipe_output + "\n")
        job_file.write("cd " + path + "\n")

    return job_path

def main(argv):
    if len(argv) != 2:
        print("Usage: ./job_generator.py <env_name>")
        sys.exit(1)

    current_dir_path = os.path.dirname(os.path.realpath(__file__))

    env_name = argv[1]
    jobs_dir = current_dir_path + "/" + env_name
    if not os.path.isdir(env_name):
        os.mkdir(env_name)

    separator_character = "/"
    repo_dir_list = current_dir_path.split("/")
    for i, d in enumerate(repo_dir_list):
        if d == "Curiosity_MARL":
            repo_dir_list = repo_dir_list[:i+1]
            break
    repo_dir = separator_character.join(repo_dir_list)

    outputs_dir = os.path.join(repo_dir, "script_outputs")
    if not os.path.isdir(outputs_dir):
        os.mkdir(outputs_dir)
    outputs_dir = os.path.join(outputs_dir, env_name)
    if not os.path.isdir(outputs_dir):
        os.mkdir(outputs_dir)

    if env_name == "mape":
        scenarios = ["simple_adversary"] #["simple_speaker_listener", "simple_spread", "simple_tag", "simple_adversary"]
        maps = [None] * len(scenarios)
    elif env_name == "smac":
        maps = ["3m", "8m", "3s5z", "8m_vs_9m", "27m_vs_30m"]
        scenarios = [None] * len(maps)

    algs = ["iql"]# ["maddpg", "iql"]
    curs = [None] #None, "icm", "rnd", "count"]
    joint_curiositys = [False]#[True, False]
    curiosity_state_rep_size = None
    count_key_dim = None
    etas = [5]
    curiosity_lrs = [1e-5] #[2e-6]
    seeds = [2, 3]
    dropout_ps = [0.0]
    partial_observable = False
    sparse_rewards = True
    no_rewards = False
    no_exploration = False

    id = 0

    print("start generation")
    scripts = []
    for scenario, map in zip(scenarios, maps):
        if scenario is not None:
            job_dir = os.path.join(jobs_dir, scenario)
            output_dir = os.path.join(outputs_dir, scenario)
        else:
            job_dir = os.path.join(jobs_dir, map)
            output_dir = os.path.join(outputs_dir, map)
        if not os.path.isdir(job_dir):
            os.mkdir(job_dir)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for alg in algs:
            for cur in curs:
                for joint_curiosity in joint_curiositys:
                    if joint_curiosity and cur is None:
                        continue
                    if alg == "maddpg":
                        lr = 1e-2
                    elif alg == "iql":
                        lr = 1e-3
                    for eta in etas:
                        for curiosity_lr in curiosity_lrs:
                            for dropout_p in dropout_ps:
                                for seed in seeds:
                                    script_path = generate_script(
                                        repo_dir,
                                        output_dir,
                                        job_dir,
                                        id,
                                        env_name,
                                        map,
                                        scenario,
                                        alg,
                                        cur,
                                        joint_curiosity,
                                        curiosity_state_rep_size,
                                        count_key_dim,
                                        eta,
                                        seed,
                                        lr,
                                        curiosity_lr,
                                        dropout_p,
                                        partial_observable,
                                        sparse_rewards,
                                        no_rewards,
                                        no_exploration
                                    )
                                    id += 1
                                    scripts.append(script_path)
    print("%d jobs generated." % id)

    script_path = os.path.join(env_name, "script.sh")
    with open(script_path, "w") as f:
        for script in scripts:
            f.write("chmod +x " + script + "\n")
            f.write(script + "\n")

if __name__ == "__main__":
    main(sys.argv)
