# Curiosity in Multi-Agent Reinforcement Learning (MSc Project)

<!-- introduction -->
## Introduction
This is a MSc dissertation project at the University of Edinburgh under supervision of [Stefano Albrecht](http://svalbrecht.de/?page=salbrecht).

<!-- abstract -->
### Abstract
Multi-agent reinforcement learning has seen considerable achievements on a variety of tasks. However, suboptimal conditions involving sparse feedback and partial observability, as frequently encountered in applications, remain a signiﬁcant challenge. In this thesis, we apply curiosity as exploration bonuses to such multi-agent systems and analyse their impact on a variety of cooperative and competitive tasks. In addition, we consider modiﬁed scenarios involving sparse rewards and partial observability to evaluate the inﬂuence of curiosity on these challenges.

We apply the independent Q-learning and state-of-the-art multi-agent deep deterministic policy gradient methods to these tasks with and without intrinsic rewards. Curiosity is deﬁned using pseudo-counts of observations or relying on models to predict environment dynamics.

Our evaluation illustrates that intrinsic rewards can cause considerable instability in training without beneﬁting exploration. This outcome can be observed on the original tasks and against our expectation under partial observability, where curiosity is unable to alleviate the introduced instability. However, curiosity leads to significantly improved stability and converged performance when applied to policy-gradient reinforcement learning with sparse rewards. While the sparsity causes training of such methods to be highly unstable, additional intrinsic rewards assist training and agents show intended behaviour on most tasks.

This work contributes to understanding the impact of intrinsic rewards in challenging multi-agent reinforcement learning environments and will serve as a foundation for further research to expand on.

--- 

More information can be found in [the MSc dissertation](assets/thesis.pdf].

<!-- requirements-->

## Dependencies

* [Python](https://www.python.org), version 3.7
* [Numpy](https://numpy.org), version 1.15
* [Pytorch](https://pytorch.org), version 1.1.0
* [Matplotlib](https://matplotlib.org), version 3.0.3
* [OpenAI Gym](https://gym.openai.com), version 0.10.5
* Own [fork](https://github.com/LukasSchaefer/multiagent-particle-envs) of Multi-agent Particle Environments adding partial observability and stochasticity to four tasks

<!-- training -->
## Training
The general training structure is implemented in `train.py`. `mape_train.py` contains the specific code to train on the multi-agent particle envirionment.

For information on all paramters, run
```
python3 mape_train.py --help
```.

<!-- evaluation -->
## Evaluation
Similarly, the evaluation structure is implemented in `eval.py` with specific multi-agent particle environment evaluation found in `mape_eval.py`.

For information on all paramters, run
```
python3 mape_eval.py --help
```.

The training and evaluation script mostly share parameters. The major difference are the generally deactived exploration in evaluation, activated rendering and added support to save animated gifs of evaluation runs (`--save_gifs`).

<!-- baselines -->
## Baselines
As multi-agent reinforcement learning baselines, we implement the following approaches:

* [Independent Q-learning (IQL)](`marl_algorithms/iql/README.md`) using deep Q-networks (DQNs)
* [Multi-agent deep deterministic policy gradient (MADDPG)](`marl_algorithms/maddpg/README.md`)

Baseline implementations can be found in `marl_algorithms`, which also includes an episodic buffer (`marl_algorithms/buffer.py`) and an abstract MARL class (`marl_algorithms/marl_algorithms`). Detailed READMEs for [IQL](`marl_algorithms/iql/README.md`) and [MADDPG](`marl_algorithms/maddpg/README.md`) with references to papers and open-source implementations can be found in the respective subdiretories.

<!-- curiosity -->
## Curiosity
We implement three variations of intrinsic rewards as exploration bonuses, which can be found in `intrinsic_rewards` with an abstract intrinsic reward interface (`intrinsic_rewards/intrinsic_reward.py`).

* [Hash-count-based curiosity](`intrinsic_rewards/count_based_bonus/readme.md`)
* [Intrinsic curiosity module (ICM)](`intrinsic_rewards/icm/readme.md`)
* [Random network distillation (RND)](`intrinsic_rewards/rnd/readme.md`)

Detailed, linked READMEs can be found in the respective subdirectories of each curiosity approach.

<!-- evironments -->
## Environments
We evaluate our approaches on the multi-agent particle environment. Instead of using the [original environment](https://github.com/openai/multiagent-particle-envs), we implemented a [fork](https://github.com/LukasSchaefer/multiagent-particle-envs) introducing partial observability and stochasticity to the tasks

* cooperative communication (`simple_speaker_listener`)
* cooperative navigation (`simple_spread`)
* physical deception (`simple_adversary`)
* predator-prey (`simple_tag`)

For more detail on the added partial observability and stochasticity, see the respective sections of the [README in our fork](https://github.com/LukasSchaefer/multiagent-particle-envs).

<!-- experiments -->
## Running experiments
Experiment scripts can easily be generated using the `script_generation/script_generator.py` script with the respective environment name.
At the moment only the multi-agent particle environment is supported. Parameters for the jobscript generation can be chosen in the code lines 152 to 164. Afterwards

```
python3 script_generator.py mape
```
will generate a directory `mape` containing a subdirectory with jobscripts for each scenario as well as a central jobscript `mape/script.sh` which executes all scripts consecutively. Hence, only this script has to be executed to run all generated jobs.


<!-- citing -->
## Citing

```bibtex
@MastersThesis{lukas:thesis:2019,
    author     =     {Schäfer, Lukas},
    title     =     {{Curiosity in Multi-Agent Reinforcement Learning}},
    school     =     {University of Edinburgh},
    year     =     {2019},
}
```
<!-- contact -->
## Contact
Lukas Schäfer - <s1874970@ed.ac.uk>
