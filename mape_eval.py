import time

import numpy as np

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

from eval import Eval


class MAPEEval(Eval):
    """
    Evaluation environment for the Multi-Agent Particle Environment (MAPE)

    paper:
    Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, O. P., & Mordatch, I. (2017).
    Multi-agent actor-critic for mixed cooperative-competitive environments.
    In Advances in Neural Information Processing Systems (pp. 6379-6390).

    Link: http://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments
    Open-source: https://github.com/openai/multiagent-particle-envs
    """

    def __init__(self):
        """
        Create MAPE Eval instance
        """
        super(MAPEEval, self).__init__()

    def parse_args(self):
        """
        parse own arguments including default args and mape specific args
        """
        self.parse_default_args()
        self.parser.add_argument(
            "--scenario", type=str, default="simple", help="name of the mape scenario"
        )
        self.parser.add_argument(
            "--partial_observable",
            action="store_true",
            default=False,
            help="use partial observable scenarios",
        )
        self.parser.add_argument(
            "--observation_noise",
            action="store_true",
            default=False,
            help="add Gaussian noise to observations",
        )
        self.parser.add_argument(
            "--environment_noise",
            action="store_true",
            default=False,
            help="add distortion field to environment which adds noise to close agent observations",
        )

    def create_environment(self):
        """
        Create environment instance
        :return: environment (gym interface), env_name, task_name, n_agents, observation_sizes,
                 action_sizes, discrete_actions
        """
        # load scenario from script
        if self.arglist.partial_observable:
            scenario = scenarios.load(
                self.arglist.scenario + "_partial_observable.py"
            ).POScenario()
        elif self.arglist.observation_noise:
            scenario = scenarios.load(self.arglist.scenario + "_observation_noise.py").ONScenario()
        elif self.arglist.environment_noise:
            scenario = scenarios.load(self.arglist.scenario + "_env_noise.py").ENScenario()
        else:
            scenario = scenarios.load(self.arglist.scenario + ".py").Scenario()

        # create world
        world = scenario.make_world()
        # create multiagent environment
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

        env_name = "mape"
        task_name = "mape_" + self.arglist.scenario

        n_agents = env.n
        print("Observation spaces: ", [env.observation_space[i] for i in range(n_agents)])
        print("Action spaces: ", [env.action_space[i] for i in range(n_agents)])
        observation_sizes = self.extract_sizes(env.observation_space)
        action_sizes = self.extract_sizes(env.action_space)
        discrete_actions = True

        return (
            env,
            env_name,
            task_name,
            n_agents,
            observation_sizes,
            action_sizes,
            discrete_actions,
        )

    def reset_environment(self):
        """
        Reset environment for new episode
        :return: observation (as torch tensor)
        """
        obs = self.env.reset()
        obs = [np.expand_dims(o, axis=0) for o in obs]
        return obs

    def select_actions(self, obs):
        """
        Select actions for agents
        :param obs: joint observations for agents
        :return: action_tensor, action_list
        """
        # get actions as torch Variables
        torch_agent_actions = self.alg.step(obs, True)
        # convert actions to numpy arrays
        agent_actions = [ac.data.numpy() for ac in torch_agent_actions]

        return agent_actions, agent_actions

    def environment_step(self, actions):
        """
        Take step in the environment
        :param actions: actions to apply for each agent
        :return: reward, done, next_obs (as Pytorch tensors)
        """
        # environment step
        next_obs, reward, done, _ = self.env.step(actions)
        next_obs = [np.expand_dims(o, axis=0) for o in next_obs]
        return reward, done, next_obs

    def environment_render(self):
        """
        Render visualisation of environment
        """
        self.env.render()
        time.sleep(0.1)


if __name__ == "__main__":
    ev = MAPEEval()
    ev.eval()
