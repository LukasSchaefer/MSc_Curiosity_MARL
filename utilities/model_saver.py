import os

import torch


class ModelSaver:
    """
    Class to save model parameters
    """

    def __init__(self, save_models_dir="models", run_name="default", alg_name="maddpg"):
        self.save_models_dir = save_models_dir
        self.run_name = run_name
        self.alg_name = alg_name

    def clear_models(self):
        """
        Remove model files in model dir
        """
        if not os.path.isdir(self.save_models_dir):
            return
        alg_dir = os.path.join(self.save_models_dir, self.alg_name)
        if not os.path.isdir(alg_dir):
            return
        model_dir = os.path.join(alg_dir, self.run_name)
        if not os.path.isdir(model_dir):
            return
        for f in os.listdir(model_dir):
            f_path = os.path.join(model_dir, f)
            if not os.path.isfile(f_path):
                continue
            os.remove(f_path)

    def save_models(self, alg, extension):
        """
        generate and save networks
        :param model_dir_path: path of model directory
        :param run_name: name of run
        :param alg_name: name of used algorithm
        :param alg: training object of trained algorithm
        :param extension: name extension
        """
        if not os.path.isdir(self.save_models_dir):
            os.mkdir(self.save_models_dir)
        alg_dir = os.path.join(self.save_models_dir, self.alg_name)
        if not os.path.isdir(alg_dir):
            os.mkdir(alg_dir)
        model_dir = os.path.join(alg_dir, self.run_name)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        if self.alg_name == "maddpg":
            for i, agent in enumerate(alg.agents):
                name_actor = "maddpg_agent%d_actor_params_" % i
                name_actor += extension
                name_critic = "maddpg_agent%d_critic_params_" % i
                name_critic += extension
                torch.save(agent.actor.state_dict(), os.path.join(model_dir, name_actor))
                torch.save(agent.critic.state_dict(), os.path.join(model_dir, name_critic))
        elif self.alg_name == "iql":
            for i, agent in enumerate(alg.agents):
                name = "iql_agent%d_params_" % i
                name += extension
                torch.save(agent.model.state_dict(), os.path.join(model_dir, name))
        else:
            raise ValueError("Unknown algorithm to save models for: " + self.alg_name)
