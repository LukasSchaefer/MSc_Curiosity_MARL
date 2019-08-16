from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import os

rc("text", usetex=True)
rc("font", family="serif", size=28)
rc("axes", labelsize=28)


COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

def plt_axis():
    """
    Plt setup with grid and axis
    """
    # adjust axis ticks
    plt.xticks(fontsize=12, ha="center", va="top")
    plt.yticks(fontsize=12, ha="right", va="center")
    # tick parameters
    plt.tick_params(
        axis="both",
        bottom=True,
        top=True,
        left=True,
        right=True,
        direction="in",
        which="major",
        grid_color="blue",
    )
    # define grid
    plt.grid(linestyle="--", linewidth=0.5, alpha=0.15)

def plt_legend():
    """
    Plot transparent legend without duplicate labels
    """
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fancybox=True, framealpha=0.5, fontsize=12)

def plt_shade(y_values, x_values=None, y_variances=None, color=COLORS[0], label=None, alpha=0.1):
    """
    Plot shaded curve
    :param y_values: list of lists of values for y-axis
    :param x_values: list of x_values
    :param y_variances: list of lists of variances for y-axis shading
    :param color: color value for plot
    :param label: label of plot
    :param alpha: transparency level
    """
    means = np.array(y_values).mean(0)
    if y_variances is not None:
        stds = np.sqrt(np.array(y_variances).mean(0))
    else:
        stds = np.array(y_values).std(0)
    if x_values is None:
        x_values = np.arange(0, len(means))
    if label is None:
        plt.plot(x_values, means, "-", linewidth=1, c=color)
    else:
        plt.plot(x_values, means, "-", linewidth=1, c=color, label=label)
    plt.fill_between(
        x_values,
        means - stds,
        means + stds,
        alpha=alpha,
        color=color,
        antialiased=True,
    )

def generate_plot(title, x_label, y_label, x_values, y_values, y_variances=None, y_min=None, y_max=None):
    """
    Generate shading plot
    :param x_label: label for x-axis
    :param y_label: label for y-axis
    :param x_values: x-values with shape (num_seeds, num_values)
    :param y_values: values to plot, shape (num_seeds, num_values, num_players)
    :param y_variances: values to shade with, shape (num_seeds, num_values, num_players)
    :param y_min: minimum y-axis value
    :param y_max: maximum y-axis value
    """
    plt.clf()
    plt_axis()
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    axes = plt.gca()
    if y_min is not None and y_max is not None:
        axes.set_ylim([y_min, y_max])
    plt.xticks(np.arange(0, 30000, 5000))

    n_agents = y_values.shape[-1]
    for i in range(n_agents):
        if y_variances is None:
            plt_shade(y_values[:,:,i], x_values, color=COLORS[i], label="agent %d" % (i + 1))
        else:
            plt_shade(
                y_values[:,:,i],
                x_values,
                y_variances[:,:,i],
                color=COLORS[i],
                label="agent %d" % (i + 1)
            )
    plt_legend()

def save_plot(dir, file_name):
    """
    Save given figure - Stored in <plot_dir>/<alg>/<run>/<file_name>.pdf
    :param dir: directory to store file in
    :param file_name: name of file
    """
    file_path = os.path.join(dir, file_name + ".pdf")
    plt.savefig(file_path, format="pdf")

def generate_reward_plot(c_dir, episodes, rewards, variances=None, intrinsic=False, y_min=None, y_max=None):
    """
    Generate and save rewards plot
    :param c_dir: directory where plot should be saved
    :param episodes: numpy array of episode numbers (num_seeds, num_rewards)
    :param rewards: numpy array of rewards (num_seeds, num_rewards, num_players)
    :param variances: numpy array of reward variances (num_seeds, num_rewards, num_players)
    :param intrinsic: flag if intrinsic rewards
    :param y_min: minimum y-axis value
    :param y_max: maximum y-axis value
    """
    if intrinsic:
        title = r"Intrinsic Rewards"
    else:
        title = r"Rewards"

    if variances is not None:
        generate_plot(title, r"Episode", r"Rewards", episodes, rewards * 25, variances * 25**2)
    else:
        generate_plot(title, r"Episode", r"Rewards", episodes, rewards * 25, variances)

    if intrinsic:
        save_plot(c_dir, 'intrinsic_rewards_summary')
    else:
        save_plot(c_dir, 'rewards_summary')

def generate_exploration_plot(c_dir, values, alg_name):
    """
    Generate and save exploration plot
    :param c_dir: directory where plot should be saved
    :param values: numpy array of (num_seeds, num_values)
    :param alg_name: name of algorithm ("maddpg" or "iql")
    """
    plt.clf()
    plt_axis()
    if alg_name == "maddpg":
        plt.title(r"Exploration Variance", fontsize=14)
    elif alg_name == "iql":
        plt.title(r"Exploration Epsilon", fontsize=14)
    plt.xlabel(r"Episode", fontsize=12)
    if alg_name == "maddpg":
        plt.ylabel(r"Variance", fontsize=12)
    elif alg_name == "iql":
        plt.ylabel(r"Epsilon", fontsize=12)

    plt_shade(values)
    plt_legend()

    save_plot(c_dir, 'exploration_summary')
    
def generate_alg_loss_plot(c_dir, episodes, losses, alg_name):
    """
    Generate and save algorithm loss plot
    :param c_dir: directory where plot should be saved
    :param episodes: episode values
    :param losses: numpy array of (num_seeds, num_values, ?, num_players)
        where ? corresponds to 1 for iql and 2 for maddpg
    :param alg_name: name of algorithm ("maddpg" or "iql")
    """
    if alg_name == "maddpg":
        critic_losses = losses[:,:,0,:]
        actor_losses = losses[:,:,1,:]
        # critic loss plot
        generate_plot(r"MADDPG Critic Loss", r"Episodes", r"Loss", episodes, critic_losses)
        save_plot(c_dir, 'maddpg_criticloss_summary')
        # actor loss plot
        generate_plot(r"MADDPG Actor Loss", r"Episodes", r"Loss", episodes, actor_losses)
        save_plot(c_dir, 'maddpg_actorloss_summary')
    elif alg_name == "iql":
        q_losses = losses
        generate_plot(r"IQL Q-Loss", r"Episodes", r"Loss", episodes, q_losses)
        save_plot(c_dir, 'iql_qloss_loss_summary')

def generate_cur_loss_plot(c_dir, episodes, losses, cur_name):
    """
    Generate and save curiosity loss plot
    :param c_dir: directory where plot should be saved
    :param episodes: episode values
    :param losses: numpy array of (num_seeds, num_values, ?, num_players)
        where ? corresponds to 1 for iql and 2 for maddpg
    :param cur_name: name of curiosity ("count", "icm" or "rnd")
    """
    if cur_name == "count":
        count_values = losses
        generate_plot(r"Intrinsic Count Value", r"Episodes", r"Count values", episodes, count_values)
        save_plot(c_dir, 'countvalue_summary')
    elif cur_name == "icm":
        forward_losses = losses[:,:,:,0]
        inverse_losses = losses[:,:,:,1]
        # plot ICM forward losses
        generate_plot(r"ICM Forward Loss", r"Episodes", r"Loss", episodes, forward_losses)
        save_plot(c_dir, 'icm_forwardloss_summary')
        # plot ICM inverse losses
        generate_plot(r"ICM Inverse Loss", r"Episodes", r"Loss", episodes, inverse_losses)
        save_plot(c_dir, 'icm_inverseloss_summary')
    if cur_name == "rnd":
        forward_losses = losses
        generate_plot(r"RND Forward Loss", r"Episodes", r"Loss", episodes, forward_losses)
        save_plot(c_dir, 'rnd_forwardloss_summary')
