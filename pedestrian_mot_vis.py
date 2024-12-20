import argparse
import gc
import os
import pickle
import random
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
from numpy import mean, std
from tqdm import tqdm

from bettergym.agents.planner_mcts import Mcts
from bettergym.agents.utils.utils import (
    epsilon_uniform_uniform,
)
from bettergym.agents.utils.vo import (
    epsilon_uniform_uniform_vo,
)
from bettergym.environments.env import EnvConfig
from bettergym.environments.robot_arena import dist_to_goal
from environment_creator import (
    create_pedestrian_env,
)
from experiment_utils import (
    plot_frame2,
    print_and_notify,
    create_animation_tree_trajectory,
)

DEBUG_DATA = False
DEBUG_ANIMATION = False
ANIMATION = True


@dataclass(frozen=True)
class ExperimentData:
    rollout_policy: Callable
    action_expansion_policy: Callable
    discrete: bool
    obstacle_reward: bool
    std_angle: float
    n_sim: int = 1000
    c: float = 150
    vo: bool = False


@njit
def seed_numba(seed_value: int):
    np.random.seed(seed_value)


def seed_everything(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    seed_numba(seed_value)


def filter_obstacles(state):
    x = state.x
    return [o for o in state.obstacles if np.linalg.norm(o.x[:2] - x[:2]) <= 5]


def run_experiment(experiment: ExperimentData, arguments):
    global exp_num
    fig, ax = plt.subplots()
    config = EnvConfig(
        dt=1, max_angle_change=1.9 * 1, n_angles=10, n_vel=5, num_humans=10
    )
    # input [forward speed, yaw_rate]

    with open(f"./debug/obs_0.pkl", "rb") as f:
        obstacles = pickle.load(f)

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(obstacles[0])))
    trajectories = [[] for _ in range(len(obstacles[0]))]
    for step in obstacles:
        for i in range(len(step)):
            trajectories[i].append(step[i].x[:2])

    for idx, c in enumerate(colors):
        trj = np.array(trajectories[idx])
        plt.plot(trj[:, 0], trj[:, 1], c=c)

    print("Creating Gif...")

    ax.set_xlim([config.left_limit - 0.5, config.right_limit + 0.5])
    ax.set_ylim([config.bottom_limit - 0.5, config.upper_limit + 0.5])
    fig.savefig(f"cosa.png", dpi=500, facecolor="white", edgecolor="none")
    # ani = FuncAnimation(
    #     fig,
    #     plot_frame2,
    #     fargs=(goal, config, obs, trajectory, ax),
    #     frames=tqdm(range(len(trajectory)), file=sys.stdout),
    #     save_count=None,
    #     cache_frame_data=False,
    # )
    # ani.save(f"debug/trajectory_{exp_name}_{exp_num}.gif", fps=150)
    # plt.close(fig)



def argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algorithm", default="vanilla", type=str, help="The algorithm to run"
    )
    parser.add_argument(
        "--nsim",
        default=1000,
        type=int,
        help="The number of simulation the algorithm will run",
    )
    parser.add_argument("--rwrd", default=-100, type=int, help="Reward for going out of the map")
    parser.add_argument("--discount", default=0.99, type=float, help="")
    parser.add_argument("--std", default=0.38 * 2, type=float, help="")
    parser.add_argument("--stdRollout", default=0.5, type=float, help="")
    parser.add_argument("--c", default=1, type=float, help="")
    parser.add_argument("--rollout", default="normal_towards_goal", type=str, help="")
    parser.add_argument(
        "--a", default=10, type=int, help="number of discretization of angles"
    )
    parser.add_argument(
        "--v", default=10, type=int, help="number of discretization of velocities"
    )
    parser.add_argument(
        "--num", default=1, type=int, help="number of experiments to run"
    )
    parser.add_argument(
        "--eps_rollout",
        default=0.1,
        type=float,
        help="Percentage of Uniform Rollout in Rollout",
    )
    parser.add_argument(
        "--max_depth",
        default=100,
        type=int,
        help="Maximum Depth of the tree",
    )
    parser.add_argument(
        "--env",
        default="EASY",
        type=str,
        help="Environment",
    )
    parser.add_argument(
        "--n_obs",
        default=40,
        type=int,
        help="Number of Pedestrian in the environment",
    )
    parser.add_argument(
        "--fixed_obs",
        default=True,
        type=bool,
        help="Whether or not to use fixed obstacles",
    )
    return parser


def get_experiment_data(arguments):
    # var_angle = 0.38 * 2
    std_angle_rollout = arguments.stdRollout

    if arguments.rollout == "epsilon_uniform_uniform":
        if arguments.algorithm == "VANILLA_VO2" or arguments.algorithm == "VANILLA_VO_ROLLOUT":
            rollout_policy = partial(
                epsilon_uniform_uniform_vo,
                std_angle_rollout=std_angle_rollout,
                eps=arguments.eps_rollout,
            )
        else:
            rollout_policy = partial(
                epsilon_uniform_uniform,
                std_angle_rollout=std_angle_rollout,
                eps=arguments.eps_rollout,
            )
    else:
        raise ValueError("rollout function not valid")

    if arguments.algorithm == "VANILLA" :
        # VANILLA
        return ExperimentData(
            action_expansion_policy=None,
            rollout_policy=rollout_policy,
            discrete=True,
            obstacle_reward=True,
            std_angle=std_angle_rollout,
            n_sim=arguments.nsim,
            c=arguments.c,
        )
    elif arguments.algorithm == "VANILLA_VO2":
        # VO2
        return ExperimentData(
            vo=True,
            action_expansion_policy=None,
            rollout_policy=rollout_policy,
            discrete=True,
            obstacle_reward=False,
            std_angle=std_angle_rollout,
            n_sim=arguments.nsim,
            c=arguments.c,
        )
    elif arguments.algorithm == "VANILLA_VO_ROLLOUT":
        return ExperimentData(
            action_expansion_policy=None,
            rollout_policy=rollout_policy,
            discrete=True,
            vo=False,
            obstacle_reward=True,
            std_angle=std_angle_rollout,
            n_sim=arguments.nsim,
            c=arguments.c,
        )
    elif arguments.algorithm == "VANILLA_VO_ALBERO":
        return ExperimentData(
            action_expansion_policy=None,
            rollout_policy=rollout_policy,
            discrete=True,
            vo=True,
            obstacle_reward=False,
            std_angle=std_angle_rollout,
            n_sim=arguments.nsim,
            c=arguments.c,
        )


def main():
    global exp_num
    # args = argument_parser().parse_args()
    # exp = get_experiment_data(args)
    seed_everything(1)
    for exp_num in range(1):
        run_experiment(experiment=None, arguments=None)


if __name__ == "__main__":
    main()
