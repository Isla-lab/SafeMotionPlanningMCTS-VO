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

from bettergym.agents.planner_mcts import RolloutStateNode
from bettergym.agents.utils.vo import (
    epsilon_uniform_uniform_vo,
)
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
    new_obstacles = []
    for o in state.obstacles:
        if o.obs_type == "wall":
            new_obstacles.append(o)
        elif np.linalg.norm(o.x[:2] - x[:2]) <= 5:
            new_obstacles.append(o)
            
    return new_obstacles


class RolloutPlanner:
    def __init__(self, rollout_policy, environment):
        self.rollout_policy = rollout_policy
        self.environment = environment

    def plan(self, state):
        return self.rollout_policy(RolloutStateNode(state), self), None


def run_experiment(experiment: ExperimentData, arguments):
    global exp_num
    # input [forward speed, yaw_rate]
    if arguments.fixed_obs:
        behaviour = "intention"
        with open(f"./bettergym/environments/fixed_obs/{behaviour}/{arguments.n_obs}/obs_{exp_num}.pkl", "rb") as f:
            obstacles = pickle.load(f)
    else:
        obstacles = None

    real_env, sim_env = create_pedestrian_env(
        discrete=experiment.discrete,
        rwrd_in_sim=experiment.obstacle_reward,
        out_boundaries_rwrd=arguments.rwrd,
        n_vel=arguments.v,
        n_angles=arguments.a,
        vo=experiment.vo,
        obs_pos=obstacles,
        n_obs=arguments.n_obs,
    )

    s0, _ = real_env.reset()
    trajectory = np.array(s0.x)
    config = real_env.config

    goal = s0.goal

    s = s0
    obs = [[o for o in filter_obstacles(s0) if o.obs_type != "wall"]]

    planner = RolloutPlanner(
        environment=sim_env,
        rollout_policy=experiment.rollout_policy,
    )
    print("Simulation Started")
    terminal = False
    rewards = []
    times = []
    infos = []
    actions = []
    step_n = 0
    while not terminal:
        step_n += 1
        if step_n == 1000:
            break
        print(f"Step Number {step_n}")
        s_copy = deepcopy(s)
        # s_copy.obstacles = filter_obstacles(s_copy)
        initial_time = time.time()
        u, info = planner.plan(s_copy)
        final_time = time.time() - initial_time
        actions.append(u)
        u_copy = np.array(u, copy=True)
        infos.append(deepcopy(info))
        # del info['q_values']
        # del info['actions']
        # del info['visits']

        times.append(final_time)
        s, r, terminal, truncated, env_info = real_env.step(s, u_copy)
        sim_env.gym_env.state = real_env.gym_env.state.copy()
        rewards.append(r)
        trajectory = np.vstack((trajectory, s.x))  # store state history
        obs.append([o for o in s_copy.obstacles if o.obs_type != "wall"])
        gc.collect()
    arguments.algorithm = "RolloutPlanner"
    exp_name = "_".join([k + ":" + str(v) for k, v in arguments.__dict__.items()])
    print_and_notify(
        f"Simulation Ended with Reward: {round(sum(rewards), 2)}\n"
        f"Discrete: {experiment.discrete}\n"
        f"Std Rollout Angle: {experiment.std_angle}\n"
        f"Number of Steps: {step_n}\n"
        f"Avg Reward Step: {round(sum(rewards) / step_n, 2)}\n"
        f"Avg Step Time: {np.round(mean(times), 2)}±{np.round(std(times), 2)}\n"
        f"Total Time: {sum(times)}\n"
        f"Num Simulations: {experiment.n_sim}",
        exp_num,
        exp_name,
    )

    dist_goal = dist_to_goal(s.x[:2], s.goal)
    reach_goal = dist_goal <= real_env.config.robot_radius
    discount = arguments.discount
    data = {
        "cumRwrd": round(sum(rewards), 2),
        "discCumRwrd": round(sum(np.array(rewards) * np.array([discount ** e for e in range(len(rewards))])), 2),
        "nSteps": step_n,
        "MeanStepTime": np.round(mean(times), 2),
        "StdStepTime": np.round(std(times), 2),
        "reachGoal": int(reach_goal),
        "maxNsteps": int(step_n == 1000),
        "meanSmoothVelocity": np.diff(trajectory[:, 3]).mean(),
        "stdSmoothVelocity": np.diff(trajectory[:, 3]).std(),
        "meanSmoothAngle": np.diff(trajectory[:, 2]).mean(),
        "stdSmoothAngle": np.diff(trajectory[:, 2]).std(),
        **env_info
    }
    data = data | arguments.__dict__
    df = pd.Series(data)
    df.to_csv(f"{exp_name}_{exp_num}.csv")

    if ANIMATION:
        print("Creating Gif...")
        fig, ax = plt.subplots()
        ani = FuncAnimation(
            fig,
            plot_frame2,
            fargs=(goal, config, obs, trajectory, ax),
            frames=tqdm(range(len(trajectory)), file=sys.stdout),
            save_count=None,
            cache_frame_data=False,
        )
        ani.save(f"debug/trajectory_{exp_name}_{exp_num}.gif", fps=150)
        plt.close(fig)

    # trajectories = [i["trajectories"] for i in infos]
    # rollout_values = [i["rollout_values"] for i in infos]

    with open(f"debug/trajectory_real_{exp_name}_{exp_num}.pkl", "wb") as f:
        pickle.dump(trajectory, f)


    gc.collect()
    print("Done")


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

    rollout_policy = partial(
        epsilon_uniform_uniform_vo,
        std_angle_rollout=std_angle_rollout,
        eps=arguments.eps_rollout,
    )

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


def main():
    global exp_num
    args = argument_parser().parse_args()
    exp = get_experiment_data(args)
    seed_everything(1)
    for exp_num in range(args.num):
        run_experiment(experiment=exp, arguments=args)


if __name__ == "__main__":
    main()
