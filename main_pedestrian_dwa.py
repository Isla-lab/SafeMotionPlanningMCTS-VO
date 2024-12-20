import argparse
import gc
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
from numpy import mean, std
from tqdm import tqdm

from bettergym.agents.planner_dwa import Dwa
from bettergym.environments.robot_arena import dist_to_goal
from environment_creator import (
    create_pedestrian_env,
)
from experiment_utils import (
    plot_frame2,
    print_and_notify,
)

DEBUG_DATA = False
DEBUG_ANIMATION = True
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
        out_boundaries_rwrd=True,
        n_vel=5,
        n_angles=11,
        vo=experiment.vo,
        obs_pos=obstacles,
        n_obs=arguments.n_obs,
    )

    s0, _ = real_env.reset()
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    s0.x = np.append(s0.x, 0.0)
    trajectory = np.array(s0.x)
    config = real_env.config
    goal = s0.goal

    s = s0
    s.obstacles = s0.obstacles = [o for o in s.obstacles if o.obs_type != "wall"]
    obs = [s0.obstacles]
    planner = Dwa(environment=sim_env)
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
        initial_time = time.time()
        ob = np.array([ob.x[:2] for ob in s.obstacles])
        u, info = planner.plan(initial_state=s, obs=ob, robot_obs=[])
        u_copy = np.array(u, copy=True)
        final_time = time.time() - initial_time
        # del info['q_values']
        # del info['actions']
        # del info['visits']
        # gc.collect()

        actions.append(u_copy)

        # Clip action
        # u_copy = np.array(u, copy=True)

        # infos.append(deepcopy(info))

        times.append(final_time)
        u_copy[1] = s.x[2] + u[1] * config.dt
        s, r, terminal, truncated, env_info = real_env.step(s, u_copy)
        s.obstacles = [o for o in s.obstacles if o.obs_type != "wall"]
        sim_env.gym_env.state = real_env.gym_env.state.copy()
        rewards.append(r)
        trajectory = np.vstack((trajectory, s.x))  # store state history
        s.x[3] = u[0]
        s.x[4] = u[1]
        obs.append(s.obstacles)
        gc.collect()

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
    discount = 0.7
    data = {
        "cumRwrd": round(sum(rewards), 2),
        "discCumRwrd": round(sum(np.array(rewards) * np.array([discount ** e for e in range(len(rewards))])), 2),
        "nSteps": step_n,
        "MeanStepTime": np.round(mean(times), 2),
        "StdStepTime": np.round(std(times), 2),
        "reachGoal": int(reach_goal),
        "meanSmoothVelocity": np.diff(trajectory[:, 3]).mean(),
        "stdSmoothVelocity": np.diff(trajectory[:, 3]).std(),
        "meanSmoothAngle": np.diff(trajectory[:, 2]).mean(),
        "stdSmoothAngle": np.diff(trajectory[:, 2]).std(),
        **env_info
    }
    data = data | arguments.__dict__
    df = pd.Series(data)
    df.to_csv(f"dwa_{exp_name}_{exp_num}.csv")

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
        ani.save(f"debug/trajectory_{exp_name}_dwa_{exp_num}.gif", fps=150)
        plt.close(fig)


def argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--num", default=1, type=int, help="number of experiments to run"
    )
    parser.add_argument(
        "--horizon", default=80, type=int, help="number of experiments to run"
    )
    parser.add_argument(
        "--fixed_obs",
        default=True,
        type=bool,
        help="Whether or not to use fixed obstacles",
    )
    parser.add_argument(
        "--n_obs",
        default=40,
        type=int,
        help="Number of Pedestrian in the environment",
    )
    return parser


def get_experiment_data(arguments):
    return ExperimentData(
        action_expansion_policy=None,
        rollout_policy=None,
        discrete=True,
        obstacle_reward=True,
        std_angle=None,
        n_sim=None,
        c=None,
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
