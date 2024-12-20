import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation

# from notify_run import Notify
#
# notify = Notify()
count = 0


def plot_robot(x, y, yaw, config, ax, color="b"):
    circle = plt.Circle((x, y), config.robot_radius, color=color)
    ax.add_artist(circle)
    # out_x, out_y = (
    #         np.array([x, y]) + np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius
    # )
    # ax.plot([x, out_x], [y, out_y], "-k")


def plot_frame(i, goal, config, obs, traj, ax):
    x = traj[i, :]
    # ob = config.ob
    ax.clear()
    # ROBOT POSITION
    # ax.plot(x[0], x[1], "xr")
    # GOAL POSITION
    ax.plot(goal[0], goal[1], "xb")
    # OBSTACLES
    for ob in obs[0]:
        circle = plt.Circle((ob.x[0], ob.x[1]), ob.radius, color="k")
        ax.add_artist(circle)
    # BOX AROUND ROBOT
    plot_robot(x[0], x[1], None, config, ax)
    # TRAJECTORY
    sub_traj = traj[:i]
    ax.plot(sub_traj[:, 0], sub_traj[:, 1], "--r")

    # ax.plot([70, 70], [100, 250], 'k-', lw=2)

    ax.set_xlim([config.left_limit, config.right_limit])
    ax.set_ylim([config.bottom_limit, config.upper_limit])
    # ax.axis("equal")
    ax.grid(True)
    # plt.savefig(f"debug/{i}.png", dpi=500, facecolor="white", edgecolor="none")


def plot_frame2(i, goal, config, obs, traj, ax):
    x = traj[i, :]
    # ob = config.ob
    ax.clear()
    # ROBOT POSITION
    ax.plot(x[0], x[1], "xr")
    # GOAL POSITION
    ax.plot(goal[0], goal[1], "xb")
    # OBSTACLES
    for ob in obs[i]:
        circle = plt.Circle((ob.x[0], ob.x[1]), ob.radius, color="k")
        ax.add_artist(circle)
    # BOX AROUND ROBOT
    plot_robot(x[0], x[1], None, config, ax)
    # TRAJECTORY
    sub_traj = traj[:i]
    ax.plot(sub_traj[:, 0], sub_traj[:, 1], "--r")

    # ax.plot([70, 70], [100, 250], 'k-', lw=2)

    ax.set_xlim([config.left_limit - 0.5, config.right_limit + 0.5])
    ax.set_ylim([config.bottom_limit - 0.5, config.upper_limit + 0.5])
    # ax.axis("equal")
    # ax.grid(True)
    # plt.savefig(f"debug/{i}.png", dpi=500, facecolor="white", edgecolor="none")

def plot_action_evolution(actions: np.ndarray, exp_num: int):
    def plot(data):
        fig, axs = plt.subplots(2, sharex=True)
        sns.lineplot(
            data=data, x=data.index, y="Linear Velocity", ax=axs[0], color="#4c72b0"
        )
        sns.lineplot(
            data=data, x=data.index, y="Angular Velocity", ax=axs[1], color="#c44e52"
        )
        fig.savefig(f"debug/action_evolution_{len(data)}_e{exp_num}.svg")

    sns.set_theme()
    df = pd.DataFrame(
        {"Linear Velocity": actions[:, 1], "Angular Velocity": actions[:, 0]}
    )
    plot(df.iloc[:100])
    plot(df.iloc[:200])
    plot(df)
    np.save(f"debug/actions_{exp_num}", actions)
    sns.reset_orig()


def print_and_notify(message: str, exp_num: int, exp_name: str):
    print(message)
    # notify.send(message)
    # with open(f"debug/{exp_name}_{exp_num}.txt", "w") as f:
    #     f.write(message)


def plot_real_trajectory_information(trj: np.ndarray, exp_num: int):
    sns.set_theme()
    sns.set_palette(sns.color_palette())

    x_vals = trj[:, 0]
    y_vals = trj[:, 1]
    angles = trj[:, 2]
    lin_vel = trj[:, 3]

    # X
    plt.clf()
    sns.lineplot(x=range(len(x_vals)), y=x_vals)
    plt.xlabel("Step")
    plt.ylabel("X")
    plt.savefig(f"debug/X_{exp_num}.svg", dpi=300)

    # Y
    plt.clf()
    sns.lineplot(x=range(len(y_vals)), y=y_vals)
    plt.xlabel("Step")
    plt.ylabel("Y")
    plt.savefig(f"debug/Y_{exp_num}.svg", dpi=300)

    # Lin Vel
    plt.clf()
    sns.lineplot(x=range(len(lin_vel)), y=lin_vel)
    plt.xlabel("Step")
    plt.ylabel("Linear Velocity")
    plt.savefig(f"debug/Lin Vel_{exp_num}.svg", dpi=300)

    # Angles
    plt.clf()
    sns.lineplot(x=range(len(angles)), y=angles)
    plt.xlabel("Step")
    plt.ylabel("Angles")
    plt.savefig(f"debug/Angles_{exp_num}.svg", dpi=300)
    sns.reset_orig()


def plot_frame_tree_traj(i, goal, config, obs, trajectories, values, fig):
    fig.clear()
    ax = fig.add_subplot()
    step = trajectories[i]
    val_points = values[i]

    last_points = np.array([trj[-1][:2] for trj in step])
    x0 = step[0][0]

    ax.cla()
    ax.set_xlim([config.left_limit, config.right_limit])
    ax.set_ylim([config.bottom_limit, config.upper_limit])
    ax.grid(True)

    # ROBOT POSITION
    ax.plot(x0[0], x0[1], "xr")
    # GOAL POSITION
    ax.plot(goal[0], goal[1], "xb")
    # OBSTACLES
    for ob in obs[i]:
        circle = plt.Circle((ob.x[0], ob.x[1]), ob.radius, color="k")
        ax.add_artist(circle)

    for trj in step:
        # last_points_trj = trj[:-1][:, :2]
        ax.plot(trj[:, 0], trj[:, 1], "r--", alpha=0.5)
    cmap = ax.scatter(last_points[:, 0], last_points[:, 1], c=val_points, marker="x")
    plt.colorbar(cmap)
    # plt.savefig(f"debug/{i}.png", dpi=500, facecolor="white", edgecolor="none")


def plot_frame_tree_traj_wsteps(i, goal, config, obs, trajectories, values, fig):
    fig.clear()
    ax = fig.add_subplot()
    step = trajectories[i]
    val_points = values[i]

    last_points = np.array([trj[-1][:2] for trj in step])

    x0 = step[0][0]

    ax.cla()
    ax.set_xlim([config.left_limit, config.right_limit])
    ax.set_ylim([config.bottom_limit, config.upper_limit])
    ax.grid(True)

    # OBSTACLES
    for ob in obs[i]:
        circle = plt.Circle((ob.x[0], ob.x[1]), ob.radius, color="k")
        ax.add_artist(circle)

    for trj in step:
        last_points_trj = trj[:-1][:, :2]
        ax.plot(last_points_trj[:, 0], last_points_trj[:, 1], "r--", alpha=0.5)
    cmap = ax.scatter(last_points[:, 0], last_points[:, 1], c=val_points, marker="x")

    # ROBOT POSITION
    ax.plot(x0[0], x0[1], "xr")
    # GOAL POSITION
    ax.plot(goal[0], goal[1], "xb")

    plt.colorbar(cmap)


def create_animation_tree_trajectory(
        goal, config, obs, exp_num, exp_name, values, trajectories
):
    fig, ax = plt.subplots()
    ani = FuncAnimation(
        fig,
        plot_frame_tree_traj,
        fargs=(goal, config, obs, trajectories, values, fig),
        frames=len(trajectories),
        # blit=True,
        save_count=None,
        cache_frame_data=False,
    )
    ani.save(f"./debug/tree_trajectory_{exp_name}_{exp_num}.mp4", fps=5, dpi=300)
    plt.close(fig)


def create_animation_tree_trajectory_w_steps(goal, config, obs, exp_num):
    with open(f"./debug/trajectories_{exp_num}.pkl", "rb") as f:
        trajectories = pickle.load(f)
    with open(f"./debug/rollout_values_{exp_num}.pkl", "rb") as f:
        values = pickle.load(f)
    fig, ax = plt.subplots()
    ani = FuncAnimation(
        fig,
        plot_frame_tree_traj_wsteps,
        fargs=(goal, config, obs, trajectories, values, fig),
        frames=len(trajectories),
    )
    ani.save(f"./debug/tree_trajectory_steps_{exp_num}.mp4", fps=5, dpi=300)


def plot_frame_multiagent(i, goal1, goal2, config, obs, traj1, traj2, ax):
    x1 = traj1[i, :]
    x2 = traj2[i, :]
    # ob = config.ob
    ax.clear()
    # ROBOT1 POSITION
    ax.plot(x1[0], x1[1], "xr")
    # ROBOT2 POSITION
    ax.plot(x2[0], x2[1], "xr")
    # GOAL1 POSITION
    ax.plot(goal1[0], goal1[1], "xb")
    # GOAL2 POSITION
    ax.plot(goal2[0], goal2[1], "xb")
    # OBSTACLES
    for ob in obs[i][:-1]:
        circle = plt.Circle((ob.x[0], ob.x[1]), ob.radius, color="k")
        ax.add_artist(circle)
    # CIRCLE AROUND ROBOT1
    plot_robot(x1[0], x1[1], x1[2], config, ax, color="b")
    # CIRCLE AROUND ROBOT2
    plot_robot(x2[0], x2[1], x2[2], config, ax, color="g")
    # TRAJECTORY1
    sub_traj1 = traj1[:i]
    ax.plot(sub_traj1[:, 0], sub_traj1[:, 1], "--r")

    # TRAJECTORY
    sub_traj2 = traj2[:i]
    ax.plot(sub_traj2[:, 0], sub_traj2[:, 1], "--b")

    ax.set_xlim([config.left_limit, config.right_limit])
    ax.set_ylim([config.bottom_limit, config.upper_limit])
    ax.grid(True)


def plot_frame_no_obs(i, goals, config, trajectories, ax):
    x = [traj[i, :] for traj in trajectories]
    colors = ["m", "b", "g", "y"]
    # ob = config.ob
    ax.clear()
    for idx in range(len(x)):
        # ROBOT POSITION
        ax.plot(x[idx][0], x[idx][1], "xr")
        # GOAL POSITION
        ax.plot(goals[idx][0], goals[idx][1], "xb")
        # CIRCLE AROUND ROBOT
        plot_robot(x[idx][0], x[idx][1], x[idx][2], config, ax, color=colors[idx])
        # TRAJECTORY1
        sub_traj = trajectories[idx][:i]
        ax.plot(sub_traj[:, 0], sub_traj[:, 1], f"--{colors[idx]}")

    ax.set_xlim([config.left_limit, config.right_limit])
    ax.set_ylim([config.bottom_limit, config.upper_limit])
    ax.grid(True)


def plot_frame_obs(i, goals, config, trajectories, ax, obs):
    x = trajectories[i, :, :]
    colors = ["m", "b", "g", "y", 'c', 'r', 'bisque', 'olive']
    # ob = config.ob
    ax.clear()
    for idx in range(len(x)):
        # ROBOT POSITION
        ax.plot(x[idx][0], x[idx][1], "xr")
        # GOAL POSITION
        ax.plot(goals[idx][0], goals[idx][1], "xb")
        # CIRCLE AROUND ROBOT
        plot_robot(x[idx][0], x[idx][1], None, config, ax, color=colors[idx])
        # TRAJECTORY1
        # sub_traj = trajectories[idx][:i]
        sub_traj = trajectories[:i, idx, :]
        ax.plot(sub_traj[:, 0], sub_traj[:, 1], f"--", color=colors[idx])

    # STATIC OBSTACLES
    for ob in obs:
        circle = plt.Circle((ob.x[0], ob.x[1]), ob.radius, color="k")
        ax.add_artist(circle)

    ax.set_xlim([config.left_limit, config.right_limit])
    ax.set_ylim([config.bottom_limit, config.upper_limit])
    ax.grid(True)
