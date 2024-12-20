import pickle

import numpy as np
from matplotlib import pyplot as plt

from environment_creator import create_env_five_small_obs_continuous, create_env_4ag


def plot_base(goal, obs, config):
    # START POSITION
    plt.plot(1, 1, "xk")
    # GOAL POSITION
    plt.plot(goal[0], goal[1], "xk")
    # OBSTACLES
    for ob in obs:
        circle = plt.Circle((ob.x[0], ob.x[1]), ob.radius, color="k")
        plt.gcf().gca().add_artist(circle)
    plt.gcf().gca().set_xlim([config.left_limit, config.right_limit])
    plt.gcf().gca().set_ylim([config.bottom_limit, config.upper_limit])
    # ax.axis("equal")
    plt.grid(True)


def plot_base_four_obs(initial_states, config):
    for s in initial_states:
        # START POSITION
        plt.plot(s.x[0], s.x[1], "xk")
    plt.gcf().gca().set_xlim([config.left_limit, config.right_limit])
    plt.gcf().gca().set_ylim([config.bottom_limit, config.upper_limit])
    # ax.axis("equal")
    plt.grid(True)


def plot_trj(traj, color, label):
    # TRAJECTORY
    plt.plot(traj[:, 0], traj[:, 1], f"--{color}", label=label)


def one_ag():
    # 1 AGENT
    BASE_PATH = "debug/TRAJECTORIES/1AG"
    path_vanilla = (
        BASE_PATH
        + "/"
        + "trajectory_real_algorithm:VANILLA_nsim:20_rwrd:-100_dt:1.0_std:0.5_stdRollout:1.0_amplitude:1_c:10.0_rollout:epsilon_uniform_uniform_alpha:0.1_k:50.0_a:11_v:5_num:1_eps_rollout:0.2_max_depth:100_env:EASY_start:corner_0.pkl"
    )
    path_vo2 = (
        BASE_PATH
        + "/"
        + "trajectory_real_algorithm:VANILLA_VO2_nsim:10_rwrd:-100_dt:1.0_std:0.5_stdRollout:1.0_amplitude:1_c:10.0_rollout:epsilon_uniform_uniform_alpha:0.1_k:50.0_a:11_v:5_num:1_eps_rollout:0.2_max_depth:100_env:EASY_start:corner_0.pkl"
    )
    path_dwa = BASE_PATH + "/" + "trajectory_real_dwa.pkl"
    path_mpc = BASE_PATH + "/" + "treajectoryMPC.pkl"
    with open(path_vanilla, "rb") as f:
        trj_vanilla = pickle.load(f)
    with open(path_vo2, "rb") as f:
        trj_vo2 = pickle.load(f)
    with open(path_dwa, "rb") as f:
        trj_dwa = pickle.load(f)
    with open(path_mpc, "rb") as f:
        trj_mpc = pickle.load(f)

    trj_mpc = np.hstack((np.array([1., 1.]).reshape(2, -1), trj_mpc[:2])).T
    real_env, sim_env = create_env_five_small_obs_continuous(
        initial_pos=(1, 1),
        goal=(10, 10),
        discrete=True,
        rwrd_in_sim=True,
        out_boundaries_rwrd=-100,
        dt_sim=1,
        n_vel=5,
        n_angles=11,
        vo=True,
    )
    s0, _ = real_env.reset()
    fig = plt.Figure()

    # ALL on the same plot
    plot_base(goal=s0.goal, obs=s0.obstacles, config=real_env.config)
    plot_trj(traj=trj_vanilla, color="r", label="MCTS")
    plot_trj(traj=trj_vo2, color="g", label="VO2")
    plot_trj(traj=trj_dwa, color="b", label="DWA")
    plot_trj(traj=trj_mpc, color="m", label="NMPC")
    plt.legend()
    plt.savefig(f"debug/all_trj.png", dpi=500, facecolor="white", edgecolor="none")

    plt.gcf().clear()
    plot_base(goal=s0.goal, obs=s0.obstacles, config=real_env.config)
    plot_trj(traj=trj_vanilla, color="r", label="MCTS")
    plt.legend()
    plt.savefig(
        f"debug/MCTS_trj.png",
        dpi=500
    )

    plt.gcf().clear()
    plot_base(goal=s0.goal, obs=s0.obstacles, config=real_env.config)
    plot_trj(traj=trj_vo2, color="g", label="VO2")
    plt.legend()
    plt.savefig(
        f"debug/VO2_trj.png",
        dpi=500
    )

    plt.gcf().clear()
    plot_base(goal=s0.goal, obs=s0.obstacles, config=real_env.config)
    plot_trj(traj=trj_dwa, color="b", label="DWA")
    plt.legend()
    plt.savefig(
        f"debug/DWA_trj.png",
        dpi=500
    )

    plt.gcf().clear()
    plot_base(goal=s0.goal, obs=s0.obstacles, config=real_env.config)
    plot_trj(traj=trj_mpc, color="m", label="NMPC")
    plt.legend()
    plt.savefig(f"debug/NMPC_trj.png", dpi=500)

def two_ag():
    # 1 AGENT
    BASE_PATH = "debug/TRAJECTORIES/2AG"
    path_vanilla = (
        BASE_PATH
        + "/"
        + "trajectory2ag_real_algorithm:VANILLA_nsim:20_rwrd:-100_dt:1.0_std:0.5_stdRollout:1.0_amplitude:1_c:10.0_rollout:epsilon_uniform_uniform_alpha:0.1_k:50.0_a:11_v:5_num:1_eps_rollout:0.2_max_depth:100_env:EASY_0.pkl"
    )
    path_vo2 = (
        BASE_PATH
        + "/"
        + "trajectory2ag_real_algorithm:VANILLA_VO2_nsim:10_rwrd:-100_dt:1.0_std:0.5_stdRollout:1.0_amplitude:1_c:10.0_rollout:epsilon_uniform_uniform_alpha:0.1_k:50.0_a:11_v:5_num:1_eps_rollout:0.2_max_depth:100_env:EASY_0.pkl"
    )
    path_dwa = BASE_PATH + "/" + "trajectory_real2ag_dwa.pkl"
    with open(path_vanilla, "rb") as f:
        trj_vanilla = pickle.load(f)
    with open(path_vo2, "rb") as f:
        trj_vo2 = pickle.load(f)
    with open(path_dwa, "rb") as f:
        trj_dwa = pickle.load(f)

    real_env, sim_env = create_env_five_small_obs_continuous(
        initial_pos=(1, 1),
        goal=(10, 10),
        discrete=True,
        rwrd_in_sim=True,
        out_boundaries_rwrd=-100,
        dt_sim=1,
        n_vel=5,
        n_angles=11,
        vo=True,
    )
    s0, _ = real_env.reset()
    fig = plt.Figure()

    # ALL on the same plot
    plot_base(goal=s0.goal, obs=s0.obstacles, config=real_env.config)
    plot_trj(traj=trj_vanilla[0], color="r", label="MCTS")
    plot_trj(traj=trj_vanilla[1], color="r", label="")
    plot_trj(traj=trj_vo2[0], color="g", label="VO2")
    plot_trj(traj=trj_vo2[1], color="g", label="")
    plot_trj(traj=trj_dwa[0], color="b", label="DWA")
    plot_trj(traj=trj_dwa[1], color="b", label="")
    plt.legend()
    plt.savefig(f"debug/all_trj_2ag.png", dpi=500, facecolor="white", edgecolor="none")

    plt.gcf().clear()
    plot_base(goal=s0.goal, obs=s0.obstacles, config=real_env.config)
    plot_trj(traj=trj_vanilla[0], color="r", label="MCTS")
    plot_trj(traj=trj_vanilla[1], color="r", label="")
    plt.legend()
    plt.savefig(
        f"debug/MCTS_trj_2ag.png",
        dpi=500
    )

    plt.gcf().clear()
    plot_base(goal=s0.goal, obs=s0.obstacles, config=real_env.config)
    plot_trj(traj=trj_vo2[0], color="g", label="VO2")
    plot_trj(traj=trj_vo2[1], color="g", label="")
    plt.legend()
    plt.savefig(
        f"debug/VO2_trj_2ag.png",
        dpi=500
    )

    plt.gcf().clear()
    plot_base(goal=s0.goal, obs=s0.obstacles, config=real_env.config)
    plot_trj(traj=trj_dwa[0], color="b", label="DWA")
    plot_trj(traj=trj_dwa[1], color="b", label="")
    plt.legend()
    plt.savefig(
        f"debug/DWA_trj_2ag.png",
        dpi=500
    )


def four_ag():
    # 1 AGENT
    BASE_PATH = "debug/TRAJECTORIES/4AG"
    path_vanilla = (
        BASE_PATH
        + "/"
        + "trajectory4ag_real_algorithm:VANILLA_nsim:20_rwrd:-100_dt:1.0_std:0.5_stdRollout:1.0_amplitude:1_c:10.0_rollout:epsilon_uniform_uniform_alpha:0.1_k:50.0_a:11_v:5_num:1_eps_rollout:0.2_max_depth:100_env:EASY_0.pkl"
    )
    path_vo2 = (
        BASE_PATH
        + "/"
        + "trajectory4ag_real_algorithm:VANILLA_VO2_nsim:20_rwrd:-100_dt:1.0_std:0.5_stdRollout:1.0_amplitude:1_c:10.0_rollout:epsilon_uniform_uniform_alpha:0.1_k:50.0_a:11_v:5_num:1_eps_rollout:0.2_max_depth:100_env:EASY_0.pkl"
    )
    path_dwa = BASE_PATH + "/" + "trajectory_real4ag_dwa.pkl"
    with open(path_vanilla, "rb") as f:
        trj_vanilla = pickle.load(f)
    with open(path_vo2, "rb") as f:
        trj_vo2 = pickle.load(f)
    with open(path_dwa, "rb") as f:
        trj_dwa = pickle.load(f)

    real_envs, sim_envs = create_env_4ag(
        discrete=True,
        rwrd_in_sim=True,
        out_boundaries_rwrd=-100,
        dt_sim=1,
        n_vel=5,
        n_angles=11,
        vo=True,
    )
    real_env = real_envs[0]
    initial_states = [env.reset()[0] for env in real_envs]
    fig = plt.Figure()

    # ALL on the same plot
    plot_base_four_obs(initial_states=initial_states, config=real_env.config)
    plot_trj(traj=trj_vanilla[0], color="r", label="MCTS")
    plot_trj(traj=trj_vanilla[1], color="r", label="")
    plot_trj(traj=trj_vanilla[2], color="r", label="")
    plot_trj(traj=trj_vanilla[3], color="r", label="")
    plot_trj(traj=trj_vo2[0], color="g", label="VO2")
    plot_trj(traj=trj_vo2[1], color="g", label="")
    plot_trj(traj=trj_vo2[2], color="g", label="")
    plot_trj(traj=trj_vo2[3], color="g", label="")
    plot_trj(traj=trj_dwa[0], color="b", label="DWA")
    plot_trj(traj=trj_dwa[1], color="b", label="")
    plot_trj(traj=trj_dwa[2], color="b", label="")
    plot_trj(traj=trj_dwa[3], color="b", label="")
    plt.legend()
    plt.savefig(f"debug/all_trj_4ag.png", dpi=500, facecolor="white", edgecolor="none")

    plt.gcf().clear()
    plot_base_four_obs(initial_states=initial_states, config=real_env.config)
    plot_trj(traj=trj_vanilla[0], color="r", label="MCTS")
    plot_trj(traj=trj_vanilla[1], color="r", label="")
    plot_trj(traj=trj_vanilla[2], color="r", label="")
    plot_trj(traj=trj_vanilla[3], color="r", label="")
    plt.legend()
    plt.savefig(
        f"debug/MCTS_trj_4ag.png",
        dpi=500
    )

    plt.gcf().clear()
    plot_base_four_obs(initial_states=initial_states, config=real_env.config)
    plot_trj(traj=trj_vo2[0], color="g", label="VO2")
    plot_trj(traj=trj_vo2[1], color="g", label="")
    plot_trj(traj=trj_vo2[2], color="g", label="")
    plot_trj(traj=trj_vo2[3], color="g", label="")
    plt.legend()
    plt.savefig(
        f"debug/VO2_trj_4ag.png",
        dpi=500
    )

    plt.gcf().clear()
    plot_base_four_obs(initial_states=initial_states, config=real_env.config)
    plot_trj(traj=trj_dwa[0], color="b", label="DWA")
    plot_trj(traj=trj_dwa[1], color="b", label="")
    plot_trj(traj=trj_dwa[2], color="b", label="")
    plot_trj(traj=trj_dwa[3], color="b", label="")
    plt.legend()
    plt.savefig(
        f"debug/DWA_trj_4ag.png",
        dpi=500
    )


if __name__ == "__main__":
    one_ag()
    # two_ag()
    # four_ag()