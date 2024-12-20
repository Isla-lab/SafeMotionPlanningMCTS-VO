import math

import numpy as np
from numpy import array

from bettergym.environments.env import BetterEnv, EnvConfig
from bettergym.environments.robot_arena import BetterRobotArena, RobotArenaState, Config


def create_env_continuous(
        initial_pos,
        goal,
        obs,
        discrete: bool,
        rwrd_in_sim: bool,
        real_c: Config,
        sim_c: Config,
        vo: bool,
):
    initial_state = RobotArenaState(
        x=np.array([initial_pos[0], initial_pos[1], math.pi / 8.0, 0.0]),
        goal=np.array([goal[0], goal[1]]),
        obstacles=obs,
        radius=real_c.robot_radius,
    )
    real_env = BetterRobotArena(
        initial_state=initial_state,
        gradient=True,
        discrete_env=discrete,
        config=real_c,
        collision_rwrd=True,
        vo=vo,
    )
    sim_env = BetterRobotArena(
        initial_state=initial_state,
        gradient=True,
        discrete_env=discrete,
        config=sim_c,
        collision_rwrd=rwrd_in_sim,
        vo=vo,
    )
    return real_env, sim_env


def create_env_five_small_obs_continuous(
        initial_pos: tuple,
        goal: tuple,
        discrete: bool,
        rwrd_in_sim: bool,
        out_boundaries_rwrd: int,
        dt_sim: float,
        n_angles: int,
        n_vel: int,
        vo: bool,
):
    obstacles_positions = np.array(
        [[3, 2.9], [3, 7.2], [7.1, 2.9], [7.1, 7.2]]
    )
    # obstacles_positions = np.array(
    #     [[4.0, 4.0], [4.0, 6.0], [5.0, 5.0], [6.0, 4.0], [6.0, 6.0]]
    # )

    dt_real = dt_sim
    real_c = Config(
        dt=dt_real, max_angle_change=1.9 * dt_real, n_angles=n_angles, n_vel=n_vel
        , obs_size=1.5
    )
    sim_c = Config(
        dt=dt_sim, max_angle_change=1.9 * dt_sim, n_angles=n_angles, n_vel=n_vel
        , obs_size=1.5
    )
    obs = [
        RobotArenaState(
            np.pad(ob, (0, 2), "constant"),
            goal=None,
            obstacles=None,
            radius=real_c.obs_size,
        )
        for ob in obstacles_positions
    ]
    real_env, sim_env = create_env_continuous(
        initial_pos=initial_pos,
        goal=goal,
        obs=obs,
        discrete=discrete,
        rwrd_in_sim=rwrd_in_sim,
        real_c=real_c,
        sim_c=sim_c,
        vo=vo,
    )
    sim_env.gym_env.WALL_REWARD = out_boundaries_rwrd
    return real_env, sim_env


def create_env_four_obs_difficult_continuous(
        initial_pos: tuple,
        goal: tuple,
        discrete: bool,
        rwrd_in_sim: bool,
        out_boundaries_rwrd: int,
        dt_sim: float,
        n_angles: int,
        n_vel: int,
        vo: bool,
):
    obstacles_positions = np.array(
        [
            [3.4, 0.8],
            [0.7, 4.3],
            [4.5, 5.0],
            [3.0, 7.0],
            [7.0, 3.0],
            [8.0, 0.5],
            [6.2, 1.0],
            [6.0, 7.0],
            [5.0, 9.0],
            [6.8, 5.0],
            [9.0, 6.0],
            [8.0, 8.0],
            [10.0, 8.0],
            [8.0, 10.0],
            [9.5, 3.0],
            [1.0, 8.0],
            [3.0, 10.0]
        ]
    )

    radiuses = [1.5, 1.5, 1, 0.5, 0.6, 0.3, 0.4, 0.5, 0.4, 0.3, 0.5, 0.3, 0.3, 0.3, 0.4, 0.6, 0.4]

    dt_real = dt_sim
    real_c = Config(
        dt=dt_real, max_angle_change=1.9 * dt_real, n_angles=n_angles, n_vel=n_vel
    )
    sim_c = Config(
        dt=dt_sim, max_angle_change=1.9 * dt_sim, n_angles=n_angles, n_vel=n_vel
    )
    obs = [
        RobotArenaState(
            np.pad(obstacles_positions[i], (0, 2), "constant"),
            goal=None,
            obstacles=None,
            radius=radiuses[i],
        )
        for i in range(len(obstacles_positions))
    ]

    real_env, sim_env = create_env_continuous(
        initial_pos=initial_pos,
        goal=goal,
        obs=obs,
        discrete=discrete,
        rwrd_in_sim=rwrd_in_sim,
        real_c=real_c,
        sim_c=sim_c,
        vo=vo,
    )
    sim_env.gym_env.WALL_REWARD = out_boundaries_rwrd
    return real_env, sim_env


def create_env_four_obs_difficult_continuous2(
        initial_pos: tuple,
        goal: tuple,
        discrete: bool,
        rwrd_in_sim: bool,
        out_boundaries_rwrd: int,
        dt_sim: float,
        n_angles: int,
        n_vel: int,
        vo: bool,
):
    obstacles_positions = array([[4.73193883e+00, -1.88885218e-01],
                                 [6.09594973e+00, 4.72386871e+00],
                                 [4.54441363e+00, 3.46401785e+00],
                                 [1.95578361e+00, 6.93125160e+00],
                                 [3.09585608e+00, 2.70192730e+00],
                                 [6.95360599e+00, 5.84970513e+00],
                                 [1.11495934e+00, 5.66293746e+00],
                                 [1.71327839e+00, 8.92402177e+00],
                                 [9.74770351e+00, 5.43084205e+00],
                                 [9.65873782e+00, 4.55745724e-01],
                                 [5.56295308e+00, 2.83438053e-01],
                                 [4.63746793e+00, 6.58370988e-01],
                                 [1.02591966e+00, 6.66094371e+00],
                                 [2.21214401e+00, 7.83348212e-01],
                                 [2.14367448e+00, 3.69791542e+00],
                                 [5.11344981e+00, 1.92091872e+00],
                                 [7.18488070e+00, 5.29683803e+00],
                                 [5.56284064e+00, 4.14271181e+00],
                                 [9.02364945e+00, 6.46005015e+00],
                                 [1.44758318e+00, 7.90902816e+00],
                                 [1.10746130e+01, 5.50010033e+00],
                                 [1.01742408e+01, 3.59936383e+00],
                                 [6.30572953e+00, 4.63055156e+00],
                                 [4.74096716e+00, 8.81871022e+00],
                                 [5.92725008e+00, 1.09449067e+01],
                                 [6.03049792e+00, 4.85139067e-01],
                                 [3.89610882e+00, 9.71020605e+00],
                                 [4.37530052e+00, -1.73571609e-01],
                                 [2.46612687e+00, 3.05732449e-01],
                                 [1.14262241e+01, 1.11469638e+01],
                                 [9.10310021e+00, 6.72180546e+00],
                                 [8.67951833e+00, 1.53070536e+00],
                                 [3.01627878e+00, 5.78880250e+00],
                                 [3.77949137e+00, 4.81475829e-02],
                                 [1.12978413e+01, 4.79625903e+00],
                                 [5.54800527e+00, 3.38249581e+00],
                                 [2.61693703e+00, 4.14267862e+00],
                                 [9.48420280e+00, 8.34096468e+00],
                                 [4.05052680e+00, -3.43791960e-01],
                                 [9.06885927e+00, 2.73266557e+00],
                                 [6.49221866e+00, -1.93388698e-01],
                                 [7.44642423e+00, 4.15028111e+00],
                                 [5.46488558e+00, 4.47887005e+00],
                                 [3.71046282e+00, 6.11173486e+00],
                                 [1.11749283e+01, 8.53314583e-01],
                                 [3.25910234e+00, 1.57251778e-03],
                                 [8.36079710e+00, 7.39014865e+00],
                                 [2.07562896e+00, 4.50104128e+00],
                                 [7.22610321e+00, 7.43777592e+00],
                                 [1.54572560e+00, 1.00798268e+01],
                                 [8.83609792e+00, 1.10745050e+00],
                                 [9.92699952e+00, 8.48533454e+00],
                                 [9.08302785e+00, 6.02014277e+00],
                                 [2.15005498e+00, 1.05215038e+01],
                                 [6.60501493e+00, 3.65485491e+00],
                                 [2.66534235e+00, 1.04669857e+01],
                                 [4.53682554e+00, 5.98229819e+00],
                                 [6.80130589e+00, 9.41499794e+00],
                                 [6.98275821e+00, 1.62054594e+00],
                                 [6.59508823e+00, 5.37119400e+00],
                                 [6.07489336e+00, 7.89424744e+00],
                                 # [2.44973397e+00, 1.73952575e+00],
                                 [8.26997774e-01, 2.78871103e+00],
                                 [-3.76999527e-01, 7.05231667e+00],
                                 # [3.04206766e+00, 1.74743854e+00],
                                 [6.43456627e-01, 2.90506969e+00],
                                 [2.07909261e+00, 2.92724481e+00],
                                 [5.15691829e+00, 6.09396185e+00],
                                 [9.64135739e+00, 1.13621164e+01],
                                 [8.64170348e-02, 2.28541902e+00],
                                 [7.21973713e+00, 1.43759875e+00],
                                 [9.94175072e+00, 2.10882912e+00],
                                 [8.40106047e+00, 7.33624616e+00],
                                 [9.08662614e+00, -1.25029253e-01],
                                 [2.25488834e+00, 7.95552999e+00],
                                 [5.50750108e-01, -1.32926202e-01],
                                 [3.78561920e+00, 6.57738390e+00],
                                 [1.26672729e-01, 2.87963994e-01],
                                 [2.20142327e-02, 4.24180910e+00],
                                 [7.52112455e+00, 1.87632534e+00],
                                 [1.00151974e+01, 4.68866915e+00]])

    radiuses = [array([0.17763018]),
                array([0.52694678]),
                array([0.47469102]),
                array([0.41897563]),
                array([0.18717842]),
                array([0.36223388]),
                array([0.16799881]),
                array([0.08760221]),
                array([0.89196181]),
                array([0.5456292]),
                array([0.66888893]),
                array([0.335648]),
                array([0.5044297]),
                array([0.50461244]),
                array([0.49022737]),
                array([0.40054892]),
                array([0.36576287]),
                array([0.08955816]),
                array([0.49152161]),
                array([0.46352454]),
                array([0.56768301]),
                array([0.35886369]),
                array([0.66127183]),
                array([0.33287487]),
                array([0.21168592]),
                array([0.54072455]),
                array([0.32974806]),
                array([0.21179195]),
                array([0.41872355]),
                array([0.51347532]),
                array([0.43621594]),
                array([0.47461385]),
                array([1.01630389]),
                array([0.52316813]),
                array([0.62976405]),
                array([0.10212313]),
                array([0.45475803]),
                array([0.51611231]),
                array([0.34532814]),
                array([0.26860308]),
                array([0.45947802]),
                array([0.35280738]),
                array([0.31093666]),
                array([0.57242509]),
                array([0.60033493]),
                array([0.4453944]),
                array([0.33186582]),
                array([0.47715561]),
                array([0.52441876]),
                array([0.4945933]),
                array([0.18199535]),
                array([0.34929733]),
                array([0.45387]),
                array([0.49333306]),
                array([0.54411729]),
                array([0.51375555]),
                array([0.22985223]),
                array([0.29388687]),
                array([0.21678563]),
                array([0.6706715]),
                array([0.42706147]),
                # array([0.48297464]),
                array([0.55495436]),
                array([0.35064963]),
                # array([0.22734956]),
                array([0.33602163]),
                array([0.37777792]),
                array([0.62521554]),
                array([0.53043973]),
                array([0.23693641]),
                array([0.49963321]),
                array([0.5102327]),
                array([0.24079514]),
                array([0.38372248]),
                array([0.1224394]),
                array([0.44957321]),
                array([0.35264602]),
                array([0.19749968]),
                array([0.29527435]),
                array([0.43599268]),
                array([0.31705758])]

    dt_real = dt_sim
    real_c = Config(
        dt=dt_real, max_angle_change=1.9 * dt_real, n_angles=n_angles, n_vel=n_vel
    )
    sim_c = Config(
        dt=dt_sim, max_angle_change=1.9 * dt_sim, n_angles=n_angles, n_vel=n_vel
    )
    obs = [
        RobotArenaState(
            np.pad(obstacles_positions[i], (0, 2), "constant"),
            goal=None,
            obstacles=None,
            radius=radiuses[i],
        )
        for i in range(len(obstacles_positions))
    ]

    real_env, sim_env = create_env_continuous(
        initial_pos=initial_pos,
        goal=goal,
        obs=obs,
        discrete=discrete,
        rwrd_in_sim=rwrd_in_sim,
        real_c=real_c,
        sim_c=sim_c,
        vo=vo,
    )
    sim_env.gym_env.WALL_REWARD = out_boundaries_rwrd
    return real_env, sim_env


def create_pedestrian_env(
        discrete: bool,
        rwrd_in_sim: bool,
        out_boundaries_rwrd: int,
        n_angles: int,
        n_vel: int,
        vo: bool,
        n_obs: int,
        obs_pos: list = None,
):
    dt_real = 1.0
    real_c = EnvConfig(
        dt=dt_real, max_angle_change=1.9 * dt_real, n_angles=n_angles, n_vel=n_vel, num_humans=n_obs
    )

    real_env = BetterEnv(discrete_env=discrete, vo=vo, config=real_c, collision_rwrd=True, sim_env=False,
                         obs_pos=obs_pos)
    sim_env = BetterEnv(discrete_env=discrete, vo=vo, config=real_c, collision_rwrd=rwrd_in_sim, sim_env=True,
                        obs_pos=None)
    sim_env.gym_env.WALL_REWARD = out_boundaries_rwrd
    real_env.gym_env.WALL_REWARD = out_boundaries_rwrd

    return real_env, sim_env
