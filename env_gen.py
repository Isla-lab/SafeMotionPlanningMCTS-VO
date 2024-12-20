import os
import pickle
import random

import numpy as np
from numba import njit
from tqdm import tqdm

from environment_creator import create_pedestrian_env


@njit
def seed_numba(seed_value: int):
    np.random.seed(seed_value)


def seed_everything(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)


seed_everything(1)
for exp_n in range(100):
    print(f"EXP {exp_n}")
    obs = []
    n_obs = 40
    real_env, sim_env = create_pedestrian_env(
        discrete=True,
        rwrd_in_sim=False,
        out_boundaries_rwrd=False,
        n_vel=5,
        n_angles=5,
        vo=False,
        n_obs=n_obs,
    )
    s0, _ = real_env.reset()
    s = s0
    for _ in tqdm(range(1000)):
        s, r, terminal, truncated, env_info = real_env.step(s, np.array([0., 0.]))
        obs.append(s.obstacles)
    with open(f"debug/obs_{exp_n}.pkl", "wb") as f:
        pickle.dump(obs, f)
