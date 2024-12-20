import timeit
from functools import partial

import numpy as np
from numba import njit
from numpy import array


@njit
def check_coll_jit(x, obs, robot_radius, obs_size):
    for i, ob in enumerate(obs):
        dist_to_ob = np.linalg.norm(ob - x[:2])
        if dist_to_ob <= robot_radius + obs_size[i]:
            return True
    return False


def check_coll_vectorized(x, obs, robot_radius, obs_size):
    if len(obs) == 0:
        return False
    dist_to_ob = np.linalg.norm(obs - x[:2], axis=1)
    return np.any(dist_to_ob <= robot_radius + obs_size)


@njit
def dist_to_goal(goal: np.ndarray, x: np.ndarray):
    return np.linalg.norm(x - goal)