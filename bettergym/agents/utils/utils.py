import math
import random
from functools import partial
from typing import Any, Callable

# import graphviz
import numpy as np
from numba import njit
from scipy.spatial.distance import cdist

from bettergym.agents.planner import Planner
from mcts_utils import uniform_random


def get_robot_angles(x, max_angle_change):
    robot_angles = [x[2] - max_angle_change, x[2] + max_angle_change]
    robot_angles = np.array(robot_angles)
    # Make sure angle is within range of -π to π
    robot_angles = (robot_angles + np.pi) % (2 * np.pi) - np.pi
    if type(robot_angles[0]) is np.float64:
        robot_angles = [robot_angles]
    new_robot_angles = []
    for a in robot_angles:
        if a[0] > a[1]:
            new_robot_angles.extend([[a[0], math.pi], [-math.pi, a[1]]])
        else:
            new_robot_angles.append(a)
    return new_robot_angles


def uniform(node: Any, planner: Planner):
    current_state = node.state
    available_actions = planner.environment.get_actions(current_state)
    return available_actions.sample()


def uniform_discrete(node: Any, planner: Planner):
    current_state = node.state
    actions = planner.environment.get_actions(current_state)
    return random.choice(actions)


@njit
def compute_towards_goal_jit(
        x: np.ndarray,
        goal: np.ndarray,
        max_angle_change: float,
        std_angle_rollout: float,
        min_speed: float,
        max_speed: float,
):
    mean_angle = np.arctan2(goal[1] - x[1], goal[0] - x[0])
    angle = np.random.normal(mean_angle, std_angle_rollout)
    linear_velocity = np.random.uniform(low=min_speed, high=max_speed)
    # Make sure angle is within range of -π to π
    min_angle = x[2] - max_angle_change
    max_angle = x[2] + max_angle_change
    angle = max(min(angle, max_angle), min_angle)
    angle = (angle + math.pi) % (2 * math.pi) - math.pi
    return np.array([linear_velocity, angle])

def towards_goal(node: Any, planner: Planner, std_angle_rollout: float):
    config = planner.environment.config
    return compute_towards_goal_jit(
        x=node.state.x,
        goal=node.state.goal,
        max_angle_change=config.max_angle_change,
        std_angle_rollout=std_angle_rollout,
        min_speed=config.min_speed,
        max_speed=config.max_speed,
    )


def epsilon_normal_uniform(
        node: Any, planner: Planner, std_angle_rollout: float, eps=0.1
):
    config = planner.environment.config
    prob = random.random()
    if prob <= 1 - eps:
        return compute_towards_goal_jit(
            x=node.state.x,
            goal=node.state.goal,
            max_angle_change=config.max_angle_change,
            std_angle_rollout=std_angle_rollout,
            min_speed=config.min_speed,
            max_speed=config.max_speed,
        )
    else:
        return uniform_random(node, planner)


def epsilon_uniform_uniform(
        node: Any, planner: Planner, std_angle_rollout: float, eps=0.1
):
    config = planner.environment.config
    prob = random.random()
    if prob <= 1 - eps:
        return compute_uniform_towards_goal_jit(
            x=node.state.x,
            goal=node.state.goal,
            max_angle_change=config.max_angle_change,
            amplitude=std_angle_rollout,
            min_speed=config.min_speed,
            max_speed=config.max_speed,
        )
    else:
        return uniform_random(node, planner)


@njit
def compute_uniform_towards_goal_jit(
        x: np.ndarray,
        goal: np.ndarray,
        max_angle_change: float,
        min_speed: float,
        max_speed: float,
        amplitude: float,
):
    mean_angle = np.arctan2(goal[1] - x[1], goal[0] - x[0])
    linear_velocity = np.random.uniform(low=min_speed, high=max_speed)
    # Make sure angle is within range of -π to π
    min_angle = x[2] - max_angle_change
    max_angle = x[2] + max_angle_change
    angle = np.random.uniform(low=mean_angle - amplitude, high=mean_angle + amplitude)

    angle = max(min(angle, max_angle), min_angle)
    angle = (angle + math.pi) % (2 * math.pi) - math.pi
    return np.array([linear_velocity, angle])


def epsilon_greedy(eps: float, other_func: Callable, node: Any, planner: Planner):
    """
    :param node:
    :param eps: defines the probability of acting according to other_func
    :param other_func:
    :param planner:
    :return:
    """
    prob = random.random()
    if prob <= 1 - eps:
        return other_func(node, planner)
    else:
        return uniform(node, planner)


def binary_policy(node: Any, planner: Planner):
    if len(node.actions) == 1:
        return uniform(node, planner)
    else:
        sorted_actions = [
            a
            for _, a in sorted(
                zip(node.a_values, node.actions), key=lambda pair: pair[0]
            )
        ]
        return np.mean([sorted_actions[0].action, sorted_actions[1].action], axis=0)


def voronoi(actions: np.ndarray, q_vals: np.ndarray, sample_centered: Callable):
    N_SAMPLE = 1000
    valid = False
    n_iter = 0
    # find the index of the action with the highest Q-value
    best_action_index = np.argmax(q_vals)

    # get the action with the highest Q-value
    best_action = actions[best_action_index]
    tmp_best = None
    tmp_dist = np.inf
    while not valid:
        if n_iter >= 100:
            return tmp_best

        # generate random points centered around the best action
        points = sample_centered(center=best_action, number=N_SAMPLE)

        # compute the Euclidean distances between each point and each action
        # column -> actions
        # rows -> points
        dists = cdist(points, actions, "euclidean")

        # find the distances between each point and the best action
        best_action_distances = dists[:, best_action_index]

        # repeat the distances for each action except the best action (necessary for doing `<=` later)
        best_action_distances_rep = np.tile(
            best_action_distances, (dists.shape[1] - 1, 1)
        ).T

        # remove the column for the best action from the distance matrix
        # dists = np.hstack((dists[:, :best_action_index], dists[:, best_action_index + 1:]))
        dists = np.delete(dists, best_action_index, axis=1)

        # find the closest action to each point
        closest = best_action_distances_rep <= dists

        # find the rows where all distances to other actions are greater than the distance to the best action
        all_true_rows = np.where(np.all(closest, axis=1))[0]

        # find the index of the point closest to the best action among the valid rows
        valid_points = best_action_distances[all_true_rows]
        if len(valid_points >= 0):
            # closest_point_idx = np.argmin(valid_points)
            closest_point_idx = all_true_rows[np.argmin(valid_points)]
            # return the closest point to the best action
            return points[closest_point_idx]
        else:
            closest_point_idx = np.argmin(best_action_distances)
            if d := best_action_distances[closest_point_idx] < tmp_dist:
                # return the closest point to the best action
                tmp_best = points[closest_point_idx]
                tmp_dist = d
        n_iter += 1
        del (
            points,
            dists,
            best_action_distances,
            best_action_distances_rep,
            closest,
            all_true_rows,
            valid_points,
        )


@njit
def clip_act(
        chosen: np.ndarray, max_angle_change: float, x: np.ndarray, allow_negative: bool
):
    if allow_negative:
        chosen[:, 0] = (chosen[:, 0] % 0.4) - 0.1
    else:
        chosen[:, 0] = chosen[:, 0] % 0.3
    min_available_angle = x[2] - max_angle_change
    max_available_angle = x[2] + max_angle_change
    # Make sure angle is within range of -min_angle to max_angle
    chosen[:, 1] = (
            chosen[:, 1] % (max_available_angle - min_available_angle) + min_available_angle
    )
    # Make sure angle is within range of -π to π
    chosen[:, 1] = (chosen[:, 1] + math.pi) % (2 * math.pi) - math.pi
    return chosen


def voo(eps: float, sample_centered: Callable, node: Any, planner: Planner):
    prob = random.random()
    if prob <= 1 - eps and len(node.actions) != 0:
        config = planner.environment.gym_env.config
        return voronoi(
            np.array([node.action for node in node.actions]),
            node.a_values,
            partial(
                sample_centered,
                clip_fn=partial(
                    clip_act,
                    max_angle_change=config.max_angle_change,
                    x=node.state.x,
                    allow_negative=True,
                ),
            ),
        )
    else:
        return uniform(node, planner)