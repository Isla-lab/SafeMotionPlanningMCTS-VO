import math
import random
from functools import partial
from typing import Callable, Any

import numpy as np
import portion as P
from numpy.distutils.command.config import config

from bettergym.agents.planner import Planner
from bettergym.agents.utils.utils import (
    voronoi,
    clip_act,
    compute_towards_goal_jit,
    get_robot_angles, compute_uniform_towards_goal_jit,
)
from mcts_utils import uniform_random, get_intersections_vectorized, check_circle_segment_intersect, \
    angle_distance_vector


# def print_to_file(param):
#     # with open("OUTPUT.txt", "a") as f:
#     #     f.write(str(param))
#     pass

def join_intersections(int_points, radius, x):
    """
    Joins intersection points that are within a specified radius.

    Parameters:
    int_points (np.ndarray): An array of intersection points with shape (N, 4),
                             where each row represents two points (x1, y1, x2, y2).
    radius (float): The radius within which intersection points should be joined.

    Returns:
    np.ndarray: An array of joined intersection points.
    """
    # Split the intersection points into two sets of points
    p1 = int_points[:, :2]
    p2 = int_points[:, 2:]

    vec_p1 = np.array([p1[:, 0] - x[0], p1[:, 1] - x[1]])
    vec_p2 = np.array([p2[:, 0] - x[0], p2[:, 1] - x[1]])
    angle1 = np.arctan2(vec_p1[1], vec_p1[0])
    angle2 = np.arctan2(vec_p2[1], vec_p2[0])


    # Create a mask to determine which points to keep
    mask_ang1_smaller = angle1 < angle2
    points = np.vstack((
        np.hstack((p2[~mask_ang1_smaller], p1[~mask_ang1_smaller])),
        int_points[mask_ang1_smaller]
    ))
    new_angles = np.vstack((
        np.hstack((np.expand_dims(angle2[~mask_ang1_smaller], 1), np.expand_dims(angle1[~mask_ang1_smaller], 1))),
        np.hstack((np.expand_dims(angle1[mask_ang1_smaller], 1), np.expand_dims(angle2[mask_ang1_smaller], 1))),
    ))


    # Sort the points based on the x-coordinate
    points = points[np.argsort(new_angles[:, 0])]


    p1 = points[:, :2]
    p2 = points[:, 2:]

    # Calculate the distance between consecutive points
    first_int = p2[:-1, :]
    second_int = p1[1:, :]
    distance = np.linalg.norm(first_int - second_int, axis=1)

    # Create a mask for points that need to be joined
    to_join_mask = distance < 2 * radius
    if np.any(to_join_mask):
        to_join_start_idx = np.where(to_join_mask)[0]
        to_join_end_idx = to_join_start_idx + 1

        # Remove the points that are being joined
        remaining_points = np.delete(points, np.hstack((to_join_start_idx, to_join_end_idx)), axis=0)

        # Add the joined points to the remaining points
        final_points = np.vstack((
            remaining_points,
            np.hstack((p1[to_join_start_idx], p2[to_join_end_idx]))
        ))
    else:
        final_points = int_points

    return final_points

# def join_intersections2(int_points, radius, x):
#     """
#     Joins intersection points that are within a specified radius.
#
#     Parameters:
#     int_points (np.ndarray): An array of intersection points with shape (N, 4),
#                              where each row represents two points (x1, y1, x2, y2).
#     radius (float): The radius within which intersection points should be joined.
#
#     Returns:
#     np.ndarray: An array of joined intersection points.
#     """
#     # Split the intersection points into two sets of points
#     p1 = int_points[:, :2]
#     p2 = int_points[:, 2:]
#
#     vec_p1 = np.array([p1[:, 0] - x[0], p1[:, 1] - x[1]])
#     vec_p2 = np.array([p2[:, 0] - x[0], p2[:, 1] - x[1]])
#     angle1 = np.arctan2(vec_p1[1], vec_p1[0])
#     angle2 = np.arctan2(vec_p2[1], vec_p2[0])
#
#
#     # Create a mask to determine which points to keep
#     mask_ang1_smaller = angle1 < angle2
#     points = np.vstack((
#         np.hstack((p2[~mask_ang1_smaller], p1[~mask_ang1_smaller])),
#         int_points[mask_ang1_smaller]
#     ))
#     new_angles = np.vstack((
#         np.hstack((np.expand_dims(angle2[~mask_ang1_smaller], 1), np.expand_dims(angle1[~mask_ang1_smaller], 1))),
#         np.hstack((np.expand_dims(angle1[mask_ang1_smaller], 1), np.expand_dims(angle2[mask_ang1_smaller], 1))),
#     ))
#
#
#     # Sort the points based on the angle
#     points = points[np.argsort(new_angles[:, 0])]
#
#     p1 = points[:, :2]
#     p2 = points[:, 2:]
#
#     initial_point = p1[0]
#     final_point = p2[-1]
#     final_points = np.expand_dims(np.hstack((initial_point, final_point)), 0)
#
#     return final_points

def uniform_towards_goal_vo(node: Any, planner: Planner, std_angle_rollout: float):
    config = planner.environment.gym_env.config
    x = node.state.x

    if len(node.state.obstacles) == 0:
        return compute_uniform_towards_goal_jit(
            x=x,
            goal=node.state.goal,
            max_angle_change=config.max_angle_change,
            amplitude=std_angle_rollout,
            min_speed=0.0,
            max_speed=config.max_speed,
        )

    # Extract robot information
    dt = config.dt
    ROBOT_RADIUS = config.robot_radius
    VMAX = config.max_speed
    wall_int = None

    # Extract obstacle information
    obstacles = node.state.obstacles
    # obs_x, obs_rad
    square_obs = [[], []]
    circle_obs = [[], []]
    wall_obs = [[], []]
    intersection_points = np.empty((0, 4), dtype=np.float64)
    for ob in obstacles:
        if ob.obs_type == "square":
            square_obs[0].append(ob.x)
            square_obs[1].append(ob.radius)
        elif ob.obs_type == "circle":
            circle_obs[0].append(ob.x)
            circle_obs[1].append(ob.radius)
        else:
            wall_obs[0].append(ob.x)
            wall_obs[1].append(ob.radius)

    # CIRCULAR OBSTACLES
    circle_obs_x = np.array(circle_obs[0])
    circle_obs_rad = np.array(circle_obs[1])

    if len(circle_obs_x) != 0:
        # Calculate radii
        r1 = circle_obs_x[:, 3] * dt + circle_obs_rad + ROBOT_RADIUS
        r0 = np.full_like(r1, VMAX * dt)

        # Calculate intersection points
        intersection_points, dist, mask = get_intersections_vectorized(x, circle_obs_x, r0, r1)

    # WALL OBSTACLES
    intersection_data = check_circle_segment_intersect(x[:2], ROBOT_RADIUS + VMAX * dt, np.array(wall_obs[0]))
    valid_discriminant = intersection_data[0]
    if valid_discriminant.any():
        wall_int = np.array(wall_obs[0])[valid_discriminant]
        unsafe_wall_angles = get_unsafe_angles_wall(wall_int, x)
    else:
        unsafe_wall_angles = None

    # CASE 1 no obs intersection and no wall intersection
    if np.isnan(intersection_points).all() and wall_int is None:
        return compute_uniform_towards_goal_jit(
            x=x,
            goal=node.state.goal,
            max_angle_change=config.max_angle_change,
            amplitude=std_angle_rollout,
            min_speed=0.0,
            max_speed=config.max_speed,
        )
    # CASE 2 only wall intersection
    # CASE 3 only obs intersection
    # CASE 4 both wall and obs intersection
    else:
        # compute intersection with our new circumference
        # print_to_file(
        #     "X: {}".format(x) + "\n" +
        #     "-" * 50 + "\n" +
        #     "Intersection points: {}".format(intersection_points) + "\n" +
        #     "-" * 50 + "\n" +
        #     "Unsafe wall angles: {}".format(unsafe_wall_angles) + "\n" +
        #     "-" * 50 + "\n" +
        #     "Wall Int: {}".format(wall_int) + "\n" +
        #     "-" * 50 + "\n" +
        #     "Square Obs: {}".format(square_obs) + "\n" +
        #     "-" * 50 + "\n" +
        #     "Circle Obs: {}".format(circle_obs) + "\n" +
        #     "-" * 50 + "\n" +
        #     "Wall Obs {}".format(wall_obs) + "\n"
        # )
        # if sum(mask) > 1:
        #     intersection_points = join_intersections2(intersection_points[mask], ROBOT_RADIUS, x)
        angle_space, velocity_space = new_get_spaces([square_obs, circle_obs, wall_obs], x, config, intersection_points,
                                                     wall_angles=unsafe_wall_angles)
        # print_to_file(
        #     "Angle Space: {}".format(angle_space) + "\n" +
        #     "-" * 50 + "\n" +
        #     "Velocity Space: {}".format(velocity_space) + "\n"
        # )
        mean_angle = np.arctan2(node.state.goal[1] - x[1], node.state.goal[0] - x[0])
        angle_space = np.array(angle_space)
        angles = np.random.uniform(low=mean_angle - std_angle_rollout, high=mean_angle + std_angle_rollout, size=20)
        in_range = (angle_space[:, 0] <= angles[:, np.newaxis]) & (angle_space[:, 1] >= angles[:, np.newaxis])
        # print_to_file("In Range: {}".format(in_range))
        if not np.any(in_range):
            action = sample_multiple_spaces(center=None, a_space=angle_space, v_space=velocity_space, number=1)[0]
            if action[0] < 0:
                action[1] = action[1] + math.pi
                action[1] = (action[1] + math.pi) % (2 * math.pi) - math.pi
            return action
        else:
            idx_angles, idx_ranges = np.where(in_range)
            idx = random.randint(0, len(idx_angles) - 1)
            angle = angles[idx_angles[idx]]
            velocity = np.random.uniform(low=velocity_space[idx_ranges[idx]][0],
                                         high=velocity_space[idx_ranges[idx]][1])
            if velocity < 0:
                angle = angle + math.pi
                angle = (angle + math.pi) % (2 * math.pi) - math.pi
        return [velocity, angle]


def sample_multiple_spaces(center, a_space, number, v_space):
    lengths_aspace = np.linalg.norm(a_space, axis=1)
    percentages_aspace = np.cumsum(lengths_aspace / np.sum(lengths_aspace))
    # print_to_file(
    #     "Lenght space: {}".format(lengths_aspace) + "\n" +
    #     "-" * 50 + "\n" +
    #     "Percentages space: {}".format(percentages_aspace) + "\n"
    # )
    pct = random.random()
    idx_space = np.flatnonzero(pct <= percentages_aspace)[0]
    return np.vstack(
        [
            np.random.uniform(low=v_space[idx_space][0], high=v_space[idx_space][1], size=number),
            np.random.uniform(
                low=a_space[idx_space][0], high=a_space[idx_space][1], size=number
            ),
        ]
    ).T


def sample_single_space(center, a_space, number, v_space):
    return np.vstack(
        [
            np.random.uniform(low=v_space[0], high=v_space[1], size=number),
            np.random.uniform(low=a_space[0], high=a_space[1], size=number),
        ]
    ).T


def sample(center, a_space, v_space, number):
    if len(a_space) == 1:
        return sample_single_space(center, a_space[0], number, v_space)
    else:
        return sample_multiple_spaces(center, a_space, number, v_space)


def angle_distance(angle1, angle2):
    # Compute the absolute difference between the angles
    diff = abs(angle1 - angle2)

    # Ensure the shortest distance is used by considering wrap-around
    diff = min(diff, 2 * math.pi - diff)

    return diff


def get_spaces_vo_special_case(obs, x, r1, config, mask, curr_velocity_space):
    alpha = np.arctan2(obs[mask, 1] - x[1], obs[mask, 0] - x[0])
    P = obs[mask, :2] - (r1[mask][:, np.newaxis] * np.column_stack((np.cos(alpha), np.sin(alpha))))
    vmin = np.sum(np.abs(P - x[:2]), axis=1) / config.dt
    idx_vmin = np.argmax(vmin)
    alpha = alpha[idx_vmin]
    # if the robot is looking toward the obstacle center then negative speed

    # vspaces = [negative_vel, positive_vel]
    vspaces = [[-0.1, max(-vmin[idx_vmin], config.min_speed)],
               [min(vmin[idx_vmin], curr_velocity_space[1]), config.max_speed]]
    # if negative speed use opposite of alpha, if speed is positive then use alpha
    alphas = (np.array([alpha, alpha + np.pi]) + math.pi) % (2 * math.pi) - math.pi
    angle_dist = [angle_distance(x[2], alphas[0]), angle_distance(x[2], alphas[1])]
    idx = np.argmin(angle_dist)
    velocity_space = vspaces[idx]
    alpha = alphas[idx]

    angle_space = [[alpha, alpha]]
    return angle_space, velocity_space


def get_spaces(intersection_points, x, obs, r1, config, disable_retro=False, dist=None):
    angle_space, forbidden_ranges = compute_safe_angle_space(
        intersection_points=intersection_points,
        max_angle_change=config.max_angle_change,
        x=x,
    )
    velocity_space = [0., config.max_speed]
    radial = False
    delta = 0.015
    if dist is not None and np.any(mask := dist - delta < r1):
        velocity_space, angle_space = get_spaces_vo_special_case(obs, x, r1, config, mask, velocity_space)
        radial = True

    # No angle at positive velocity is safe
    if angle_space is None:
        if not disable_retro:
            retro_available, angle_space = vo_negative_speed(obs, x, config)
        else:
            retro_available = False

        if retro_available:
            # If VO with negative speed is possible, use it
            velocity_space = [config.min_speed, 0.]
        else:
            # TODO: check if this is correct
            velocity_space = [0.0, 0.0]
            angle_space = get_robot_angles(x, config.max_angle_change)

    return angle_space, velocity_space, radial


def compute_ranges_difference(robot_angles, forbidden_ranges):
    # Convert lists of ranges to portion intervals
    robot_intervals = P.Interval(*[P.closed(a[0], a[1]) for a in robot_angles])
    forbidden_intervals = P.Interval(*[P.closed(a[0], a[1]) for a in forbidden_ranges])

    # Compute the difference
    result_intervals = robot_intervals - forbidden_intervals

    # Convert the result back to a list of ranges
    result_ranges = [[i.lower, i.upper] for i in result_intervals]

    return result_ranges


def get_unsafe_angles(intersection_points, robot_angles, x):
    forbidden_ranges = []
    none_points = np.isnan(intersection_points).all(axis=1)
    inf_points = np.isinf(intersection_points).all(axis=1)
    if np.any(inf_points):
        forbidden_ranges.extend(robot_angles)

    new_points = intersection_points[np.logical_not(np.logical_or(none_points, inf_points))]
    if new_points.shape[0] != 0:
        if len(new_points.shape) == 1:
            new_points = np.expand_dims(new_points, axis=0)
        p1 = new_points[:, :2]
        p2 = new_points[:, 2:]
        vec_p1 = np.array([p1[:, 0] - x[0], p1[:, 1] - x[1]])
        vec_p2 = np.array([p2[:, 0] - x[0], p2[:, 1] - x[1]])
        angle1 = np.arctan2(vec_p1[1], vec_p1[0])
        angle2 = np.arctan2(vec_p2[1], vec_p2[0])
        angle1_greater_mask = angle1 > angle2
        forbidden_ranges.extend(np.column_stack((angle1[~angle1_greater_mask], angle2[~angle1_greater_mask])))
        forbidden_ranges.extend(
            np.vstack((
                np.column_stack((angle1[angle1_greater_mask], np.full_like(angle1[angle1_greater_mask], math.pi))),
                np.column_stack((np.full_like(angle2[angle1_greater_mask], -math.pi), angle2[angle1_greater_mask]))
            ))
        )
    return forbidden_ranges


def get_unsafe_angles_wall(intersection_points, x):
    unsafe_angles = np.array(get_unsafe_angles(intersection_points, None, x), copy=True)
    approximations = np.arange(-np.pi, np.pi + 1, np.pi / 2)
    forbidden_ranges = []

    for i, unsafe_angle in enumerate(unsafe_angles):
        for j, angle in enumerate(unsafe_angles[i]):
            if angle in approximations:
                continue
            dist = angle_distance_vector(angle, approximations)
            idx = np.argmin(dist)
            unsafe_angles[i][j] = approximations[idx]

    angle1 = unsafe_angles[:, 0]
    angle2 = unsafe_angles[:, 1]
    angle1_greater_mask = angle1 > angle2
    forbidden_ranges.extend(np.column_stack((angle1[~angle1_greater_mask], angle2[~angle1_greater_mask])))
    forbidden_ranges.extend(
        np.vstack((
            np.column_stack((angle1[angle1_greater_mask], np.full_like(angle1[angle1_greater_mask], math.pi))),
            np.column_stack((np.full_like(angle2[angle1_greater_mask], -math.pi), angle2[angle1_greater_mask]))
        ))
    )
    return forbidden_ranges


def compute_safe_angle_space(intersection_points, max_angle_change, x, wall_angles):
    robot_angles = get_robot_angles(x, max_angle_change)

    # convert points into angles and define the forbidden angles space
    forbidden_ranges = get_unsafe_angles(intersection_points, robot_angles, x)
    if wall_angles is not None:
        forbidden_ranges.extend(wall_angles)

    new_robot_angles = compute_ranges_difference(robot_angles, forbidden_ranges)
    if len(new_robot_angles) == 0:
        return None, robot_angles
    else:
        return new_robot_angles, robot_angles


def vo_negative_speed(obstacles, x, config):
    VELOCITY = np.abs(config.min_speed)
    ROBOT_RADIUS = config.robot_radius
    intersection_points = np.empty((0, 4), dtype=np.float64)

    square_obs, circle_obs, wall_obs = obstacles

    # CIRCULAR OBSTACLES
    circle_obs_x = np.array(circle_obs[0])
    circle_obs_rad = np.array(circle_obs[1])

    if len(circle_obs_x) != 0:
        # Calculate radii
        r1 = circle_obs_x[:, 3] * config.dt + circle_obs_rad + ROBOT_RADIUS
        r0 = np.full_like(r1, VELOCITY * config.dt)
        intersection_points, dist, mask = get_intersections_vectorized(x, circle_obs_x, r0, r1)

    # intersection_points = np.vstack((intersection_points, wall_int))
    intersection_data = check_circle_segment_intersect(x[:2], ROBOT_RADIUS + VELOCITY * config.dt, np.array(wall_obs[0]))
    valid_discriminant = intersection_data[0]
    wall_int = None
    if valid_discriminant.any():
        wall_int = np.array(wall_obs[0])[valid_discriminant]
        unsafe_wall_angles = get_unsafe_angles_wall(wall_int, x)
    else:
        unsafe_wall_angles = None

    if np.isnan(intersection_points).all() and wall_int is None:
        # all robot angles are safe
        return get_robot_angles(x, config.max_angle_change)
    else:
        x_copy = x.copy()
        val = x_copy[2] + np.pi
        x_copy[2] = val
        x_copy[2] = (x_copy[2] + math.pi) % (2 * math.pi) - math.pi
        safe_angles, robot_span = compute_safe_angle_space(intersection_points, config.max_angle_change, x_copy,
                                                           unsafe_wall_angles)
        return safe_angles


def new_get_spaces(obstacles, x, config, intersection_points, wall_angles):
    safe_angles, robot_span = compute_safe_angle_space(intersection_points, config.max_angle_change, x, wall_angles)
    if safe_angles is None:
        safe_angles = vo_negative_speed(obstacles, x, config)
        if safe_angles is None:
            vspace = [0.0, 0.0]
            safe_angles = [[-math.pi, math.pi]]
        else:
            vspace = [config.min_speed, config.min_speed]
    else:
        vspace = [config.max_speed, config.max_speed]

    velocity_space = [*([vspace] * len(safe_angles))]
    angle_space = [*safe_angles]

    return (angle_space,
            velocity_space)

def uniform_random_vo(node, planner):
    config = planner.environment.gym_env.config
    if len(node.state.obstacles) == 0:
        return uniform_random(node, planner)

    # Extract robot information
    x = node.state.x
    dt = planner.environment.config.dt
    ROBOT_RADIUS = planner.environment.config.robot_radius
    VMAX = 0.3
    wall_int = None

    # Extract obstacle information
    obstacles = node.state.obstacles
    # obs_x, obs_rad
    square_obs = [[], []]
    circle_obs = [[], []]
    wall_obs = [[], []]
    intersection_points = np.empty((0, 4), dtype=np.float64)
    for ob in obstacles:
        if ob.obs_type == "square":
            square_obs[0].append(ob.x)
            square_obs[1].append(ob.radius)
        elif ob.obs_type == "circle":
            circle_obs[0].append(ob.x)
            circle_obs[1].append(ob.radius)
        else:
            wall_obs[0].append(ob.x)
            wall_obs[1].append(ob.radius)

    # CIRCULAR OBSTACLES
    circle_obs_x = np.array(circle_obs[0])
    circle_obs_rad = np.array(circle_obs[1])

    if len(circle_obs_x) != 0:
        # Calculate radii
        r1 = circle_obs_x[:, 3] * dt + circle_obs_rad + ROBOT_RADIUS
        r0 = np.full_like(r1, VMAX * dt)

        # Calculate intersection points
        intersection_points, dist, mask = get_intersections_vectorized(x, circle_obs_x, r0, r1)

    # WALL OBSTACLES
    intersection_data = check_circle_segment_intersect(x[:2], ROBOT_RADIUS + VMAX * dt, np.array(wall_obs[0]))
    valid_discriminant = intersection_data[0]
    if valid_discriminant.any():
        wall_int = np.array(wall_obs[0])[valid_discriminant]
        unsafe_wall_angles = get_unsafe_angles_wall(wall_int, x)
    else:
        unsafe_wall_angles = None

    # If there are no intersection points
    if np.isnan(intersection_points).all() and wall_int is None:
        return uniform_random(node, planner)
    else:
        angle_space, velocity_space = new_get_spaces([square_obs, circle_obs, wall_obs], x, config, intersection_points,  wall_angles=unsafe_wall_angles)
        return sample_multiple_spaces(center=None, a_space=angle_space, v_space=velocity_space, number=1)[0]


def epsilon_uniform_uniform_vo(
        node: Any, planner: Planner, std_angle_rollout: float, eps=0.1
):
    # print(settings.FLAG)
    # planner.c.obstacles = [o.x for o in node.state.obstacles]
    prob = random.random()
    if prob <= 1 - eps:
        return uniform_towards_goal_vo(node, planner, std_angle_rollout)
    else:
        return uniform_random_vo(node, planner)
