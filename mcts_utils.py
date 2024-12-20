import math

import numpy as np


def uniform_random(node, planner):
    state = node.state
    config = planner.environment.gym_env.config
    action = np.random.uniform(
        low=np.array(
            [config.min_speed, state.x[2] - config.max_angle_change], dtype=np.float64
        ),
        high=np.array(
            [config.max_speed, state.x[2] + config.max_angle_change], dtype=np.float64
        ),
    )
    action[1] = (action[1] + math.pi) % (2 * math.pi) - math.pi
    return action


def compute_int_vectorized(r0, r1, d, x0, x1, y0, y1):
    """
    Vectorized computation of intersection points between two circles
    :param r0: radius of the first circle
    :param r1: radius of the second circle
    :param d: distance between the two circle centers
    :param x0: x position of the first circle center
    :param x1: x position of the second circle center
    :param y0: y position of the first circle center
    :param y1: y position of the second circle center
    :return: array of coordinates of the two intersection points
    """
    # https://stackoverflow.com/a/55817881
    a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(r0 ** 2 - a ** 2)
    x2 = x0 + a * (x1 - x0) / d
    y2 = y0 + a * (y1 - y0) / d
    x3 = x2 + h * (y1 - y0) / d
    y3 = y2 - h * (x1 - x0) / d
    x4 = x2 - h * (y1 - y0) / d
    y4 = y2 + h * (x1 - x0) / d

    return np.column_stack((np.column_stack((x3, y3)), np.column_stack((x4, y4))))


def check_circle_segment_intersect(robot_pos, robot_radius, segments):
    r2 = robot_radius ** 2
    A = np.array(segments[:, 2] - segments[:, 0])
    B = np.array(segments[:, 3] - segments[:, 1])
    C = np.array(segments[:, 0] - robot_pos[0])
    D = np.array(segments[:, 1] - robot_pos[1])
    a = A ** 2 + B ** 2
    b = 2 * (A * C + B * D)
    c = C ** 2 + D ** 2 - r2
    discriminant = b ** 2 - 4 * a * c

    valid_discriminant = discriminant >= 0
    return valid_discriminant, discriminant, b, a, segments, A, B


def find_circle_segment_intersections(discriminant, valid_discriminant, b, a, segments, A, B):
    sqrt_discriminant = np.sqrt(discriminant[valid_discriminant])
    t1 = (-b[valid_discriminant] + sqrt_discriminant) / (2 * a[valid_discriminant])
    t2 = (-b[valid_discriminant] - sqrt_discriminant) / (2 * a[valid_discriminant])

    valid_t1 = np.logical_and(t1 >= 0, t1 <= 1)
    valid_t2 = np.logical_and(t2 >= 0, t2 <= 1)

    if np.any(valid_t1):
        xi_t1 = segments[valid_discriminant, 0] + t1[valid_t1] * A[valid_discriminant]
        yi_t1 = segments[valid_discriminant, 1] + t1[valid_t1] * B[valid_discriminant]
    else:
        return np.empty((0, 4))

    if np.any(valid_t2):
        xi_t2 = segments[valid_discriminant, 0] + t2[valid_t2] * A[valid_discriminant]
        yi_t2 = segments[valid_discriminant, 1] + t2[valid_t2] * B[valid_discriminant]
    else:
        return np.empty((0, 4))

    return np.column_stack((np.column_stack((xi_t1, yi_t1)), np.column_stack((xi_t2, yi_t2))))


def angle_distance_vector(a1, angles):
    # Compute the absolute difference between the angles
    diff = np.abs(a1 - angles)

    # Ensure the shortest distance is used by considering wrap-around

    diff = np.minimum(diff, 2 * math.pi - diff)

    return diff

def get_tangents(robot_state, obs_r, obstacles, d):
    # Calculate angles from the robot to each obstacle
    alphas = np.arctan2(robot_state[1]-obstacles[:, 1], robot_state[0]-obstacles[:, 0])

    # Create rotation matrices for each angle
    matrices = [np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]]) for alpha in alphas]

    # Calculate the angles for the tangent points
    phi = np.arccos(obs_r / d)

    # Calculate the tangent points on the obstacles
    obs_r_expanded = np.expand_dims(obs_r, 1)
    P1 = obs_r_expanded * np.stack((np.cos(phi), np.sin(phi)), axis=1)
    P2 = obs_r_expanded * np.stack((np.cos(-phi), np.sin(-phi)), axis=1)

    # Apply the rotation matrices and translate the points to the robot's position
    new_P1 = np.array([matrices[i] @ P1[i] + robot_state[:2] for i in range(len(P1))])
    new_P2 = np.array([matrices[i] @ P2[i] + robot_state[:2] for i in range(len(P2))])

    # Combine the tangent points into a single array and return them
    intersections = np.hstack((new_P1, new_P2))
    return intersections

def get_intersections_vectorized(x, obs_x, r0, r1):
    x_exp = np.expand_dims(x, 1)
    d = np.hypot(obs_x[:, 0] - x_exp[0, :], obs_x[:, 1] - x_exp[1, :])

    # Non-intersecting
    no_intersection = d > r0 + r1

    # One circle within the other
    one_within_other = d < np.abs(r0 - r1)

    # Coincident circles
    coincident = np.logical_and(d == 0, r0 == r1)

    intersecting = np.logical_not(np.logical_or.reduce((no_intersection, one_within_other, coincident)))

    # Compute intersection points
    if np.any(intersecting):
        intersection_points = compute_int_vectorized(
            r0[intersecting],
            r1[intersecting],
            d[intersecting],
            x_exp[0, :],
            obs_x[intersecting, 0],
            x_exp[1, :],
            obs_x[intersecting, 1],
        )
    else:
        intersection_points = None

    output_vec = np.empty((len(d), 4))
    output_vec[:] = None
    output_vec[np.logical_or(one_within_other, coincident), :] = np.inf
    output_vec[intersecting, :] = intersection_points

    return output_vec, d, intersecting
