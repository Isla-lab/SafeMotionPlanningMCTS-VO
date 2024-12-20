import copy
import math

import numpy as np
from scipy.optimize import dual_annealing, Bounds

from bettergym.agents.planner import Planner
from bettergym.better_gym import BetterGym
from bettergym.environments.env import State


def update_state(x0, u, timestep):
    """
    Computes the states of the system after applying a sequence of control signals u on
    initial state x0
    """
    # print("UPDATING STATE")
    N = int(len(u) / 2)
    lower_triangular_ones_matrix = np.tril(np.ones((N, N)))
    kron = np.kron(lower_triangular_ones_matrix, np.eye(2))

    new_state = np.vstack([np.eye(2)] * int(N)) @ x0 + kron @ u * timestep

    return new_state


class Nmpc(Planner):

    def __init__(self, environment: BetterGym, obs_rad=0.2, gamma=0.99, horizon_length=80):
        super().__init__(environment)
        config = environment.config
        self.timestep = config.dt
        self.robot_radius = config.robot_radius
        self.obs_radius = obs_rad
        self.gamma = gamma
        self.vmax = config.max_speed
        self.vmin = config.min_speed
        self.horizon_length = horizon_length
        self.nmpc_timestep = 1.0
        self.goal_cost = -100.
        self.coll_cost = 100.
        bl_corner = np.array([config.bottom_limit, config.left_limit])
        ur_corner = np.array([config.upper_limit, config.right_limit])
        self.max_eudist = math.hypot(
            ur_corner[0] - bl_corner[0], ur_corner[1] - bl_corner[1]
        )

    def tracking_cost_discrete(self, x, p_desired):
        # print(x)
        # print("COMPUTING TRACKING COST")
        cost = 0.
        for i in range(self.horizon_length):
            # cost += np.linalg.norm(x[2 * i: 2 * (i + 1)] - p_desired) / self.map_size / np.sqrt(2) * (self.gamma ** (i))
            cost += (np.linalg.norm(x[2 * i: 2 * (i + 1)] - p_desired) / self.max_eudist) * (self.gamma ** i)
            if np.linalg.norm(x[2 * i: 2 * (i + 1)] - p_desired) < self.robot_radius:
                cost += self.goal_cost * (self.gamma ** i)
                break
        return cost

    def collision_cost(self, x0, x1, rad):
        """
        Cost of collision between two robot_state
        """
        d = np.linalg.norm(x0 - x1)
        # cost = Qc / (1 + np.exp(kappa * (d - ROBOT_RADIUS - OBS_RADIUS)))
        cost = 0
        if d <= self.robot_radius + rad:
            cost = self.coll_cost
        return cost

    def total_collision_cost(self, robot, obstacles, obst_radii):
        total_cost = 0
        for i in range(self.horizon_length):
            for j in range(np.shape(obstacles)[0]):
                obstacle = obstacles[j]
                rob = robot[2 * i: 2 * i + 2]
                total_cost += self.collision_cost(rob, obstacle[:2], obst_radii[j])
        return total_cost

    def total_cost(self, u, robot_state, obstacle_predictions, p_desired, obst_radii):
        # now = time.time()
        x_robot = update_state(robot_state, u, self.nmpc_timestep)
        # print("TIME FOR UPDATE STATE")
        # print(time.time() - now)
        # now = time.time()
        # c1 = tracking_cost(x_robot, xref)
        c1 = self.tracking_cost_discrete(x_robot, p_desired)
        # print("TIME FOR GOAL")
        # print(time.time() - now)
        # now = time.time()
        c2 = self.total_collision_cost(x_robot, obstacle_predictions, obst_radii)
        # print("TIME FOR COLLISION")
        # print(time.time() - now)
        # now = time.time()
        # print("==========")
        total = c1 + c2
        return total

    def compute_velocity(self, robot_state, obstacle_predictions, p_desired, obst_radii):
        """
        Computes control velocity of the copter
        """
        # u0 = np.array(2 * [0] * HORIZON_LENGTH)
        u0 = np.zeros(2 * self.horizon_length, )
        ub = [0, 0]
        lb = [0, 0]
        if (p_desired - robot_state)[0] > 0:
            lb[0] = np.sqrt(2) / 2 * self.vmin
            ub[0] = np.sqrt(2) / 2 * self.vmax
        else:
            lb[0] = -np.sqrt(2) / 2 * self.vmax
            ub[0] = -np.sqrt(2) / 2 * self.vmin
        if (p_desired - robot_state)[1] > 0:
            lb[1] = np.sqrt(2) / 2 * self.vmin
            ub[1] = np.sqrt(2) / 2 * self.vmax
        else:
            lb[1] = -np.sqrt(2) / 2 * self.vmax
            ub[1] = -np.sqrt(2) / 2 * self.vmin
        for h in range(self.horizon_length):
            u0[2 * h] = np.random.uniform(low=lb[0], high=ub[0])
            u0[2 * h + 1] = np.random.uniform(low=lb[1], high=ub[1])

        def cost_fn(u):
            return self.total_cost(
                u, robot_state, obstacle_predictions, p_desired, obst_radii)

        bounds = Bounds(lb * self.horizon_length, ub * self.horizon_length)
        res = dual_annealing(cost_fn, bounds=bounds, maxiter=50, no_local_search=True)

        velocity = res.x[:2]
        # print("FOUND OPTIMUM WITH COST " + str(res.fun))
        return velocity, res.x

    def plan(self, initial_state: State):
        # compute velocity using nmpc
        fixed_obstacles = np.array([o.x for o in initial_state.obstacles])
        fixed_obst_radii = np.array([o.radius for o in initial_state.obstacles])
        dist = np.hypot(fixed_obstacles[:, 0] - initial_state.x[0], fixed_obstacles[:, 1] - initial_state.x[1])
        mask = dist < 5
        vel, _ = self.compute_velocity(
            initial_state.x[:2], copy.deepcopy(fixed_obstacles[mask]), initial_state.goal, copy.deepcopy(fixed_obst_radii[mask])
        )
        return vel, None
