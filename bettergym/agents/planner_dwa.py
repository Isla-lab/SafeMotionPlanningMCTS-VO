import math
from typing import Any

import numpy as np


class Dwa:
    def __init__(self, environment):
        self.environment = environment

    def calc_dynamic_window(self, x, config):
        """
        calculation dynamic window based on current state x
        """

        # Dynamic window from robot specification
        Vs = [config.min_speed, config.max_speed,
              -config.max_yaw_rate, config.max_yaw_rate]

        # Dynamic window from motion model
        Vd = [x[3] - config.max_accel * config.dt,
              x[3] + config.max_accel * config.dt,
              x[4] - config.max_delta_yaw_rate * config.dt,
              x[4] + config.max_delta_yaw_rate * config.dt]

        #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
              max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        return dw

    def predict_trajectory(self, x_init, v, y, config):
        """
        predict trajectory with an input
        """

        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= config.predict_time:
            u_copy = np.array([v, x[2] + y * config.dt])
            x = self.environment.robot_motion(x, u_copy)
            x[3] = y
            x[4] = v
            trajectory = np.vstack((trajectory, x))
            time += config.dt

        return trajectory

    def calc_to_goal_cost(self, trajectory, goal, config):
        """
            calc to goal cost with angle difference and pos difference
        """

        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        # if np.linalg.norm([dx, dy]) <= config.robot_radius:
        #     return -float("Inf")
        error_angle = math.atan2(dy, dx)
        cost_angle = abs(error_angle - trajectory[-1, 2]) / math.pi
        # cost_angle = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

        cost_dist = np.linalg.norm([dx, dy])  # / np.linalg.norm([goal[0]-trajectory[0, 0], goal[1]-trajectory[0, 1]])

        if cost_dist < config.robot_radius:
            return -100.

        # return np.linalg.norm([cost_dist, cost_angle])
        return cost_dist

    def calc_obstacle_cost(self, trajectory, ob, config, robot_ob):
        """
        calc obstacle cost inf: collision
        """
        try:
            ox = ob[:, 0]
            oy = ob[:, 1]
            dx = trajectory[:, 0] - ox[:, None]
            dy = trajectory[:, 1] - oy[:, None]
            r1 = np.hypot(dx, dy).min()
            theta1 = np.arctan2(dy, dx).min()
        except:
            theta1 = 100.
            r1 = 100.
        try:
            rob_ox = robot_ob[:, 0]
            rob_oy = robot_ob[:, 1]
            dx_rob = trajectory[:, 0] - rob_ox[:, None]
            dy_rob = trajectory[:, 1] - rob_oy[:, None]
            r2 = np.hypot(dx_rob, dy_rob).min()
            theta2 = np.arctan2(dy_rob, dx_rob).min()
        except:
            theta2 = 100.
            r2 = 100.

        if np.array(r1 <= config.robot_radius + config.obs_size).any() or np.array(
                r2 <= config.robot_radius + config.robot_radius).any() or np.array(
            trajectory[:, 0] + config.robot_radius > config.right_limit).any() or np.array(
            trajectory[:, 1] + config.robot_radius > config.upper_limit).any():
            return 1.0
            # return float("Inf")

        # min_theta = min(theta1.min(), theta2)
        # cost_theta = math.pi / abs(min_theta)

        # min_r1 = r1.min()
        # min_r2 = r2
        # cost_dist = 0.
        # if min_r1 <= min_r2:
        #     min_r1_arg = [np.where(r1 == min_r1)[0][0], np.where(r1 == min_r1)[1][0]]
        #     cost_dist = np.linalg.norm([dx[min_r1_arg[0], min_r1_arg[1]], dy[min_r1_arg[0], min_r1_arg[1]]]) / r1[min_r1_arg[0], min_r1_arg[1]]
        # else:
        #     min_r2_arg = [np.where(r2 == min_r2)[0][0], np.where(r2 == min_r2)[1][0]]
        #     cost_dist = np.linalg.norm([dx[min_r2_arg[0], min_r2_arg[1]], dy[min_r2_arg[0], min_r2_arg[1]]]) / r2[min_r2_arg[0], min_r2_arg[1]]

        # return np.sqrt(cost_dist**2 + cost_theta**2)
        return 0.0  # OK
        # return 1.0 / min(min_r1, min_r2)  # OK

    def calc_control_and_trajectory(self, x, dw, config, goal, ob, robot_ob):
        """
        calculation final input with dynamic window
        """

        x_init = x[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])

        # v_list = np.arange(dw[0], dw[1], config.v_resolution)
        # if 0. not in v_list:
        #     v_list = np.append(v_list, 0.)
        # y_list = np.arange(dw[2], dw[3], config.yaw_rate_resolution)
        # if x_init[2] not in y_list:
        #     y_list = np.append(y_list, x_init[2])

        # evaluate all trajectory with sampled input in dynamic window
        for v in np.arange(dw[0], dw[1], config.v_resolution):
            for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):

                trajectory = self.predict_trajectory(x_init, v, y, config)
                # calc cost
                to_goal_cost = config.to_goal_cost_gain * self.calc_to_goal_cost(trajectory, goal, config)
                if to_goal_cost == -float("Inf"):
                    return best_u, best_trajectory
                speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
                ob_cost = config.obstacle_cost_gain * self.calc_obstacle_cost(trajectory, ob, config, robot_ob)

                # final_cost = to_goal_cost
                # print("to_goal_cost")
                # print(to_goal_cost)
                # print("ob_cost")
                # print(ob_cost)
                final_cost = np.nan_to_num(to_goal_cost + speed_cost + ob_cost)

                # search minimum trajectory
                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [v, y]
                    best_trajectory = trajectory
                    if abs(best_u[0]) < config.robot_stuck_flag_cons \
                            and abs(x[3]) < config.robot_stuck_flag_cons:
                        # to ensure the robot do not get stuck in
                        # best v=0 m/s (in front of an obstacle) and
                        # best omega=0 rad/s (heading to the goal with
                        # angle difference of 0)
                        best_u[1] = -config.max_delta_yaw_rate

            # print(min_cost)
        return best_u, best_trajectory

    def plan(self, initial_state: Any, obs, robot_obs):
        dw = self.calc_dynamic_window(initial_state.x, self.environment.config)
        u, trajectory = self.calc_control_and_trajectory(
            initial_state.x,
            dw,
            self.environment.config,
            initial_state.goal,
            obs,
            robot_obs
        )
        return u, trajectory
