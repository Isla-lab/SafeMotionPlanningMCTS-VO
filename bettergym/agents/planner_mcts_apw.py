import math
from typing import Union, Any, Dict, Callable

import numpy as np

from bettergym.agents.planner import Planner
from bettergym.better_gym import BetterGym


class ActionNode:
    def __init__(self, action: np.ndarray):
        self.action = action
        self.action_bytes = action.tobytes()
        self.state_to_id: Dict[Any, int] = {}

    def __hash__(self):
        return hash(self.action_bytes)

    def __repr__(self):
        return np.array2string(self.action)

    def __eq__(self, other):
        if isinstance(other, ActionNode) and hash(self) == hash(other):
            return True
        return False


class StateNode:
    def __init__(self, state, node_id):
        self.id = node_id
        self.state = state
        self.actions = []
        self.num_visits_actions = np.array([], dtype=np.float64)
        self.a_values = np.array([], dtype=np.float64)
        self.num_visits: int = 0


class RolloutStateNode:
    def __init__(self, state):
        self.state = state


class MctsApw(Planner):
    def __init__(
        self,
        num_sim: int,
        c: float | int,
        environment: BetterGym,
        computational_budget: int,
        k: float | int,
        alpha: float | int,
        action_expansion_function: Callable,
        rollout_policy: Callable,
        discount: float | int = 1,
    ):
        """
        Mcts algorithm with Action Progressive Widening
        :param num_sim: number of simulations
        :param c: exploration constant of the UCB
        :param environment: the simulated environments
        :param computational_budget: maximum rollout depth
        :param k: first parameter of the action progressive widening
        :param alpha: second parameter of the action progressive widening
        :param action_expansion_function: function to choose which action to add to the tree
        :param discount: the discount factor of the mdp
        """
        super().__init__(environment)
        self.num_sim: int = num_sim
        self.c: float | int = c
        self.computational_budget: int = computational_budget
        self.discount: float | int = discount
        self.k = k
        self.alpha = alpha
        self.action_expansion_function = action_expansion_function
        self.rollout_policy = rollout_policy

        self.id_to_state_node = None
        self.last_id = None
        self.info = None

    def initialize_variables(self):
        self.id_to_state_node: dict[int, StateNode] = {}
        self.last_id = -1
        self.info = {
            "trajectories": [],
            "q_values": [],
            "actions": [],
            "visits": [],
            "rollout_values": [],
        }

    def get_id(self):
        self.last_id += 1
        return self.last_id

    def plan(self, initial_state: Any):
        self.initialize_variables()
        root_id = self.get_id()
        root_node = StateNode(initial_state, root_id)
        self.id_to_state_node[root_id] = root_node
        for sn in range(self.num_sim):
            self.info["trajectories"].append(np.array([initial_state.x]))
            # root should be at depth 0
            total_reward = self.simulate(state_id=root_id, depth=0)
            self.info["rollout_values"].append(total_reward)

        q_vals = np.divide(
            root_node.a_values,
            root_node.num_visits_actions,
            out=np.full_like(root_node.a_values, -np.inf),
            where=root_node.num_visits_actions != 0,
        )
        # DEBUG INFORMATION
        self.info["q_values"] = q_vals
        self.info["actions"] = root_node.actions
        self.info["visits"] = root_node.num_visits_actions

        # randomly choose between actions which have the maximum q value
        action_idx = np.random.choice(np.flatnonzero(q_vals == np.max(q_vals)))
        action = root_node.actions[action_idx].action
        return action, self.info

    def simulate(self, state_id: int, depth: int):
        node = self.id_to_state_node[state_id]
        node.num_visits += 1
        current_state = node.state

        if (
            len(node.actions) <= math.ceil(self.k * (node.num_visits**self.alpha))
            or len(node.actions) == 0
        ):
            new_action: np.ndarray = self.action_expansion_function(
                node=node, planner=self
            )

            # add child
            new_action_node = ActionNode(new_action)
            if new_action_node not in node.actions:
                node.actions.append(new_action_node)
                node.num_visits_actions = np.append(node.num_visits_actions, 0.0)
                node.a_values = np.append(node.a_values, 0.0)
        # UCB
        # Q + c * sqrt(ln(Parent_Visit)/Child_visit)
        q_vals = np.divide(
            node.a_values,
            node.num_visits_actions,
            out=np.full_like(node.a_values, np.inf),
            where=node.num_visits_actions != 0,
        )

        ucb_scores = q_vals + self.c * np.sqrt(
            np.divide(
                np.log(node.num_visits),
                node.num_visits_actions,
                out=np.full_like(node.num_visits_actions, np.inf),
                where=node.num_visits_actions != 0,
            )
        )
        # randomly choose between actions which have the maximum ucb value
        action_idx = np.random.choice(np.flatnonzero(ucb_scores == np.max(ucb_scores)))
        # get action corresponding to the index
        action_node = node.actions[action_idx]
        action = action_node.action
        # increase action visits
        node.num_visits_actions[action_idx] += 1

        current_state, r, terminal, _, _ = self.environment.step(current_state, action)
        new_state_id = action_node.state_to_id.get(current_state, None)
        self.info["trajectories"][-1] = np.vstack(
            (self.info["trajectories"][-1], current_state.x)
        )

        prev_node = node
        if (
            new_state_id is None
            and depth + 1 < self.computational_budget
            and not terminal
        ):
            # Leaf Node
            state_id = self.get_id()
            # Initialize State Data
            node = StateNode(current_state, state_id)
            self.id_to_state_node[state_id] = node
            action_node.state_to_id[current_state] = state_id
            node.num_visits += 1
            # Do Rollout
            # the value returned by the rollout is already discounted
            total_rwrd = r + self.discount * self.rollout(current_state, depth + 1)
            prev_node.a_values[action_idx] += total_rwrd
            return total_rwrd
        else:
            # Node in the tree
            state_id = new_state_id
            if terminal or depth + 1 >= self.computational_budget:
                # self.info["rollout_values"].append(r)
                prev_node.a_values[action_idx] += r
                return r
            else:
                total_rwrd = r + self.discount * self.simulate(state_id, depth + 1)
                # BackPropagate
                # since I only need action nodes for action selection I don't care about the value of State nodes
                prev_node.a_values[action_idx] += total_rwrd
                return total_rwrd

    def rollout(self, current_state, curr_depth) -> Union[int, float]:
        terminal = False
        trajectory = []
        total_reward = 0
        starting_depth = 0
        while not terminal and curr_depth + starting_depth != self.computational_budget:
            chosen_action = self.rollout_policy(RolloutStateNode(current_state), self)
            current_state, r, terminal, _, _ = self.environment.step(
                current_state, chosen_action
            )
            total_reward += r * pow(self.discount, starting_depth)
            trajectory.append(current_state.x)
            starting_depth += 1

        self.info["trajectories"][-1] = np.vstack(
            (self.info["trajectories"][-1], np.array(trajectory))
        )
        # self.info["rollout_values"].append(total_reward)
        return total_reward
