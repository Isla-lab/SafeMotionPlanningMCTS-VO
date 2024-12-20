from typing import Any

from bettergym.agents.planner import Planner
from bettergym.better_gym import BetterGym


class RandomPlanner(Planner):
    def __init__(self, environment: BetterGym):
        super().__init__(environment)

    def plan(self, initial_state: Any):
        available_actions = self.environment.get_actions(initial_state)
        return available_actions.sample(), {}
