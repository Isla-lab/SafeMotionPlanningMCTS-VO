from abc import ABC, abstractmethod
from typing import Any

from bettergym.better_gym import BetterGym


class Planner(ABC):
    def __init__(self, environment: BetterGym):
        self.environment = environment

    @abstractmethod
    def plan(self, initial_state: Any):
        pass
