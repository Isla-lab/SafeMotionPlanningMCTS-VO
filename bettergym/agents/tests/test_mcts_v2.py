from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from bettergym.agents.planner_mcts import Mcts


class UnitTestMcts(TestCase):
    def test_rollout_until_terminal(self):
        # given
        expected_reward = 5
        np.random.seed(1)
        s = 0
        env = Mock()
        env.get_actions.return_value = np.array([1, 2, 3, 4])
        # s', r, terminal, truncated, info
        env.step.return_value = (1, expected_reward, True, None, None)

        planner = Mcts(
            num_sim=100,
            c=1,
            s0=s,
            environment=env,
            computational_budget=100,
            discount=0.9,
        )
        # when
        value = planner.rollout(s)
        # then
        env.get_actions.assert_called_with(s)
        env.step.assert_called_with(s, 2)
        self.assertEqual(value, 5)

    def test_rollout_until_budget_end(self):
        # given
        np.random.seed(1)
        s = 0
        env = Mock()
        env.get_actions.return_value = np.array([1, 2, 3, 4])
        # s', r, terminal, truncated, info
        env.step.side_effect = [
            (3, 5, False, None, None),
            (1, 8, False, None, None),
            (1, 3, False, None, None),
        ]

        planner = Mcts(
            num_sim=100,
            c=1,
            s0=s,
            environment=env,
            computational_budget=3,
            discount=0.9,
        )
        # when
        value = planner.rollout(s)
        # then
        self.assertEqual(value, 3)

    def test_simulate(self):
        # given
        np.random.seed(1)
        s = 0
        env = Mock()
        env.get_actions.return_value = np.array([1, 2, 3, 4])
        # s', r, terminal, truncated, info
        env.step.side_effect = [
            (3, 5, False, None, None),
            (1, 8, False, None, None),
            (1, 3, False, None, None),
        ]

        planner = Mcts(
            num_sim=100,
            c=1,
            s0=s,
            environment=env,
            computational_budget=3,
            discount=0.9,
        )
        # when
        value = planner.rollout(s)
        # then
        self.assertEqual(value, 3)
