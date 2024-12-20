import random
from typing import Tuple

import gymnasium as gym
import numpy as np


class RecyclingRobot(gym.Env):
    """
    Class that implements the environments Recycling Robot of the book: 'Reinforcement
    Learning: an introduction, Sutton & Barto'. Example 3.3 page 52 (second edition).

    Attributes
    ----------
        observation_space : int
            define the number of possible actions of the environments
        action_space: int
            define the number of possible states of the environments
        actions: dict
            a dictionary that translate the 'action code' in human languages
        states: dict
            a dictionary that translate the 'state code' in human languages

    Methods
    -------
        reset( self )
            method that reset the environments to an initial state; returns the state
        step( self, action )
            method that perform the action given in input, computes the next state and the reward; returns
            next_state and reward
        render( self )
            method that print the internal state of the environments
    """

    def __init__(self):
        # Loading the default parameters
        self.alfa = 0.7
        self.beta = 0.7
        self.r_search = 0.5
        self.r_wait = 0.2

        # Defining the environments variables
        self.observation_space = 2
        self.action_space = 3
        self.actions = {0: "SEARCH", 1: "WAIT", 2: "RECHARGE"}
        self.states = {0: "HIGH", 1: "LOW"}
        self.state = None
        self.reset()

    def reset(self, **kwargs):
        self.state = random.choice(range(self.observation_space))
        return self.state

    def step(self, action):
        # the first key is the state and the second is the actions
        transition_table = {
            # high state
            0: {
                # search action
                0: {
                    "probabilities": [self.alfa, 1 - self.alfa],
                    "states": [0, 1],
                    "rewards": [self.r_search, self.r_search],
                },
                # wait action
                1: {"probabilities": [1], "states": [0], "rewards": [self.r_wait]},
                # recharge action
                2: {"probabilities": [1], "states": [0], "rewards": [0]},
            },
            # low state
            1: {
                # search action
                0: {
                    "probabilities": [1 - self.beta, self.beta],
                    "states": [0, 1],
                    "rewards": [-3, self.r_search],
                },
                # wait action
                1: {"probabilities": [1], "states": [1], "rewards": [self.r_wait]},
                # recharge action
                2: {"probabilities": [1], "states": [0], "rewards": [0]},
            },
        }

        probs, states, rewards = transition_table[self.state][action].values()
        state_idx = random.choices(population=range(len(states)), weights=probs)[0]
        self.state = states[state_idx]
        reward = rewards[state_idx]
        return self.state, reward, False, False, None


class RecyclingRobot2(gym.Env):
    """
    Class that implements the environments Recycling Robot of the book: 'Reinforcement
    Learning: an introduction, Sutton & Barto'. Example 3.3 page 52 (second edition).

    Attributes
    ----------
        observation_space : int
            define the number of possible actions of the environments
        action_space: int
            define the number of possible states of the environments
        actions: dict
            a dictionary that translate the 'action code' in human languages
        states: dict
            a dictionary that translate the 'state code' in human languages

    Methods
    -------
        reset( self )
            method that reset the environments to an initial state; returns the state
        step( self, action )
            method that perform the action given in input, computes the next state and the reward; returns
            next_state and reward
        render( self )
            method that print the internal state of the environments
    """

    def __init__(self):
        # Loading the default parameters
        self.alfa = 0.7
        self.beta = 0.7
        self.r_search = 0.5
        self.r_wait = 0.2

        # Defining the environments variables
        self.observation_space = 2
        self.action_space = 3
        self.actions = {0: "SEARCH", 1: "WAIT", 2: "RECHARGE"}
        self.states = {0: "HIGH", 1: "LOW"}
        self.state = None
        self.reset()

    def get_actions(self, state):
        # HIGH
        if state == 0:
            return np.array([0, 1])
        else:  # LOW
            return np.array([0, 1, 2])

    def reset(self, **kwargs):
        self.state = random.choice(range(self.observation_space))
        return self.state

    def step(self, action):
        # the first key is the state and the second is the actions
        transition_table = {
            # high state
            0: {
                # search action
                0: {
                    "probabilities": [self.alfa, 1 - self.alfa],
                    "states": [0, 1],
                    "rewards": [self.r_search, self.r_search],
                },
                # wait action
                1: {"probabilities": [1], "states": [0], "rewards": [self.r_wait]},
                # recharge action
                2: {"probabilities": [1], "states": [0], "rewards": [0]},
            },
            # low state
            1: {
                # search action
                0: {
                    "probabilities": [1 - self.beta, self.beta],
                    "states": [0, 1],
                    "rewards": [-3, self.r_search],
                },
                # wait action
                1: {"probabilities": [1], "states": [1], "rewards": [self.r_wait]},
                # recharge action
                2: {"probabilities": [1], "states": [0], "rewards": [0]},
            },
        }

        probs, states, rewards = transition_table[self.state][action].values()
        state_idx = random.choices(population=range(len(states)), weights=probs)[0]
        self.state = states[state_idx]
        reward = rewards[state_idx]
        return self.state, reward, False, False, None
