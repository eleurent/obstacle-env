from __future__ import print_function, division
import numpy as np

import gym
from gym import spaces

from obstacle_env.dynamics import Dynamics2D
from obstacle_env.graphics import EnvViewer
from obstacle_env.scene import Scene2D, PolarGrid


class ObstacleEnv(gym.Env):
    SIMULATION_FREQUENCY = 30
    POLICY_FREQUENCY = 1

    ACTIONS = {
        0: 'IDLE',
        1: 'UP',
        2: 'DOWN',
        3: 'LEFT',
        4: 'RIGHT'}
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    def __init__(self):
        self.scene = Scene2D()
        self.dynamics = Dynamics2D()
        self.grid = PolarGrid(self.scene)
        self.viewer = None
        self.done = False
        self.desired_action = self.ACTIONS_INDEXES['IDLE']

    def reset(self):
        """
            Reset the environment to it's initial configuration
        :return: the observation of the reset state
        """
        self.dynamics.state *= 0

        return self._observation()

    def step(self, action):
        """
            Perform an action and step the environment dynamics.

            The action is executed and the dynamics are stepped.
        :param int action: the action performed
        :return: a tuple (observation, reward, terminal, info)
        """
        # Forward action to the dynamics
        self.dynamics.act(self.ACTIONS[action])

        # Simulate
        for k in range(int(self.SIMULATION_FREQUENCY // self.POLICY_FREQUENCY)):
            self.dynamics.step()

            # Render simulation
            if self.viewer is not None:
                self.render()

            # Stop at terminal states
            if self.done or self._is_terminal():
                break

        obs = self._observation()
        reward = self._reward(action)
        terminal = self._is_terminal()
        info = {}

        return obs, reward, terminal, info

    def render(self, mode='human'):
        """
            Render the environment.

            Create a viewer if none exists, and use it to render an image.
        :param mode: the rendering mode
        """
        if self.viewer is None:
            self.viewer = EnvViewer(self, record_video=False)

        if mode == 'rgb_array':
            raise NotImplementedError()
        elif mode == 'human':
            self.viewer.display()
            self.viewer.handle_events()

    def close(self):
        """
            Close the environment.

            Will close the environment viewer if it exists.
        """
        self.done = True
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None

    def _observation(self):
        """
            Return the observation of the current state, which must be consistent with self.observation_space.

            It is a single vector of size 37 composed of:
            - 2 velocities
            - a one-hot encoding of the 5 actions
            - a vector of range measurements in 30 directions
        :return: the observation
        """
        velocities = self.dynamics.velocity
        ranges = self.grid.trace(self.dynamics.position)

    def _reward(self, action):
        """
            Return the reward associated with performing a given action and ending up in the current state.

        :param action: the last action performed
        :return: the reward
        """
        desired_command = self.dynamics.action_to_command(self.desired_action)
        command = self.dynamics.action_to_command(action)
        return np.linalg.norm(desired_command - command)

    def _is_terminal(self):
        """
            Check whether the current state is a terminal state
        :return:is the state terminal
        """
        return False
