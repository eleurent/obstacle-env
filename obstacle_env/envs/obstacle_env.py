from __future__ import print_function, division
import numpy as np

import gym
from gym import spaces

from obstacle_env.dynamics import Dynamics2D
from obstacle_env.graphics import EnvViewer
from obstacle_env.scene import Scene2D, PolarGrid


class ObstacleEnv(gym.Env):
    SIMULATION_FREQUENCY = 30
    POLICY_FREQUENCY = 2
    MAX_DURATION = 15

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
        self.desired_action = self.ACTIONS_INDEXES['RIGHT']
        self.action_space = spaces.Discrete(len(self.ACTIONS))

        low_obs = np.hstack((-self.dynamics.terminal_velocity * np.ones((2,)),
                             -self.dynamics.params['acceleration'] * np.ones((2,)),
                             0 * np.ones((self.grid.cells,)),))
        high_obs = np.hstack((self.dynamics.terminal_velocity * np.ones((2,)),
                              self.dynamics.params['acceleration'] * np.ones((2,)),
                              self.grid.MAXIMUM_RANGE * np.ones((self.grid.cells,)),))
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        self.steps = 0

    def reset(self):
        """
            Reset the environment to it's initial configuration
        :return: the observation of the reset state
        """
        self.steps = 0
        self.dynamics.state *= 0
        self.dynamics.crashed = False

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
            self.dynamics.check_collisions(self.scene)

            # Render simulation
            if self.viewer is not None:
                self.render()

            # Stop at terminal states
            if self.done or self._is_terminal():
                break

        self.steps += 1
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

            It is a single vector of size 4+N composed of:
            - 2 normalized velocities
            - 2 normalized desired commands
            - a vector of normalized range measurements in N directions
        :return: the observation
        """
        velocities = self.dynamics.velocity / self.dynamics.terminal_velocity
        desired_command = self.dynamics.action_to_command(self.ACTIONS[self.desired_action]) / \
                          self.dynamics.params['acceleration']

        ranges = self.grid.trace(self.dynamics.position)
        ranges = np.minimum(ranges, self.grid.MAXIMUM_RANGE)/self.grid.MAXIMUM_RANGE
        observation = np.vstack((velocities, desired_command, ranges))
        return np.ravel(observation)

    def _reward(self, action):
        """
            Return the reward associated with performing a given action and ending up in the current state.

        :param action: the last action performed
        :return: the reward
        """
        desired_command = self.dynamics.action_to_command(self.ACTIONS[self.desired_action])
        command = self.dynamics.action_to_command(self.ACTIONS[action])
        return (1 - np.linalg.norm(desired_command - command)/(2*self.dynamics.params['acceleration']))**2

    def _is_terminal(self):
        """
            Check whether the current state is a terminal state
        :return:is the state terminal
        """
        return self.dynamics.crashed or self.steps >= self.MAX_DURATION
