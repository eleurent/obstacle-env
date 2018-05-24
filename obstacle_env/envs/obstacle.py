from __future__ import print_function, division
import copy
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

from obstacle_env.dynamics import Dynamics2D
from obstacle_env.graphics import EnvViewer
from obstacle_env.scene import Scene2D, PolarGrid


class ObstacleEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    SIMULATION_FREQUENCY = 10
    POLICY_FREQUENCY = 2
    GRID_CELLS = 16
    COLLISION_REWARD = -10

    def __init__(self):
        # Seeding
        self.np_random = None
        self.seed()

        # Dynamics
        params = Dynamics2D.DEFAULT_PARAMS.update({'dt': 1 / self.SIMULATION_FREQUENCY})
        self.dynamics = Dynamics2D(params=params)
        self.dynamics.desired_action = self.np_random.randint(1, len(self.dynamics.ACTIONS))

        # Scene
        self.scene = Scene2D()
        self.scene.create_random_scene(np_random=self.np_random)
        self.grid = PolarGrid(self.scene, cells=self.GRID_CELLS)

        # Spaces
        self.action_space = spaces.Discrete(len(self.dynamics.ACTIONS))
        low_obs = np.hstack((-self.dynamics.terminal_velocity * np.ones((2,)),
                             -self.dynamics.params['acceleration'] * np.ones((2,)),
                             0 * np.ones((self.grid.cells,)),))
        high_obs = np.hstack((self.dynamics.terminal_velocity * np.ones((2,)),
                              self.dynamics.params['acceleration'] * np.ones((2,)),
                              self.grid.MAXIMUM_RANGE * np.ones((self.grid.cells,)),))
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # Viewer
        self.viewer = None
        self.automatic_rendering_callback = None
        self.should_update_rendering = True
        self.rendering_mode = 'human'
        self.enable_auto_render = False

        self.steps = 0
        self.done = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
            Reset the environment to it's initial configuration
        :return: the observation of the reset state
        """
        self.steps = 0
        self.scene.create_random_scene(np_random=self.np_random)
        self.dynamics.state *= 0
        self.dynamics.crashed = False
        self.dynamics.desired_action = self.np_random.randint(1, len(self.dynamics.ACTIONS))

        return self._observation()

    def step(self, action):
        """
            Perform an action and step the environment dynamics.

            The action is executed and the dynamics are stepped.
        :param int action: the action performed
        :return: a tuple (observation, reward, terminal, info)
        """
        # Forward action to the dynamics
        self.dynamics.act(action)

        # Simulate
        for k in range(int(self.SIMULATION_FREQUENCY // self.POLICY_FREQUENCY)):
            self.dynamics.step()
            self.dynamics.check_collisions(self.scene)

            # Render simulation
            self._automatic_rendering()

            # Stop at terminal states
            if self.done or self._is_terminal():
                break

        self.enable_auto_render = False

        self.steps += 1
        obs = self._observation()
        reward = self._reward(action)
        terminal = self._is_terminal()
        info = {}

        return obs, reward, terminal, info

    def _automatic_rendering(self):
        """
            Automatically render the intermediate frames while an action is still ongoing.
            This allows to render the whole video and not only single steps corresponding to agent decision-making.

            If a callback has been set, use it to perform the rendering. This is useful for the environment wrappers
            such as video-recording monitor that need to access these intermediate renderings.
        """
        if self.viewer is not None and self.enable_auto_render:
            self.should_update_rendering = True

            if self.automatic_rendering_callback:
                self.automatic_rendering_callback()
            else:
                self.render(self.rendering_mode)

    def render(self, mode='human'):
        """
            Render the environment.

            Create a viewer if none exists, and use it to render an image.
        :param mode: the rendering mode
        """
        self.rendering_mode = mode

        if self.viewer is None:
            self.viewer = EnvViewer(self, record_video=False)

        self.enable_auto_render = True

        # If the frame has already been rendered, do nothing
        if self.should_update_rendering:
            self.viewer.display()

        if mode == 'rgb_array':
            image = self.viewer.get_image()
            self.viewer.handle_events()
            return image
        elif mode == 'human':
            self.viewer.handle_events()
        self.should_update_rendering = False

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
        desired_command = self.dynamics.action_to_command(self.dynamics.desired_action) / \
                          self.dynamics.params['acceleration']

        ranges = self.grid.trace(self.dynamics.position)
        ranges = np.minimum(ranges, self.grid.MAXIMUM_RANGE) / self.grid.MAXIMUM_RANGE
        observation = np.vstack((velocities, desired_command, ranges))
        return np.ravel(observation)

    def _reward(self, action):
        """
            Return the reward associated with performing a given action and ending up in the current state.

        :param action: the last action performed
        :return: the reward
        """
        desired_command = self.dynamics.action_to_command(self.dynamics.desired_action)
        command = self.dynamics.action_to_command(action)
        return (1 - np.linalg.norm(desired_command - command) / (2 * self.dynamics.params['acceleration'])) ** 2 + \
               self.COLLISION_REWARD * self.dynamics.crashed

    def _is_terminal(self):
        """
            Check whether the current state is a terminal state
        :return:is the state terminal
        """
        return self.dynamics.crashed

    def __deepcopy__(self, memo):
        """
            Perform a deep copy but without copying the environment viewer.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ['viewer', 'automatic_rendering_callback']:
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, None)
        return result

    def simplified(self):
        """
            Return a simplified copy of the environment where distant vehicles have been removed from the road.

            This is meant to lower the policy computational load while preserving the optimal actions set.

        :return: a simplified environment state
        """
        state_copy = copy.deepcopy(self)
        obstacles_close = [obstacle for obstacle in state_copy.scene.obstacles
                           if np.linalg.norm(obstacle['position'] - self.dynamics.position) < 2*state_copy.grid.MAXIMUM_RANGE]
        state_copy.scene.obstacles = obstacles_close
        return state_copy
