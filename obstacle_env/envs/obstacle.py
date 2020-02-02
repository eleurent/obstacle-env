from __future__ import print_function, division
import copy
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

from obstacle_env.dynamics import Dynamics2D2
from obstacle_env.graphics import EnvViewer
from obstacle_env.scene import Scene2D, PolarGrid


class ObstacleEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    config = {
        "simulation_frequency": 10,
        "policy_frequency": 2,
        "grid_cells": 16,
        "collision_reward": -0.5,
        "observation_type": "grid",
        "observation_noise": 0.1,
    }

    def __init__(self):
        # Seeding
        self.np_random = None
        self.seed()

        # Dynamics
        params = Dynamics2D2.DEFAULT_PARAMS.update({'dt': 1 / self.config["simulation_frequency"]})
        self.dynamics = Dynamics2D2(params=params)
        self.dynamics.desired_action = self.np_random.randint(1, len(self.dynamics.ACTIONS))

        # Scene
        self.scene = Scene2D()
        self.scene.create_random_scene(np_random=self.np_random)
        self.grid = PolarGrid(self.scene, cells=self.config["grid_cells"])

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

        self.lpv = None

        self.steps = 0
        self.time = 0
        self.done = False
        self.trajectory = []
        self.interval_trajectory = []
        self.automatic_record_callback = None

    def configure(self, config):
        self.config.update(config)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
            Reset the environment to it's initial configuration
        :return: the observation of the reset state
        """
        self.steps = 0
        self.time = 0
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
        # Simulate
        for k in range(int(self.config["simulation_frequency"] // self.config["policy_frequency"])):
            if action is not None and \
                    self.time % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
                # Forward action to the dynamics
                self.dynamics.act(action)

            self.store_data()
            state = self.dynamics.state.copy()
            self.dynamics.step()
            self.dynamics.add_perturbation(self.np_random)
            self.dynamics.check_collisions(self.scene)
            if self.automatic_record_callback:
                observation = self.dynamics.derivative + \
                              self.config["observation_noise"] * self.np_random.randn(*self.dynamics.derivative.shape)
                self.automatic_record_callback(state, observation, self.dynamics.control)
            if self.lpv:
                self.lpv.set_control((self.dynamics.B @ self.dynamics.control).squeeze(-1))
                self.lpv.step(1 / self.config["simulation_frequency"])
            self.time += 1

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
        reward = reward if not terminal else 0
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
            - 2 normalized desired controls
            - a vector of normalized range measurements in N directions
        :return: the observation
        """
        if self.config["observation_type"] == "grid":
            velocities = self.dynamics.velocity / self.dynamics.terminal_velocity
            desired_control = self.dynamics.action_to_control(self.dynamics.desired_action) / \
                              self.dynamics.params['acceleration']

            ranges = self.grid.trace(self.dynamics.position)
            ranges = np.minimum(ranges, self.grid.MAXIMUM_RANGE) / self.grid.MAXIMUM_RANGE
            observation = np.vstack((velocities, desired_control, ranges))
            return np.ravel(observation)
        else:
            return self.dynamics.state

    def _reward(self, action):
        """
            Return the reward associated with performing a given action and ending up in the current state.

        :param action: the last action performed
        :return: the reward
        """
        if self.lpv is not None:
            return self.pessimistic_reward(action, self.lpv.x_i_t)
        if self.scene.goal:
            d = np.linalg.norm(self.scene.goal["position"] - self.dynamics.position)
            d0 = 10
            reward = (action > 0)/(1+d/d0)
        else:
            desired_control = self.dynamics.action_to_control(self.dynamics.desired_action)
            control = self.dynamics.action_to_control(action)
            reward = (1 - np.linalg.norm(desired_control - control) / (2 * self.dynamics.params['acceleration'])) ** 2
        return remap(reward + self.config["collision_reward"] * self.dynamics.crashed, [-1, 1], [0, 1])

    def pessimistic_reward(self, action, interval):
        """
            Return the reward associated with performing a given action and ending up in the current state.

        :param interval: a state interval [min, max]
        :return: the reward
        """
        reward = 0
        collision = self.pessimistic_is_terminal(interval)
        corners = [[interval[0, 0], interval[0, 1]],
                   [interval[0, 0], interval[1, 1]],
                   [interval[1, 0], interval[0, 1]],
                   [interval[1, 0], interval[1, 1]]]
        if self.scene.goal:
            d = np.linalg.norm(self.scene.goal["position"] - np.mean(corners, axis=0)[:, np.newaxis])
            d0 = 10
            reward = (action > 0)/(1+d/d0)
        return remap(reward + self.config["collision_reward"] * collision, [-1, 1], [0, 1])

    def pessimistic_is_terminal(self, interval):
        if not self.dynamics.crashed:
            corners = np.array([[interval[0, 0], interval[0, 1]],
                               [interval[0, 0], interval[1, 1]],
                               [interval[1, 0], interval[0, 1]],
                               [interval[1, 0], interval[1, 1]]])
            for position in corners:
                for obstacle in self.scene.obstacles:
                    if np.linalg.norm(position[:, np.newaxis] - obstacle['position']) < obstacle['radius']:
                        self.dynamics.crashed = True
                        break
        return self.dynamics.crashed

    def _is_terminal(self):
        """
            Check whether the current state is a terminal state
        :return:is the state terminal
        """
        if self.lpv:
            return self.pessimistic_is_terminal(self.lpv.x_i_t)
        return self.dynamics.crashed

    def __deepcopy__(self, memo):
        """
            Perform a deep copy but without copying the environment viewer.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ['viewer', 'automatic_rendering_callback', 'automatic_record_callback']:
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

    def store_data(self):
        self.trajectory.append(self.dynamics.state)
        if self.lpv:
            self.interval_trajectory.append(self.lpv.x_i_t)


def remap(v, x, y, clip=False):
    if x[1] == x[0]:
        return y[0]
    out = y[0] + (v-x[0])*(y[1]-y[0])/(x[1]-x[0])
    if clip:
        out = constrain(out, y[0], y[1])
    return out