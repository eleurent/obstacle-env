import gym
from gym import spaces

from obstacle_env.dynamics import Dynamics2D
from obstacle_env.graphics import EnvViewer
from obstacle_env.scene import Scene2D, PolarGrid


class ObstacleEnv(gym.Env):
    SIMULATION_FREQUENCY = 30
    POLICY_FREQUENCY = 1

    ACTIONS = {0: 'UP',
               1: 'DOWN',
               2: 'LEFT',
               3: 'RIGHT'}
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    def __init__(self):
        self.scene = Scene2D()
        self.dynamics = Dynamics2D()
        self.grid = PolarGrid(self.scene)
        self.viewer = None
        self.done = False

    def observation(self):
        """
            Return the observation of the current state, which must be consistent with self.observation_space.
        :return: the observation
        """
        return self.grid.trace(self.dynamics.position)

    def reward(self, action):
        """
            Return the reward associated with performing a given action and ending up in the current state.

        :param action: the last action performed
        :return: the reward
        """
        return 0

    def is_terminal(self):
        """
            Check whether the current state is a terminal state
        :return:is the state terminal
        """
        return False

    def reset(self):
        """
            Reset the environment to it's initial configuration
        :return: the observation of the reset state
        """
        raise NotImplementedError()

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
            if self.done or self.is_terminal():
                break

        obs = self.observation()
        reward = self.reward(action)
        terminal = self.is_terminal()
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

    def get_available_actions(self):
        """
            Get the list of currently available actions.

            Lane changes are not available on the boundary of the road, and velocity changes are not available at
            maximal or minimal velocity.

        :return: the list of available actions
        """
        actions = [self.ACTIONS_INDEXES['IDLE']]
        li = self.vehicle.lane_index
        if li > 0 \
                and self.road.lanes[li-1].is_reachable_from(self.vehicle.position):
            actions.append(self.ACTIONS_INDEXES['LANE_LEFT'])
        if li < len(self.road.lanes) - 1 \
                and self.road.lanes[li+1].is_reachable_from(self.vehicle.position):
            actions.append(self.ACTIONS_INDEXES['LANE_RIGHT'])
        if self.vehicle.velocity_index < self.vehicle.SPEED_COUNT - 1:
            actions.append(self.ACTIONS_INDEXES['FASTER'])
        if self.vehicle.velocity_index > 0:
            actions.append(self.ACTIONS_INDEXES['SLOWER'])
        return actions

    # def change_agents_to(self, agent_type):
    #     """
    #         Change the type of all agents on the road
    #     :param agent_type: The new type of agents
    #     :return: a new RoadMDP with modified agents type
    #     """
    #     state_copy = copy.deepcopy(self)
    #     vehicles = state_copy.ego_vehicle.road.vehicles
    #     for i, v in enumerate(vehicles):
    #         if v is not state_copy.ego_vehicle and not isinstance(v, Obstacle):
    #             vehicles[i] = agent_type.create_from(v)
    #     return state_copy

    def simplified(self):
        """
            Return a simplified copy of the environment where distant vehicles have been removed from the road.

            This is meant to lower the policy computational load while preserving the optimal actions set.

        :return: a simplified environment state
        """
        state_copy = copy.deepcopy(self)
        ev = state_copy.vehicle
        close_vehicles = []
        for v in state_copy.road.vehicles:
            if -self.PERCEPTION_DISTANCE/2 < ev.lane_distance_to(v) < self.PERCEPTION_DISTANCE:
                close_vehicles.append(v)
        state_copy.road.vehicles = close_vehicles
        return state_copy

    def __deepcopy__(self, memo):
        """
            Perform a deep copy but without copying the environment viewer.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k != 'viewer':
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, None)
        return result
