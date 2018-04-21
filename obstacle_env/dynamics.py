from __future__ import print_function, division
import numpy as np
from scipy import signal


class Dynamics1D(object):
    """
        A fourth-order one-dimensional dynamical system.
    """
    DEFAULT_PARAMS = {'Cx': 1.0,
                      'w0': 50.0,
                      'zeta': 1.0,
                      'dt': 1/30,
                      'acceleration': 5}

    def __init__(self, state=None, params=None):
        self.A = self.B = self.C = self.D = None
        self.discrete = self.continuous = None
        if params is None:
            params = self.DEFAULT_PARAMS
        self.params = params

        self.continuous_dynamics()
        self.discrete_dynamics()

        if state is None:
            state = np.zeros((np.shape(self.A)[0], 1))
        self.state = state
        self.command = np.zeros((1, 1))

        self.crashed = False

    def continuous_dynamics(self):
        """
            Continuous state-space model
        """
        self.A = np.array(
            [[0, 1, 0, 0],
             [0, -self.params['Cx'], 1, 0],
             [0, 0, 0, 1],
             [0, 0, -self.params['w0']**2, -2*self.params['zeta']*self.params['w0']]])
        self.B = np.array([[0], [0], [0], [self.params['w0']**2]])
        self.C = np.identity(np.shape(self.A)[0])
        self.D = np.zeros(np.shape(self.B))

        self.continuous = (self.A, self.B, self.C, self.D)

    def discrete_dynamics(self):
        """
            Discrete state-space model
        """

        self.discrete = signal.cont2discrete(self.continuous, self.params['dt'], method='zoh')

    def act(self, action):
        if action == 'RIGHT':
            self.command = np.array([1])
        elif action == 'LEFT':
            self.command = np.array([-1])
        else:
            self.command = np.array([0])

    def step(self):
        """
            Step the dynamics
        """
        self.state = np.dot(self.discrete[0], self.state)+np.dot(self.discrete[1], self.command)

    @property
    def position(self):
        return self.state[0, 0]

    @property
    def terminal_velocity(self):
        return self.params['acceleration']/self.params['Cx']


class Dynamics2D(Dynamics1D):
    """
        A fourth-order two-dimensional dynamical system.
    """

    def __init__(self, state=None):
        super(Dynamics2D, self).__init__()

        self.continuous_dynamics_2d()
        self.discrete_dynamics()

        if state is None:
            state = np.zeros((np.shape(self.A)[0], 1))
        self.state = state
        self.command = np.zeros((2, 1))

    def continuous_dynamics_2d(self):
        """
            Continuous state-space model
        """
        a, b = self.A, self.B

        self.A = np.vstack((np.hstack((np.array(a), np.zeros(np.shape(a)))),
                            np.hstack((np.zeros(np.shape(a)), np.array(a)))))
        self.B = np.vstack((np.hstack((np.array(b), np.zeros(np.shape(b)))),
                            np.hstack((np.zeros(np.shape(b)), np.array(b)))))
        self.C = np.identity(np.shape(self.A)[0])
        self.D = np.zeros(np.shape(self.B))

        self.continuous = (self.A, self.B, self.C, self.D)

    def act(self, action):
        self.command = self.action_to_command(action)

    def action_to_command(self, action):
        if action == 'UP':
            command = np.array([[0], [1]])
        elif action == 'DOWN':
            command = np.array([[0], [-1]])
        elif action == 'RIGHT':
            command = np.array([[1], [0]])
        elif action == 'LEFT':
            command = np.array([[-1], [0]])
        else:
            command = np.array([[0], [0]])
        return command * self.params['acceleration']

    def check_collisions(self, scene):
        position = self.position
        for obstacle in scene.obstacles:
            if np.linalg.norm(position - obstacle['position']) < obstacle['radius']:
                self.crashed = True

    @property
    def position(self):
        return self.state[0::4, 0, np.newaxis]

    @property
    def velocity(self):
        return self.state[1::4, 0, np.newaxis]
