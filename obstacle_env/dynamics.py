from __future__ import print_function
from __future__ import division
import numpy as np
from scipy import signal


class Dynamics1D(object):
    """
        A fourth-order one-dimensional dynamical system.
    """
    DEFAULT_PARAMS = {'Cx': 1.0,
                      'w0': 50.0,
                      'zeta': 1.0,
                      'dt': 1/30}

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

    def position(self):
        return self.state[0, 0]


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
        self.command = np.zeros((1, 2))

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
        if action == 'UP':
            self.command = np.array([[0], [1]])
        elif action == 'DOWN':
            self.command = np.array([[0], [-1]])
        elif action == 'RIGHT':
            self.command = np.array([[1], [0]])
        elif action == 'LEFT':
            self.command = np.array([[-1], [0]])
        else:
            self.command = np.array([[0], [0]])
        self.command *= 2

    @property
    def position(self):
        return self.state[0::4, 0, np.newaxis]
