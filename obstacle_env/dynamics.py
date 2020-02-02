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
                      'acceleration': 5,
                      'perturbation': 0.1}

    def __init__(self, state=None, params=None):
        self.A = self.B = self.C = self.D = None
        self.discrete = self.continuous = None
        if params is None:
            params = self.DEFAULT_PARAMS
        self.params = params
        self.derivative = None

        self.continuous_dynamics()
        self.discrete_dynamics()

        if state is None:
            state = np.zeros((np.shape(self.A)[0], 1))
        self.state = state
        self.control = np.zeros((1, 1))
        self.desired_action = 0

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
            self.control = np.array([1])
        elif action == 'LEFT':
            self.control = np.array([-1])
        else:
            self.control = np.array([0])

    def step(self):
        """
            Step the dynamics
        """
        # self.state = np.dot(self.discrete[0], self.state)+np.dot(self.discrete[1], self.control)
        self.derivative = np.dot(self.continuous[0], self.state)+np.dot(self.continuous[1], self.control)
        self.state += self.derivative*self.params["dt"]

    def add_perturbation(self, np_random):
        perturbation = self.params["perturbation"] * np_random.randn(*self.state.shape)
        self.derivative += perturbation
        self.state += perturbation*self.params["dt"]

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

    ACTIONS = {
        0: 'IDLE',
        1: 'UP',
        2: 'DOWN',
        3: 'LEFT',
        4: 'RIGHT',
    }
    OTHER_ACTIONS = {
        5: 'UL',
        6: 'DR',
        7: 'RU',
        8: 'LD',
    }
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    def __init__(self, state=None, params=None):
        # self.ACTIONS.update(self.OTHER_ACTIONS)
        super(Dynamics2D, self).__init__(params)

        self.continuous_dynamics_2d()
        self.discrete_dynamics()

        if state is None:
            state = np.zeros((np.shape(self.A)[0], 1))
        self.state = state
        self.derivative = np.zeros(state.shape)
        self.control = np.zeros((2, 1))

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
        self.control = self.action_to_control(action)

    def action_to_control(self, action):
        if self.ACTIONS[action] == 'UP':
            control = np.array([[0], [-1]])
        elif self.ACTIONS[action] == 'DOWN':
            control = np.array([[0], [1]])
        elif self.ACTIONS[action] == 'RIGHT':
            control = np.array([[1], [0]])
        elif self.ACTIONS[action] == 'LEFT':
            control = np.array([[-1], [0]])
        elif self.ACTIONS[action] == 'UL':
            control = np.array([[-1], [-1]])
        elif self.ACTIONS[action] == 'DR':
            control = np.array([[1], [1]])
        elif self.ACTIONS[action] == 'RU':
            control = np.array([[1], [-1]])
        elif self.ACTIONS[action] == 'LD':
            control = np.array([[-1], [1]])
        else:
            control = np.array([[0], [0]])
        return control * self.params['acceleration']

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


class Dynamics2D2(Dynamics2D):
    """
        A second-order two-dimensional dynamical system.
    """

    def __init__(self, state=None, params=None):
        super().__init__(params)

        self.A0, self.phi, self.theta = None, None, None
        self.continuous_dynamics_2d()
        self.discrete_dynamics()

        if state is None:
            state = np.zeros((np.shape(self.A)[0], 1))
        self.state = state
        self.control = np.zeros((2, 1))

    def continuous_dynamics_2d(self):
        """
            Continuous state-space model
        """
        self.A0 = np.array([[0, 0, 1, 0],
                            [0, 0, 0, 1],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])
        self.phi = np.array([[[0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, -1, 0],
                              [0, 0, 0, 0]],
                             [[0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, -1]]])
        self.theta = np.array([0.3, 0.3])
        self.A = self.A0 + np.tensordot(self.theta, self.phi, axes=[0, 0])
        self.B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
        self.C = np.identity(self.A.shape[0])
        self.D = np.zeros(self.B.shape)
        self.continuous = (self.A, self.B, self.C, self.D)

    @property
    def position(self):
        return self.state[0:2, 0, np.newaxis]

    @property
    def velocity(self):
        return self.state[2:4, 0, np.newaxis]
