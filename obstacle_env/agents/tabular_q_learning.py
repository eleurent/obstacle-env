from __future__ import print_function
from __future__ import division
import numpy as np


class TabularQLearningAgent(object):
    def __init__(self, state):
        self.state = state

    def quantized_state(self):
        observation = self.state.observe()
