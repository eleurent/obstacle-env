
class SingleTrajectoryAgent(object):
    """
        Execute a given list of actions
    """
    def __init__(self, actions, default_action):
        self.actions = actions
        self.default_action = default_action

    def act(self):
        if self.actions:
            return self.actions.pop(0)
        else:
            return self.default_action



