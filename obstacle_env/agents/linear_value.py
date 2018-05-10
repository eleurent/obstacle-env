import numpy as np
import operator

class LinearEstimator(object):
    ALPHA = 0.05

    def __init__(self, observation=None):
        self.parameters = None
        if observation is not None:
            self.parameters = np.random.random(np.shape(np.transpose(observation)))

    def predict(self, observations):
        if self.parameters is None:
            self.parameters = np.random.random(np.shape(np.transpose(observations[np.newaxis, 0, :])))
        return np.dot(observations, self.parameters)

    def update(self, observations, values):
        gradient = observations.T.dot(values - self.predict(observations)) / len(observations)
        self.parameters += self.ALPHA * gradient


class LinearAgent(object):
    GAMMA = 0.99

    def __init__(self, action_space):
        self.action_space = action_space
        self.models = {action: LinearEstimator() for action in self.action_space}

    def update(self, observations, actions, rewards, next_observations, done):
        # if done:
        #     target = rewards
        # else:
        #     next_values = [model.predict(next_observations) for model in self.models.values()]
        #     target = rewards + self.GAMMA * max(next_values)
        # self.models[actions].update(observations, target)

        for action in self.models.keys():
            sel = actions == action
            sel_obs = observations.select(sel)
            sel_reward = rewards.select(sel)
            sel_next_obs = next_observations.select(sel)
            sel_done = done.select(sel)
            next_values = np.hstack((model.predict(next_observations) for model in self.models.values()))
            best_value = np.amax(next_values, axis = 1, keepdims=True)
            targets = rewards + self.GAMMA*best_value
            targets[done == 1] = rewards

            self.models[action].update(observations, targets)

    def act(self, observation):
        action_values = {action: model.predict(observation) for (action, model) in self.models.items()}
        action = max(action_values.items(), key=operator.itemgetter(1).predict(observation))[0]
        return action


def test_estimator():
    observations = np.array([[1, 2, 3], [-4, 5, 6], [-3, 4, -1]])
    obs = observations[np.newaxis, 0, :]
    value = [[1.0], [2.0], [3.0]]
    model = LinearEstimator(obs)
    for _ in range(500):
        model.update(observations, value)
    print("updated parameters", model.parameters)
    print("updated value\n", )
    assert np.testing.assert_array_almost_equal(model.predict(observations), value)


def test_agent():
    o1 = np.array([[1, 0, 0]])
    o2 = np.array([[0, 1, 0]])
    o3 = np.array([[0, 0, 1]])
    o4 = np.array([[0, 0, -1]])
    observations = np.vstack((o1, o2, o1, o2))
    rewards = np.array([[1, 10, -10, -10]])
    next_observations = np.vstack((o2, o3, o4, o4))
    done = np.array([[False, True, True, True]])
    # transitions = [(o1, 0, 1, o2, False),
    #                (o2, 0, 10, o3, True),
    #                (o1, 1, -10, o4, True),
    #                (o2, 1, -10, o4, True)]

    agent = LinearAgent([0, 1])
    for _ in range(500):
        agent.update(observations, rewards, next_observations, done)
    action = agent.act(o1)
    np.testing.assert_almost_equal(agent.models[0].predict(o1), 11, decimal=1)
    np.testing.assert_almost_equal(agent.models[1].predict(o1), -10, decimal=1)
    assert o1 == 0


if __name__ == '__main__':
    test_estimator()
    test_agent()
