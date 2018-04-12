from __future__ import print_function, division

import gym
import obstacle_env


def test():
    env = gym.make('obstacle-v0')

    observation = env.reset()
    actions = [1, 1, 4]*10
    for i in range(len(actions)):
        observation, reward, done, info = env.step(actions[i])
        env.render()
        print(env.dynamics.state)


if __name__ == "main":
    test()
