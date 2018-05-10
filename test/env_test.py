from __future__ import print_function, division

import gym
from gym.wrappers import Monitor
import obstacle_env


def test(episodes=3):
    env = gym.make('obstacle-v0')
    monitor = Monitor(env, directory='out', force=True)
    for i in range(episodes):
        print(i)
        observation = monitor.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            observation, reward, done, info = monitor.step(action)
            monitor.render()
            # print(observation)
    monitor.close()


if __name__ == "__main__":
    test()
