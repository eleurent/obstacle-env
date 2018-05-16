from __future__ import print_function, division
from gym import wrappers
from highway_env.wrappers.monitor import MonitorV2
import gym

import obstacle_env


def run(episodes=1):
    env = gym.make('obstacle-v0')
    env = MonitorV2(env, 'out', video_callable=wrappers.monitor.capped_cubic_video_schedule)

    for i in range(episodes):
        env.reset()
        done = False
        while not done:
            action = env.unwrapped.dynamics.desired_action
            observation, reward, done, info = env.step(action)
            env.render()
    env.close()


if __name__ == '__main__':
    run()
