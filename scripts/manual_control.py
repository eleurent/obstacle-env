from __future__ import print_function, division
from gym import wrappers
from gym.wrappers import Monitor
import gym

import obstacle_env


def run(episodes=1):
    env = gym.make('obstacle-v0')
    env = Monitor(env, 'out', force=True)

    for _ in range(episodes):
        env.reset()
        env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame  # Capture in-between frames
        done = False
        while not done:
            action = env.unwrapped.dynamics.desired_action
            observation, reward, done, info = env.step(action)
            env.render()
    env.close()


if __name__ == '__main__':
    run()
