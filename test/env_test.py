from __future__ import print_function, division
import gym

import obstacle_env
from obstacle_env.dynamics import Dynamics2D
from test.single_trajectory import SingleTrajectoryAgent


def test_random(episodes=1):
    env = gym.make('obstacle-v0')
    for i in range(episodes):
        env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
    env.close()


def test_up(episodes=1):
    env = gym.make('obstacle-v0')
    agent = SingleTrajectoryAgent([], Dynamics2D.ACTIONS_INDEXES['UP'])
    for i in range(episodes):
        env.reset()
        done = False
        while not done:
            action = agent.act()
            observation, reward, done, info = env.step(action)
    env.close()


def test_left(episodes=1):
    env = gym.make('obstacle-v0')
    agent = SingleTrajectoryAgent([], Dynamics2D.ACTIONS_INDEXES['LEFT'])
    for i in range(episodes):
        env.reset()
        done = False
        while not done:
            action = agent.act()
            observation, reward, done, info = env.step(action)
    env.close()


def test_down(episodes=1):
    env = gym.make('obstacle-v0')
    agent = SingleTrajectoryAgent([], Dynamics2D.ACTIONS_INDEXES['DOWN'])
    for i in range(episodes):
        env.reset()
        done = False
        while not done:
            action = agent.act()
            observation, reward, done, info = env.step(action)
    env.close()


def test_right(episodes=1):
    env = gym.make('obstacle-v0')
    agent = SingleTrajectoryAgent([], Dynamics2D.ACTIONS_INDEXES['RIGHT'])
    for i in range(episodes):
        env.reset()
        done = False
        while not done:
            action = agent.act()
            observation, reward, done, info = env.step(action)
    env.close()


def test_idle(episodes=1):
    env = gym.make('obstacle-v0')
    agent = SingleTrajectoryAgent([], Dynamics2D.ACTIONS_INDEXES['IDLE'])
    for i in range(episodes):
        env.reset()
        done = False
        while not done:
            action = agent.act()
            observation, reward, done, info = env.step(action)
    env.close()
