import gym
import obstacle_env


def test():
    env = gym.make('obstacle-v0')

    for _ in range(5):
        env.step(0)
        env.render()


if __name__ == "main":
    test()
