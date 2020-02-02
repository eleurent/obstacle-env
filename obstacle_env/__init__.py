from gym.envs.registration import register

register(
    id='obstacle-v0',
    entry_point='obstacle_env.envs:ObstacleEnv',
    max_episode_steps=20,
)