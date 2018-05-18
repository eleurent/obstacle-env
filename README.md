# obstacle-env

An environment for *obstacle avoidance* tasks

<p align="center">
    <img src="docs/media/obstacle-env.gif"><br/>
    <em>A few episodes of the environment.</em>
</p>

[![Build Status](https://travis-ci.org/eleurent/obstacle-env.svg?branch=master)](https://travis-ci.org/eleurent/obstacle-env)

## Installation

`pip install --user git+https://github.com/eleurent/obstacle-env`

## Usage

```python
import obstacle_env

env = gym.make("obstacle-v0")

done = False
while not done:
    action = ... # Your agent code here
    obs, reward, done, _ = env.step(action)
    env.render()
```
