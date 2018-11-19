import gym
from gym import spaces
from gym.utils import seeding

from rl_parsers.pomdp import parse
import numpy as np


class POMDP(gym.Env):
    """Environment specified by POMDP file."""

    def __init__(self, path, episodic=False, seed=None):
        self.episodic = episodic
        self.seed(seed)

        with open(path) as f:
            model = parse(f.read())

        if model.values == 'cost':
            raise ValueError('Unsupported `cost` values.')

        self.discount = model.discount
        self.state_space = spaces.Discrete(len(model.states))
        self.action_space = spaces.Discrete(len(model.actions))
        self.observation_space = spaces.Discrete(len(model.observations))
        self.reward_range = model.R.min(), model.R.max()

        if model.start is None:
            self.start = np.full(self.state_space.n, 1 / self.state_space.n)
        else:
            self.start = model.start
        self.T = model.T.transpose(1, 0, 2).copy()
        self.O = np.stack([model.O] * self.state_space.n)
        self.R = model.R.transpose(1, 0, 2, 3).copy()

        if episodic:
            self.D = model.reset.T.copy()  # only if episodic

        self.state = None

    def seed(self, seed):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def reset(self):
        self.state = self.np_random.multinomial(1, self.start).argmax().item()

    def step(self, action):
        assert self.state is not None, 'State has not been initialized'

        state1 = self.np_random.multinomial(
            1, self.T[self.state, action]).argmax().item()
        obs = self.np_random.multinomial(
            1, self.O[self.state, action, state1]).argmax().item()
        reward = self.R[self.state, action, state1, obs].item()

        if self.episodic:
            done = self.D[self.state, action].item()
        else:
            done = False

        if done:
            self.state = None
        else:
            self.state = state1

        return obs, reward, done, {}
