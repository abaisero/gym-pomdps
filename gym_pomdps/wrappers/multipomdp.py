import gym
from ..envs import POMDP

import numpy as np


class MultiPOMDP(gym.Wrapper):
    """Simulates multiple POMDP trajectories at the same time."""
    def __init__(self, env, ntrajectories):
        assert isinstance(env, POMDP)
        if env.episodic:
            raise NotImplementedError(
                'MultiPOMDP does not support episodic PODMPs yet.')

        super().__init__(env)
        self.ntrajectories = ntrajectories

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def reset(self):
        if self.env.start is None:
            self.state = self.np_random.randint(
                self.state_space.n, size=self.ntrajectories)
        else:
            self.state = self.np_random.multinomial(
                1, self.env.start, size=self.ntrajectories).argmax(1)

    def step(self, action):
        state1 = np.array([self.np_random.multinomial(1, p).argmax()
                           for p in self.env.T[self.state, action]])
        obs = np.array([self.np_random.multinomial(1, p).argmax()
                        for p in self.env.O[self.state, action, state1]])
        reward = self.env.R[self.state, action, state1, obs]
        done = np.zeros(self.ntrajectories, dtype=bool)

        self.state = state1
        return obs, reward, done, {}
