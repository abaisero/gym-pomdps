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
        self.state = self.reset_functional()

    def step(self, action):
        self.state, *ret = self.step_functional(self.state, action)
        return ret

    def reset_functional(self):
        if self.env.start is None:
            state = self.np_random.randint(self.state_space.n,
                                           size=self.ntrajectories)
        else:
            state = self.np_random.multinomial(1, self.env.start,
                                               size=self.ntrajectories).argmax(1)
        return state

    def step_functional(self, state, action):
        state1 = np.array([self.np_random.multinomial(1, p).argmax()
                           for p in self.env.T[state, action]])
        obs = np.array([self.np_random.multinomial(1, p).argmax()
                        for p in self.env.O[state, action, state1]])
        reward = self.env.R[state, action, state1, obs]
        done = np.zeros(self.ntrajectories, dtype=bool)

        return state1, obs, reward, done, {}
