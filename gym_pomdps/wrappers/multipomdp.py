import gym
from gym_pomdps.envs import POMDP

import numpy as np


class MultiPOMDP(gym.Wrapper):
    """Simulates multiple POMDP trajectories at the same time."""
    def __init__(self, env, ntrajectories=None):
        if not isinstance(env, POMDP):
            raise TypeError(f'Input env is not a POMDP ({type(env)})')

        super().__init__(env)
        self.ntrajectories = ntrajectories

    def reset(self, ntrajectories=None):
        self.state = self.reset_functional(ntrajectories)

    def step(self, action):
        self.state, *ret = self.step_functional(self.state, action)
        return ret

    def reset_functional(self, ntrajectories=None):
        if ntrajectories is None:
            ntrajectories = self.ntrajectories

        if ntrajectories is None:
            raise ValueError('Number of trajectories ({ntrajectories}) not set.')

        if self.env.start is None:
            state = self.np_random.randint(
                self.state_space.n, size=self.ntrajectories)
        else:
            state = self.np_random.multinomial(
                1, self.env.start, size=self.ntrajectories).argmax(1)
        return state

    def step_functional(self, state, action):
        shape = state.shape

        # mask for non-previourly-completed episodes
        mask = state != -1

        # episodic POMDP does not support any -1
        if not (self.env.episodic or mask.all()):
            raise ValueError(f'Non-episodic POMDP does not support '
                             'uninitialized states (-1)'
                             'Perhaps the POMDP was not reset?')

        # all -1s is never ok
        if not mask.any():
            raise ValueError(f'All states are all not initialized (-1).  '
                             'Perhaps the POMDP was not reset?')

        # unmasked states should be within bounds
        s, a = state[mask], action[mask]
        if not ((0 <= s) & (s < self.state_space.n)).all():
            raise ValueError(f'State (min={s.min()}, max={s.max()}) '
                             'outside of bounds.  '
                             'Perhaps the POMDP was not reset?')

        s1 = np.array([self.np_random.multinomial(1, p).argmax()
                       for p in self.env.T[s, a]])
        o = np.array([self.np_random.multinomial(1, p).argmax()
                      for p in self.env.O[s, a, s1]])
        r = self.env.R[s, a, s1, o]

        if self.env.episodic:
            d = self.env.D[s, a]
            s1[d] = -1
        else:
            d = np.zeros(mask.sum(), dtype=bool)

        state1 = np.full(shape, -1)
        state1[mask] = s1

        obs = np.full(shape, -1)
        obs[mask] = o

        reward = np.full(shape, float('nan'))
        reward[mask] = r

        done = np.full(shape, True)
        done[mask] = d

        reward_cat = np.full(shape, -1)
        reward_cat[mask] = [self.rewards_dict[r_] for r_ in r]
        info = dict(reward_cat=reward_cat)

        return state1, obs, reward, done, info
