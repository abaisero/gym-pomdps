import gym
import numpy as np

from gym_pomdps.envs import POMDP


class MultiPOMDP(gym.Wrapper):
    """Simulates multiple POMDP trajectories at the same time."""

    def __init__(self, env, ntrajectories=None):
        if not isinstance(env.unwrapped, POMDP):
            raise TypeError(f'Input env is not a POMDP ({type(env)})')

        super().__init__(env)
        self.ntrajectories = ntrajectories
        self.state = np.full([ntrajectories], -1)

    def reset(self, ntrajectories=None):
        self.state = self.reset_functional(ntrajectories)

    def step(self, action):
        self.state, *ret = self.step_functional(self.state, action)
        return ret

    def reset_functional(self, ntrajectories=None):
        if ntrajectories is None:
            ntrajectories = self.ntrajectories

        if ntrajectories is None:
            raise ValueError(
                'Number of trajectories ({ntrajectories}) not set.'
            )

        if self.env.start is None:
            state = self.np_random.randint(
                self.state_space.n, size=self.ntrajectories
            )
        else:
            state = self.np_random.multinomial(
                1, self.env.start, size=self.ntrajectories
            ).argmax(1)
        return state

    def step_functional(self, state, action):
        if ((state == -1) != (action == -1)).any():
            raise ValueError('Invalid state-action pair ({state}, {action}).')

        shape = state.shape
        mask = state != -1

        state1 = np.full(shape, -1)
        obs = np.full(shape, -1)
        reward = np.full(shape, 0.0)
        reward_cat = np.full(shape, -1)
        done = np.full(shape, True)

        if mask.any():
            # unmasked states should be within bounds
            s, a = state[mask], action[mask]
            assert ((s >= 0) & (s < self.state_space.n)).all()
            assert ((a >= 0) & (a < self.action_space.n)).all()

            s1 = np.array(
                [
                    self.np_random.multinomial(1, p).argmax()
                    for p in self.env.T[s, a]
                ]
            )
            o = np.array(
                [
                    self.np_random.multinomial(1, p).argmax()
                    for p in self.env.O[s, a, s1]
                ]
            )
            # NOTE below is the same but unified in single sampling op; requires TO
            # s1, o = np.array([
            #     divmod(
            #         self.np_random.multinomial(1, p.ravel()).argmax(),
            #         self.observation_space.n,
            #     )
            #     for p in self.env.TO[s, a]
            # ]).T

            r = self.env.R[s, a, s1, o]

            if self.env.episodic:
                d = self.env.D[s, a]
                s1[d] = -1
            else:
                d = np.zeros(mask.sum(), dtype=bool)

            state1[mask] = s1
            obs[mask] = o
            reward[mask] = r
            done[mask] = d

            reward_cat[mask] = [self.rewards_dict[r_] for r_ in r]
            info = dict(reward_cat=reward_cat)

        info = dict(reward_cat=reward_cat)
        return state1, obs, reward, done, info
