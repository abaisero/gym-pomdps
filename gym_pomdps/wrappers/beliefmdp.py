import gym

from gym_pomdps.belief import belief_init, belief_step, expected_reward
from gym_pomdps.envs import POMDP

__all__ = ['BeliefMDP']


class BeliefMDP(gym.Wrapper):
    """Belief-MDP associated with input POMDP."""

    def __init__(self, env):
        if not isinstance(env.unwrapped, POMDP):
            raise TypeError(f'Env is not a POMDP (got {type(env)}).')

        super().__init__(env)
        self.belief = None

    def reset(self):  # pylint: disable=arguments-differ
        self.env.reset()
        self.belief = belief_init(self.env, self.state.shape)
        return self.belief

    def step(self, action):
        r = expected_reward(self.env, self.belief, action)
        o, _, done, info = self.env.step(action)
        self.belief = belief_step(self.env, self.belief, action, o)

        return self.belief, r, done, info
