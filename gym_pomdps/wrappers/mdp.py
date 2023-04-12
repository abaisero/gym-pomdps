import gym
from gym_pomdps.envs.pomdp import POMDP


class MDP(gym.Wrapper):
    """Exposes the underlying MDP of a POMDP"""

    def __init__(self, env):
        if not isinstance(env.unwrapped, POMDP):
            raise TypeError(f'Env is not a POMDP (got {type(env)}).')

        super().__init__(env)

    def reset(self):  # pylint: disable=arguments-differ
        self.env.reset()
        return self.state

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        info.update({'observation': observation})

        return self.state, reward, done, info
