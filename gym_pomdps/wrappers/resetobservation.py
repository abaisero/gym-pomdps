import gym
import gym.spaces

from gym_pomdps.envs.pomdp import POMDP


class ResetObservationWrapper(gym.ObservationWrapper):
    """Reset Observation Wrapper.

    Takes a POMDP environment and extends the observation space by introducing
    a novel observation to be emitted after every reset.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        if not isinstance(env.unwrapped, POMDP):
            raise TypeError('Unwrapped input environment must be POMDP')

        if not isinstance(env.observation_space, gym.spaces.Discrete):
            raise ValueError('Observation space must be Discrete')

        self.observation_space = gym.spaces.Discrete(env.observation_space.n + 1)

    def observation(self, observation):
        return observation if observation is not None else self.observation_space.n - 1
