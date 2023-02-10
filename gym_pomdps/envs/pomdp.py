from typing import Callable, Final, Optional, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from gym.utils import seeding
from rl_parsers.pomdp import parse
from typing_extensions import TypeAlias

__all__ = ["POMDP"]

State: TypeAlias = int
Action: TypeAlias = int
Observation: TypeAlias = int

NoState: Final = -1
NoAction: Final = -1
NoObservation: Final = -1


class POMDP(gym.Env):  # pylint: disable=abstract-method
    """Environment specified by POMDP file."""

    def __init__(
        self,
        text,
        *,
        episodic: bool,
        render: Optional[Callable] = None,
        seed=None,
    ):
        model = parse(text)
        self.episodic = episodic
        self._render = render
        self.seed(seed)

        if model.values == "cost":
            raise ValueError("Unsupported `cost` values.")

        self.model = model
        self.discount = model.discount
        self.state_space = spaces.Discrete(len(model.states))
        self.action_space = spaces.Discrete(len(model.actions))
        self.observation_space = spaces.Discrete(len(model.observations))
        self.reward_range = model.R.min(), model.R.max()

        self.rewards_dict = {r: i for i, r in enumerate(np.unique(model.R))}

        if model.start is None:
            self.start = np.full(self.state_space.n, 1 / self.state_space.n)
        else:
            self.start = model.start
        self.T = model.T.transpose(1, 0, 2).copy()
        if model.flags["O_includes_state"]:
            self.O = model.O.transpose(1, 0, 2, 3).copy()
        else:
            self.O = np.expand_dims(model.O, axis=0).repeat(self.state_space.n, axis=0)
        self.R = model.R.transpose(1, 0, 2, 3).copy()

        if episodic:
            self.D = model.reset.T.copy()  # only if episodic

        # NOTE currently elsewhere
        # self.TO = np.expand_dims(self.T, axis=-1) * self.O
        # self.EO = self.TO.sum(-2)
        # self.ER = (self.TO * self.R).sum((-2, -1))

        self.state: State = NoState
        self.observation: Observation = NoObservation

    def seed(self, seed):  # pylint: disable=signature-differs
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def reset(self) -> Observation:
        self.state, self.observation = self.reset_functional()
        return self.observation

    def step(self, action: Action) -> Tuple[Observation, float, bool, Optional[dict]]:
        ret = self.step_functional(self.state, action)
        self.state, self.observation, reward, done, info = ret
        return self.observation, reward, done, info

    def reset_functional(self) -> Tuple[State, Observation]:
        state = self.np_random.multinomial(1, self.start).argmax().item()
        observation = NoObservation
        return (state, observation)

    def step_functional(
        self, state: State, action: Action
    ) -> Tuple[State, Observation, float, bool, Optional[dict]]:
        if (state == NoState) != (action == NoAction):
            raise ValueError(f"Invalid state-action pair ({state}, {action}).")

        if state == NoState and action == NoAction:
            return NoState, NoObservation, 0.0, True, None

        assert 0 <= state < self.state_space.n
        assert 0 <= action < self.action_space.n

        next_state = (
            self.np_random.multinomial(1, self.T[state, action]).argmax().item()
        )
        observation = (
            self.np_random.multinomial(1, self.O[state, action, next_state])
            .argmax()
            .item()
        )
        # NOTE below is the same but unified in single sampling op; requires TO
        # next_state, observation = divmod(
        #     self.np_random.multinomial(
        #         1, self.TO[state, action].ravel()).argmax().item(),
        #     self.observation.n,
        # )

        reward = self.R[state, action, next_state, observation].item()

        done = self.D[state, action].item() if self.episodic else False
        if done:
            next_state = NoState

        reward_cat = self.rewards_dict[reward]
        info = dict(reward_cat=reward_cat)

        return next_state, observation, reward, done, info

    def render(self, mode="human"):
        if self._render is None:
            return super().render(mode)

        image = self._render(self.observation)

        if mode == "human":
            plt.imshow(image)
            plt.show(block=False)

        return image
