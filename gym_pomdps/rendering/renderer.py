import abc
from typing import Callable, Optional

import matplotlib.pyplot as plt
from typing_extensions import TypeAlias

from gym_pomdps.rendering.canvas import Image
from gym_pomdps.types import Observation

RenderFunction: TypeAlias = Callable[[Observation], Image]


class Renderer(metaclass=abc.ABCMeta):
    def __init__(self, render_function: RenderFunction):
        super().__init__()
        self.render_function = render_function

    def render(self, observation: Observation) -> Image:
        return self.render_function(observation)

    @abc.abstractmethod
    def show(self, observation: Observation) -> Image:
        assert False


class PltRenderer(Renderer):
    def __init__(self, render_function: RenderFunction):
        super().__init__(render_function)
        self.fig = Optional[plt.Figure]

    def render(self, observation: Observation) -> Image:
        return self.render_function(observation)

    def show(self, observation: Observation) -> Image:
        image = self.render_function(observation)

        if self.fig is None:
            self.fig = plt.figure()

        plt.imshow(image)
        plt.show(block=False)

        return image
