from typing import Literal, Union

import numpy as np

from gym_pomdps.rendering.canvas import Canvas
from gym_pomdps.rendering.colors import SimplePalette

palette = SimplePalette(
    wall="black",
    floor="gray",
    heaven="green",
    hell="red",
    agent="blue",
)


RenderMode = Union[Literal["human"], Literal["rgb_array"]]


def render_heavenhell1(state: int) -> np.ndarray:
    canvas = Canvas((3, 3), background=palette.wall)
    canvas[0, :] = palette.floor
    canvas[:, 1] = palette.floor
    canvas[-1, 1:] = palette.floor

    if state in [0, 6]:
        canvas[1, 1] = palette.agent
    elif state in [1, 7]:
        canvas[0, 1] = palette.agent
    elif state in [2, 8]:
        canvas[0, 0] = palette.agent
    elif state in [3, 9]:
        canvas[0, 2] = palette.agent
    elif state in [4, 10]:
        canvas[2, 1] = palette.agent
    elif state in [5, 11]:
        canvas[2, 2] = palette.agent

    if state == 5:
        canvas[0, 0] = palette.heaven
        canvas[0, -1] = palette.hell
    elif state == 11:
        canvas[0, 0] = palette.hell
        canvas[0, -1] = palette.heaven

    return canvas.image()


def render_heavenhell2(state: int) -> np.ndarray:
    canvas = Canvas((4, 5), background=palette.wall)
    canvas[0, :] = palette.floor
    canvas[:, 2] = palette.floor
    canvas[-1, 2:] = palette.floor

    if state in [0, 10]:
        canvas[2, 2] = palette.agent
    elif state in [1, 11]:
        canvas[1, 2] = palette.agent
    elif state in [2, 12]:
        canvas[0, 2] = palette.agent
    elif state in [3, 13]:
        canvas[0, 1] = palette.agent
    elif state in [4, 14]:
        canvas[0, 0] = palette.agent
    elif state in [5, 15]:
        canvas[0, 3] = palette.agent
    elif state in [6, 16]:
        canvas[0, 4] = palette.agent
    elif state in [7, 17]:
        canvas[3, 2] = palette.agent
    elif state in [8, 18]:
        canvas[3, 3] = palette.agent
    elif state in [9, 19]:
        canvas[3, 4] = palette.agent

    if state == 9:
        canvas[0, 0] = palette.heaven
        canvas[0, -1] = palette.hell
    elif state == 19:
        canvas[0, 0] = palette.hell
        canvas[0, -1] = palette.heaven

    return canvas.image()


def render_heavenhell3(state: int) -> np.ndarray:
    canvas = Canvas((5, 7), background=palette.wall)
    canvas[0, :] = palette.floor
    canvas[:, 3] = palette.floor
    canvas[-1, 3:] = palette.floor

    if state in [0, 14]:
        canvas[3, 3] = palette.agent
    elif state in [1, 15]:
        canvas[2, 3] = palette.agent
    elif state in [2, 16]:
        canvas[1, 3] = palette.agent
    elif state in [3, 17]:
        canvas[0, 3] = palette.agent
    elif state in [4, 18]:
        canvas[0, 2] = palette.agent
    elif state in [5, 19]:
        canvas[0, 1] = palette.agent
    elif state in [6, 20]:
        canvas[0, 0] = palette.agent
    elif state in [7, 21]:
        canvas[0, 4] = palette.agent
    elif state in [8, 22]:
        canvas[0, 5] = palette.agent
    elif state in [9, 23]:
        canvas[0, 6] = palette.agent
    elif state in [10, 24]:
        canvas[4, 3] = palette.agent
    elif state in [11, 25]:
        canvas[4, 4] = palette.agent
    elif state in [12, 26]:
        canvas[4, 5] = palette.agent
    elif state in [13, 27]:
        canvas[4, 6] = palette.agent

    if state == 13:
        canvas[0, 0] = palette.heaven
        canvas[0, -1] = palette.hell
    elif state == 27:
        canvas[0, 0] = palette.hell
        canvas[0, -1] = palette.heaven

    return canvas.image()
