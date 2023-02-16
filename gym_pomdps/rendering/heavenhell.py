from gym_pomdps.rendering.canvas import Canvas, Image
from gym_pomdps.rendering.colors import SimplePalette
from gym_pomdps.types import Observation

palette = SimplePalette(
    wall="black",
    floor="gray",
    heaven="green",
    hell="red",
    agent="blue",
)


def render_heavenhell1(observation: Observation) -> Image:
    canvas = Canvas((3, 3), background=palette.wall)
    canvas[0, :] = palette.floor
    canvas[:, 1] = palette.floor
    canvas[-1, 1:] = palette.floor

    if observation == 0 or observation == 7:
        canvas[1, 1] = palette.agent
    elif observation == 1:
        canvas[0, 1] = palette.agent
    elif observation == 2:
        canvas[0, 0] = palette.agent
    elif observation == 3:
        canvas[0, 2] = palette.agent
    elif observation == 4:
        canvas[2, 1] = palette.agent
    elif observation == 5:
        canvas[2, 2] = palette.agent
        canvas[0, 0] = palette.heaven
        canvas[0, -1] = palette.hell
    elif observation == 6:
        canvas[2, 2] = palette.agent
        canvas[0, 0] = palette.hell
        canvas[0, -1] = palette.heaven

    return canvas.image()


def render_heavenhell2(observation: Observation) -> Image:
    canvas = Canvas((4, 5), background=palette.wall)
    canvas[0, :] = palette.floor
    canvas[:, 2] = palette.floor
    canvas[-1, 2:] = palette.floor

    if observation == 0 or observation == 11:
        canvas[2, 2] = palette.agent
    elif observation == 1:
        canvas[1, 2] = palette.agent
    elif observation == 2:
        canvas[0, 2] = palette.agent
    elif observation == 3:
        canvas[0, 1] = palette.agent
    elif observation == 4:
        canvas[0, 0] = palette.agent
    elif observation == 5:
        canvas[0, 3] = palette.agent
    elif observation == 6:
        canvas[0, 4] = palette.agent
    elif observation == 7:
        canvas[3, 2] = palette.agent
    elif observation == 8:
        canvas[3, 3] = palette.agent
    elif observation == 9:
        canvas[3, 4] = palette.agent
        canvas[0, 0] = palette.heaven
        canvas[0, -1] = palette.hell
    elif observation == 10:
        canvas[3, 4] = palette.agent
        canvas[0, 0] = palette.hell
        canvas[0, -1] = palette.heaven

    return canvas.image()


def render_heavenhell3(observation: Observation) -> Image:
    canvas = Canvas((5, 7), background=palette.wall)
    canvas[0, :] = palette.floor
    canvas[:, 3] = palette.floor
    canvas[-1, 3:] = palette.floor

    if observation == 0 or observation == 15:
        canvas[3, 3] = palette.agent
    elif observation == 1:
        canvas[2, 3] = palette.agent
    elif observation == 2:
        canvas[1, 3] = palette.agent
    elif observation == 3:
        canvas[0, 3] = palette.agent
    elif observation == 4:
        canvas[0, 2] = palette.agent
    elif observation == 5:
        canvas[0, 1] = palette.agent
    elif observation == 6:
        canvas[0, 0] = palette.agent
    elif observation == 7:
        canvas[0, 4] = palette.agent
    elif observation == 8:
        canvas[0, 5] = palette.agent
    elif observation == 9:
        canvas[0, 6] = palette.agent
    elif observation == 10:
        canvas[4, 3] = palette.agent
    elif observation == 11:
        canvas[4, 4] = palette.agent
    elif observation == 12:
        canvas[4, 5] = palette.agent
    elif observation == 13:
        canvas[4, 6] = palette.agent
        canvas[0, 0] = palette.heaven
        canvas[0, -1] = palette.hell
    elif observation == 14:
        canvas[4, 6] = palette.agent
        canvas[0, 0] = palette.hell
        canvas[0, -1] = palette.heaven

    return canvas.image()
