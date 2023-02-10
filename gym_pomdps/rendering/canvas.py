from typing import Any, Optional, Tuple

import numpy as np

from gym_pomdps.rendering import colors

Shape = Tuple[int, int]


def upscale(image: np.ndarray, shape: Shape) -> np.ndarray:
    if shape[0] != 1:
        image = np.repeat(image, shape[0], axis=0)

    if shape[1] != 1:
        image = np.repeat(image, shape[1], axis=1)

    return image


class Canvas:
    def __init__(self, shape: Shape, *, background: Optional[Any] = None):
        if background is not None:
            background = colors.to_rgb(background)

        super().__init__()

        shape_3d = *shape, 3
        self.__pixels = (
            np.zeros(shape_3d) if background is None else np.full(shape_3d, background)
        )

    def __getitem__(self, key):
        return self.__pixels[key]

    def __setitem__(self, key, value):
        self.__pixels[key] = colors.to_rgb(value)

    def image(self, shape: Optional[Shape] = None):
        if shape is None or shape == (1, 1):
            return self.__pixels.copy()

        return upscale(self.__pixels.copy(), shape)
