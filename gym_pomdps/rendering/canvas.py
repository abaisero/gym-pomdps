from typing import Any, Optional, Tuple

import numpy as np
from typing_extensions import TypeAlias

from gym_pomdps.rendering import colors

Shape: TypeAlias = Tuple[int, int]
Image: TypeAlias = np.ndarray


def upscale(image: Image, shape: Shape) -> Image:
    if shape[0] != 1:
        image = np.repeat(image, shape[0], axis=0)

    if shape[1] != 1:
        image = np.repeat(image, shape[1], axis=1)

    return image


class Canvas:
    def __init__(self, shape: Shape, *, background: Optional[Any] = None):
        super().__init__()

        if background is not None:
            background = colors.to_rgb(background)

        shape_3d = *shape, 3
        self.__image = (
            np.zeros(shape_3d) if background is None else np.full(shape_3d, background)
        )

    def __getitem__(self, key):
        return self.__image[key]

    def __setitem__(self, key, value):
        self.__image[key] = colors.to_rgb(value)

    def image(self, shape: Optional[Shape] = None) -> Image:
        image = self.__image.copy()

        if shape is not None and shape != (1, 1):
            image = upscale(image, shape)

        return image
