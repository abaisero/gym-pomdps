from types import SimpleNamespace
from typing import Any, Tuple

import matplotlib.colors

RGBColor = Tuple[float, float, float]


def to_rgb(color: Any) -> RGBColor:
    return matplotlib.colors.to_rgb(color)


class SimplePalette(SimpleNamespace):
    def __init__(self, **kwargs):
        kwargs = {key: to_rgb(value) for key, value in kwargs.items()}
        super().__init__(**kwargs)
