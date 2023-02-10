from typing import Callable, Dict, Optional

import numpy as np

from .heavenhell import render_heavenhell1, render_heavenhell2, render_heavenhell3

Image = np.ndarray
RenderFunction = Callable[[int], Image]


_render_functions: Dict[str, RenderFunction] = {
    'heavenhell_1': render_heavenhell1,
    'heavenhell_2': render_heavenhell2,
    'heavenhell_3': render_heavenhell3,
}


def get_render(name: str) -> Optional[RenderFunction]:
    return _render_functions.get(name)
