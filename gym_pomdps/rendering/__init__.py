from typing import Callable, Optional

import numpy as np

from .heavenhell import (render_heavenhell1, render_heavenhell2,
                         render_heavenhell3)

Image = np.ndarray
RenderFunction = Callable[[int], Image]


def get_render(name: str) -> Optional[RenderFunction]:
    if name == 'heavenhell_1':
        return render_heavenhell1
    if name == 'heavenhell_2':
        return render_heavenhell2
    if name == 'heavenhell_3':
        return render_heavenhell3

    return None
