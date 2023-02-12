from typing import Dict, Optional

from gym_pomdps.rendering.heavenhell import (render_heavenhell1,
                                             render_heavenhell2,
                                             render_heavenhell3)
from gym_pomdps.rendering.renderer import RenderFunction

_render_functions: Dict[str, RenderFunction] = {
    'heavenhell_1': render_heavenhell1,
    'heavenhell_2': render_heavenhell2,
    'heavenhell_3': render_heavenhell3,
}


def get_render_function(name: str) -> Optional[RenderFunction]:
    return _render_functions.get(name)
