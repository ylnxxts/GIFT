"""GIFT: Global Irreplaceability Frame Targeting.

The scoring functions depend only on torch, so they can be imported and reused
on their own. The selectors additionally pull in SigLIP via transformers, so
they are resolved lazily.
"""

from .irreplaceability import select_keyframes, select_most_irreplaceable
from .registry import build_frame_selector

__all__ = [
    "BaseFrameSelector",
    "GIFTFrameSelector",
    "SigLipScorer",
    "build_frame_selector",
    "select_keyframes",
    "select_most_irreplaceable",
]

_LAZY = {
    "BaseFrameSelector": ("base", "BaseFrameSelector"),
    "GIFTFrameSelector": ("selector", "GIFTFrameSelector"),
    "SigLipScorer": ("siglip", "SigLipScorer"),
}


def __getattr__(name):
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        module = __import__(f"{__name__}.{module_name}", fromlist=[attr])
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
