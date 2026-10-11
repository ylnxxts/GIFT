"""Factory used by the lmms-eval model wrappers to build a frame selector."""

import torch

SUPPORTED_SELECTORS = ("GIFT", "AKS", "BOLT", "uniform")


class UniformFrameSelector:
    """Evenly spaced frames from the candidate pool; needs no scorer.

    With a pool no larger than the budget every frame is kept, which is plain
    uniform sampling of the video.
    """

    def __init__(self, select_frame_num):
        self.archor_frame_num = select_frame_num

    def __call__(self, frames, prompt, sample=None):
        num_frames = frames.shape[0]
        if num_frames <= self.archor_frame_num:
            return torch.arange(num_frames)
        return torch.linspace(0, num_frames - 1, self.archor_frame_num).round().long()


def build_frame_selector(name, select_frame_num, device="cuda"):
    if name == "GIFT":
        from .selector import GIFTFrameSelector

        return GIFTFrameSelector(select_frame_num=select_frame_num, device=device)
    if name == "AKS":
        from baselines.aks.selector import AKSFrameSelector

        return AKSFrameSelector(select_frame_num=select_frame_num, device=device)
    if name == "BOLT":
        from baselines.bolt.selector import BOLTFrameSelector

        return BOLTFrameSelector(select_frame_num=select_frame_num, device=device)
    if name == "uniform":
        return UniformFrameSelector(select_frame_num)
    raise ValueError(f"Unknown frame selector {name!r}; expected one of {SUPPORTED_SELECTORS}")
