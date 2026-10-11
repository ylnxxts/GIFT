"""GIFT keyframe selector."""

import torch

from .base import BaseFrameSelector
from .irreplaceability import select_keyframes


class GIFTFrameSelector(BaseFrameSelector):
    """Training-free keyframe selection by Global Irreplaceability Frame Targeting."""

    def select_frames(self, image_pool_embeds, text_embeds, image_embeds):
        select_idx = select_keyframes(image_pool_embeds, text_embeds, self.archor_frame_num)
        # Restore temporal order; selection order carries no meaning downstream.
        return torch.tensor(sorted(select_idx.tolist()), device=image_embeds.device)
