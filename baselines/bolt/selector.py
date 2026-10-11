"""BOLT baseline: inverse-transform sampling over query-relevance scores.

Reference: BOLT: Boost Large Vision-Language Model Without Training for
Long-form Video Understanding (CVPR 2025).
"""

import torch

from gift.base import BaseFrameSelector


class BOLTFrameSelector(BaseFrameSelector):
    #: Exponent that sharpens the relevance distribution before integrating it.
    sharpening_exponent = 2.5

    def select_frames(self, image_pool_embeds, text_embeds, image_embeds):
        num_frames, _ = image_pool_embeds.shape

        frame_embeds_normalized = image_pool_embeds / image_pool_embeds.norm(dim=-1, keepdim=True)
        frame_embeds_normalized = frame_embeds_normalized.float()
        text_embeds_normalized = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        text_embeds_normalized = text_embeds_normalized.float()

        scores = torch.matmul(
            frame_embeds_normalized, text_embeds_normalized.transpose(0, 1)
        ).mean(-1)
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
        sharpened_scores = torch.pow(normalized_scores, self.sharpening_exponent)

        cdf = torch.cumsum(sharpened_scores, dim=0) / sharpened_scores.sum()

        # Inverse transform sampling: evenly spaced probabilities are mapped back
        # through the CDF, so frames are drawn densely where relevance is high.
        target_probs = torch.linspace(
            1.0 / self.archor_frame_num,
            1.0,
            self.archor_frame_num,
            device=image_pool_embeds.device,
        )
        archor_frame_idx = torch.searchsorted(cdf, target_probs)
        archor_frame_idx = torch.clamp(archor_frame_idx, max=num_frames - 1)

        # Clamping can collapse several probabilities onto the same frame.
        return torch.unique(archor_frame_idx.sort().values)
