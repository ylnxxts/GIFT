"""AKS baseline: adaptive keyframe sampling via recursive temporal splitting.

Reference: Adaptive Keyframe Sampling for Long Video Understanding (CVPR 2025).
"""

import torch

from gift.base import BaseFrameSelector


class AKSFrameSelector(BaseFrameSelector):
    #: A bin whose top frames stand out by more than this margin is left intact.
    relevance_threshold = 0.2
    std_threshold = -100
    max_depth = 3

    def select_frames(self, image_pool_embeds, text_embeds, image_embeds):
        frame_embeds_normalized = image_pool_embeds / image_pool_embeds.norm(dim=-1, keepdim=True)
        frame_embeds_normalized = frame_embeds_normalized.float()
        text_embeds_normalized = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        text_embeds_normalized = text_embeds_normalized.float()

        scores = torch.matmul(
            frame_embeds_normalized, text_embeds_normalized.transpose(0, 1)
        ).mean(-1)
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

        original_indices = torch.arange(len(scores), device=scores.device)
        leaf_bins = self._adaptive_split(normalized_scores, original_indices, depth=0)

        selected_indices = []
        for scores_bin, indices_bin, depth in leaf_bins:
            num_frames_to_sample = max(1, int(self.archor_frame_num / (2**depth)))
            num_frames_to_sample = min(num_frames_to_sample, len(scores_bin))
            if len(scores_bin) > 0:
                _, topk_indices_in_bin = torch.topk(scores_bin, k=num_frames_to_sample)
                selected_indices.append(indices_bin[topk_indices_in_bin])

        if not selected_indices:
            return torch.linspace(
                0, len(scores) - 1, self.archor_frame_num, device=scores.device
            ).long()

        final_selected_indices = torch.cat(selected_indices)

        # Per-bin budgets are rounded independently, so the union can overshoot.
        if len(final_selected_indices) > self.archor_frame_num:
            final_scores = scores[final_selected_indices]
            _, topk_indices = torch.topk(final_scores, self.archor_frame_num)
            final_selected_indices = final_selected_indices[topk_indices]

        return torch.sort(final_selected_indices).values

    def _adaptive_split(self, scores_bin: torch.Tensor, indices_bin: torch.Tensor, depth: int):
        """Recursively halve a temporal bin until its relevance is concentrated."""
        if depth >= self.max_depth or len(scores_bin) < 2:
            return [(scores_bin, indices_bin, depth)]

        mean_score = scores_bin.mean()
        std_dev = scores_bin.std()

        k = min(self.archor_frame_num, len(scores_bin))
        top_k_scores, _ = torch.topk(scores_bin, k=k)
        relevance_diff = top_k_scores.mean() - mean_score

        # Relevance is already peaked here: stop and treat the bin as a leaf.
        if relevance_diff > self.relevance_threshold and std_dev > self.std_threshold:
            return [(scores_bin, indices_bin, depth)]

        midpoint = len(scores_bin) // 2
        left_scores, right_scores = torch.split(
            scores_bin, [midpoint, len(scores_bin) - midpoint]
        )
        left_indices, right_indices = torch.split(
            indices_bin, [midpoint, len(indices_bin) - midpoint]
        )
        return self._adaptive_split(left_scores, left_indices, depth + 1) + self._adaptive_split(
            right_scores, right_indices, depth + 1
        )
