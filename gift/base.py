"""Common scaffolding for query-aware keyframe selectors.

Subclasses only implement `select_frames`; encoding the video in fixed-size
chunks (so that long videos fit in memory) is handled here.
"""

import copy

import torch

from .siglip import SigLipScorer


class BaseFrameSelector:
    #: Frames encoded per forward pass, bounded by SigLIP activation memory.
    chunk_size = 128

    def __init__(self, select_frame_num=64, device="cuda"):
        self.archor_frame_num = select_frame_num
        self.vlm = SigLipScorer(device)
        print(f"{type(self).__name__}, select_num: {self.archor_frame_num}")

    def __call__(self, frames, prompt, sample=None):
        input_frames = copy.deepcopy(frames)
        num_frames = input_frames.shape[0]
        num_chunks = num_frames // self.chunk_size

        all_image_pool_embeds = []
        all_image_embeds = []
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            chunk_video = input_frames[start_idx : start_idx + self.chunk_size]
            pool_embeds, text_embeds, embeds = self.vlm(chunk_video, prompt)
            all_image_pool_embeds.append(pool_embeds)
            all_image_embeds.append(embeds)

        remaining_frames = num_frames % self.chunk_size
        if remaining_frames > 0:
            start_idx = num_chunks * self.chunk_size
            remaining_video = input_frames[start_idx : start_idx + remaining_frames]
            pool_embeds, text_embeds, embeds = self.vlm(remaining_video, prompt)
            all_image_pool_embeds.append(pool_embeds)
            all_image_embeds.append(embeds)

        final_image_pool_embeds = torch.cat(all_image_pool_embeds, dim=0)
        final_image_embeds = torch.cat(all_image_embeds, dim=0)

        with torch.no_grad():
            return self.select_frames(final_image_pool_embeds, text_embeds, final_image_embeds)

    def select_frames(self, image_pool_embeds, text_embeds, image_embeds):
        """Return a sorted LongTensor of selected frame indices."""
        raise NotImplementedError
