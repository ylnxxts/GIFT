"""SigLIP scorer shared by GIFT and the baseline frame selectors.

Encodes video frames and the (cleaned) user query into a common embedding
space, which every query-aware frame selector in this repo builds on.
"""

import copy
import os

import torch
from transformers import SiglipModel, SiglipProcessor

DEFAULT_SIGLIP_PATH = "google/siglip-so400m-patch14-384"

# Benchmark-specific boilerplate that carries no visual grounding signal; keeping
# it would dominate the text embedding and wash out the actual question.
PROMPT_HEADS = [
    "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.",
    "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.",
]
PROMPT_TAILS = [
    "Answer with the option's letter from the given choices directly.",
    "The best answer is:",
    "Answer the question using a single word or phrase.",
    "Only give the best option.\n",
    "Best option: (",
]


def resolve_siglip_path() -> str:
    """Path or HF hub id of the SigLIP checkpoint, overridable via env var."""
    return os.environ.get("GIFT_SIGLIP_PATH", DEFAULT_SIGLIP_PATH)


class SigLipScorer:
    def __init__(self, device="cuda", model_path=None):
        self.device = device
        path = model_path or resolve_siglip_path()
        self.model = SiglipModel.from_pretrained(path, device_map=self.device)
        self.processor = SiglipProcessor.from_pretrained(path)
        self.max_position_embeddings = self.model.text_model.config.max_position_embeddings

    def __call__(self, images, texts):
        texts = self.clear_prompt(copy.deepcopy(texts))

        inputs = self.processor(
            text=texts, images=images, padding="max_length", return_tensors="pt"
        ).to(self.model.device)

        # SigLIP's text tower has a fixed context window. Long prompts are split
        # into equal chunks taken alternately from the head and the tail so that
        # both the question and the answer options stay represented.
        stride_num = (int(inputs["input_ids"].shape[-1]) + 63) // self.max_position_embeddings
        stride = (inputs["input_ids"].shape[-1] + stride_num - 1) // stride_num

        input_id_heads, input_id_tails = [], []
        l, r = 0, inputs["input_ids"].shape[-1]
        while l < r:
            input_id_heads.append(inputs["input_ids"][:, l : l + stride])
            l += stride
            if l < r:
                input_id_tails.append(inputs["input_ids"][:, r - stride : r])
                r -= stride

        input_ids = torch.cat(input_id_heads + input_id_tails[::-1])

        with torch.no_grad():
            outputs = self.model(input_ids, pixel_values=inputs["pixel_values"])

        image_pool_embeds = outputs.image_embeds
        image_embeds = outputs.vision_model_output["last_hidden_state"]
        text_embeds = outputs.text_embeds
        return image_pool_embeds, text_embeds.mean(dim=0, keepdim=True), image_embeds

    @staticmethod
    def clear_prompt(prompt):
        for head in PROMPT_HEADS:
            prompt = prompt.split(head)[-1]
        for tail in PROMPT_TAILS:
            prompt = prompt.split(tail)[0]
        return prompt.strip()
