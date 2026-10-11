"""Core of GIFT: irreplaceability scoring and budget-aware refinement.

A frame is *irreplaceable* when no superior substitute exists for it, i.e. when
it is highly relevant to the query and, at the same time, visually remote from
every frame that is *more* relevant than itself. Restricting the diversity term
to that higher-relevance "substitute set" is what the paper calls Directed
Diversity, and it couples relevance and diversity into a single score instead
of trading them off with a hand-tuned weight.
"""

import torch
import torch.nn.functional as F


def select_most_irreplaceable(frame_embeds, text_embed, selection_num):
    """Pick `selection_num` frames with the highest irreplaceability score.

    `frame_embeds` and `text_embed` are expected to be L2-normalised, so inner
    products are cosine similarities.
    """
    if frame_embeds.shape[0] <= selection_num:
        return torch.arange(frame_embeds.shape[0], device=frame_embeds.device)

    # Query relevance. The tiny noise term breaks ties so that the strict
    # inequality below cannot leave a group of equal-relevance frames masking
    # each other out.
    relevance = torch.matmul(frame_embeds, text_embed.transpose(0, 1)).mean(-1)
    relevance = relevance + torch.rand(
        relevance.shape, device=relevance.device, dtype=relevance.dtype
    ) * 1e-6
    relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min() + 1e-6)

    pairwise_distance = 1 - torch.matmul(frame_embeds, frame_embeds.transpose(0, 1))
    pairwise_distance = torch.clamp(pairwise_distance, min=1e-6)

    # Directed Diversity: distance to the nearest *more relevant* frame, i.e. to
    # the closest potential substitute. Frames outside the substitute set are
    # masked to the maximum distance so they cannot win the min(). The single
    # most relevant frame has an empty substitute set and therefore attains the
    # maximum, which is the intended behaviour: nothing can replace it.
    substitutes = (relevance[None, :] > relevance[:, None]).float()
    dist_max = pairwise_distance.max()
    directed_diversity, _ = (
        pairwise_distance * substitutes + dist_max * (1 - substitutes)
    ).min(dim=-1)

    irreplaceability = relevance * directed_diversity
    _, best_local_indices = irreplaceability.topk(selection_num, dim=-1)

    return best_local_indices


def select_keyframes(
    frame_embeds: torch.Tensor,
    text_embed: torch.Tensor,
    selection_num: int,
    refine_group_size: int = 9,
):
    """Budget-aware refinement: select keyframes in successive rounds.

    Scoring all frames once would keep only globally irreplaceable frames and
    suppress their neighbours, which costs temporal coherence. Instead we take
    `refine_group_size` frames per round and drop them from the candidate pool.
    Removing a selected frame also removes the suppression it exerted, so on the
    next round its previously shadowed neighbours can surface — the budget first
    buys a core set of key events, then the context around them.

    Returns a LongTensor of frame indices in selection order.
    """
    num_frames, _ = frame_embeds.shape

    frame_embeds_normalized = F.normalize(frame_embeds.float(), p=2, dim=1)
    text_embed_normalized = F.normalize(text_embed.float(), p=2, dim=1)

    available_indices = torch.arange(num_frames, device=frame_embeds.device)
    selected_indices = []

    while len(selected_indices) < selection_num:
        current_selection_num = min(refine_group_size, selection_num - len(selected_indices))
        if len(available_indices) == 0:
            break

        current_embeds = frame_embeds_normalized[available_indices]
        best_local_indices = select_most_irreplaceable(
            current_embeds, text_embed_normalized, current_selection_num
        )
        selected_indices.extend(available_indices[best_local_indices].tolist())

        keep = torch.ones(available_indices.shape[0], dtype=torch.bool, device=frame_embeds.device)
        keep[best_local_indices] = False
        available_indices = available_indices[keep]

    return torch.tensor(selected_indices, device=frame_embeds.device)
