"""Smoke tests for GIFT's scoring. Requires only torch:

    python -m pytest tests/ -q
"""

import torch

from gift.irreplaceability import select_keyframes, select_most_irreplaceable
from gift.registry import build_frame_selector


def _inputs(num_frames=128, dim=512, seed=0):
    torch.manual_seed(seed)
    return torch.randn(num_frames, dim), torch.randn(1, dim)


def test_returns_requested_budget():
    frames, text = _inputs()
    for budget in (4, 8, 16, 32, 64):
        assert len(select_keyframes(frames, text, budget)) == budget


def test_indices_are_unique_and_in_range():
    frames, text = _inputs(num_frames=97)
    keep = select_keyframes(frames, text, 32)
    assert len(set(keep.tolist())) == len(keep)
    assert keep.min() >= 0 and keep.max() < 97


def test_budget_larger_than_video_returns_every_frame():
    frames, text = _inputs(num_frames=10)
    assert sorted(select_keyframes(frames, text, 32).tolist()) == list(range(10))


def test_most_relevant_frame_is_never_suppressed():
    """It has an empty substitute set, so it attains maximum directed diversity."""
    torch.manual_seed(0)
    frames = torch.nn.functional.normalize(torch.randn(64, 128), dim=1)
    text = torch.nn.functional.normalize(torch.randn(1, 128), dim=1)
    most_relevant = (frames @ text.T).mean(-1).argmax().item()
    assert most_relevant in select_most_irreplaceable(frames, text, 8).tolist()


def test_duplicate_frames_are_not_both_selected():
    """A frame with an identical but more relevant twin is replaceable."""
    torch.manual_seed(0)
    base = torch.nn.functional.normalize(torch.randn(16, 64), dim=1)
    frames = torch.cat([base, base]).clone()  # every frame has an exact twin
    text = torch.nn.functional.normalize(torch.randn(1, 64), dim=1)
    keep = sorted(select_most_irreplaceable(frames, text, 8).tolist())
    assert not any(i + 16 in keep for i in keep if i < 16)


def test_refinement_group_size_changes_selection_not_count():
    frames, text = _inputs()
    for group in (1, 6, 9, 32):
        assert len(select_keyframes(frames, text, 32, refine_group_size=group)) == 32


def test_uniform_selector_spreads_budget_over_pool():
    selector = build_frame_selector("uniform", 4)
    assert selector(torch.zeros(128, 8), "query").tolist() == [0, 42, 85, 127]
    assert selector(torch.zeros(3, 8), "query").tolist() == [0, 1, 2]
