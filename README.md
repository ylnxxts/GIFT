# GIFT: Global Irreplaceability Frame Targeting for Efficient Video Understanding

Official implementation of **GIFT** (CVPR 2026).

[[Paper](https://arxiv.org/abs/2603.25072)] · [中文说明](README_zh.md)

GIFT is a **training-free** keyframe selector for Video LLMs. Instead of greedily
asking *"what is the next best frame to add?"*, it scores every frame by its
**irreplaceability**: a frame matters when it is relevant to the query *and* no
superior substitute for it exists elsewhere in the video.

Because it only changes which frames reach the model, GIFT drops into any Video
LLM without retraining and without touching model weights.

## Method

**Directed Diversity.** For a frame `Fi`, its *substitute set* is every frame
that is *more* query-relevant than itself. Directed Diversity is the distance
from `Fi` to the nearest member of that set. Multiplying it by relevance yields
a single irreplaceability score, so relevance and diversity no longer need a
hand-tuned trade-off weight — and the most relevant frame, having an empty
substitute set, is never suppressed.

**Budget-Aware Refinement.** Scoring once would keep only globally irreplaceable
frames and suppress their temporal neighbours, costing coherence. GIFT instead
selects in rounds of `B` frames, removing each round from the candidate pool.
Removing a frame also removes the suppression it exerted, so previously shadowed
neighbours resurface. Small budgets buy the key events; larger budgets
progressively buy the context around them.

Both steps live in [`gift/irreplaceability.py`](gift/irreplaceability.py) and
together are about 90 lines of PyTorch.

## Results

LLaVA-Video-7B, average score across MVBench / LongVideoBench / MLVU / VideoMME,
as a percentage of the 64-frame upper bound (Table 1 of the paper):

| Frame budget | Uniform | BOLT (CVPR'25) | AKS (CVPR'25) | **GIFT** |
|---|---|---|---|---|
| 32 | 96.5 | 100.0 | 98.9 | **100.6** |
| 16 | 93.4 | 97.8 | 97.0 | **99.2** |
| 8  | 90.6 | 93.9 | 94.9 | **97.3** |
| 4  | 85.6 | 89.6 | 91.7 | **93.9** |

At a 4-frame budget GIFT recovers 93.9% of full-frame accuracy, an 8.3-point
gain over uniform sampling. Gains grow as the budget tightens.

## Repository layout

```
gift/                  Core method
  irreplaceability.py    Directed Diversity + Budget-Aware Refinement
  selector.py            GIFTFrameSelector
  siglip.py              SigLIP frame/query scorer
  base.py                Chunked encoding shared by all selectors
  registry.py            build_frame_selector(name, budget, device)
baselines/             AKS and BOLT reference selectors
llava/                 LLaVA-NeXT (vendored)
lmms-eval/             lmms-eval harness with the GIFT hooks
scripts/eval/          Ready-to-run evaluation scripts
tests/                 Torch-only smoke tests
```

Every edit to the vendored code is bracketed by `GIFT Start` / `GIFT End`
comments, so `grep -rn "GIFT Start"` shows the complete integration surface: the
four model wrappers above (about ten lines each), plus two unrelated one-line
fixes to a dataset path and the VILA loader signature.

## Installation

```bash
conda create -n gift python=3.10 -y
conda activate gift
pip install -r requirements.txt
pip install flash-attn==2.5.7 --no-build-isolation   # match your CUDA/torch build
pip install -e .
```

The scoring code needs nothing but torch, so you can check it works before
downloading any checkpoint:

```bash
python -m pytest tests/ -q
```

## Preparing models and data

GIFT scores frames with SigLIP. The scripts default to
`google/siglip-so400m-patch14-384` from the Hub; set `SIGLIP_PATH` to a local
directory to run offline.

Benchmarks are pulled from the Hub by lmms-eval on first use, so
MVBench, LongVideoBench, MLVU and VideoMME need no manual setup.

If your machine has no internet access, download the datasets elsewhere and
point `HF_HOME` at them; lmms-eval resolves every video path underneath it:

```bash
export HF_HOME=/path/to/huggingface
```

LVBench is the one exception: it is not on the Hub in a form this harness can
fetch, so place its videos in `./data/benchmarks/lvbench/videos/` (or edit
`dataset_path` in `lmms-eval/lmms_eval/tasks/lvbench/lvbench.yaml`).

## Running evaluation

Each script in `scripts/eval/` opens with an `edit me` block. Set your paths
there and run it:

```bash
bash scripts/eval/eval_llava_video.sh
```

```bash
# ------------------------------- edit me -------------------------------------
MODEL_PATH="/path/to/LLaVA-Video-7B-Qwen2"
SIGLIP_PATH="google/siglip-so400m-patch14-384"  # local dir, or HF hub id
TASKS="videomme"        # videomme | longvideobench_val_v | mlvu_dev | mvbench
FRAME_POOL=128          # dense candidate pool that GIFT scores
BUDGET=32               # frames kept and actually fed to the VLM
SELECTOR="GIFT"         # GIFT | AKS | BOLT | uniform
NUM_PROCESSES=8         # number of GPUs
```

`FRAME_POOL` is the dense candidate pool GIFT scores; `BUDGET` is how many
frames survive and reach the VLM. Switching `SELECTOR` between `GIFT`, `AKS`,
`BOLT`, and `uniform` reproduces any row of the table above. For the `uniform`
row, set `FRAME_POOL` equal to `BUDGET` so frames are spread over the whole video
rather than over a denser pool.

Scripts are provided for `llava_video`, `llava_onevision`, `qwen2_vl`, and
`qwen2_5_vl`.

## Using GIFT on your own model

The selector is independent of the evaluation harness:

```python
from gift import GIFTFrameSelector

selector = GIFTFrameSelector(select_frame_num=32, device="cuda")
keep = selector(frames, "Who scored the third goal?")  # frames: uint8 [T, H, W, 3]
frames = frames[keep.cpu().numpy()]                    # sorted frame indices
```

If you already have frame and query embeddings, call the scoring function
directly and skip SigLIP entirely:

```python
from gift import select_keyframes

keep = select_keyframes(frame_embeds, text_embed, selection_num=32, refine_group_size=9)
```

`refine_group_size` is the `B` of Budget-Aware Refinement; the paper uses 9, and
performance is stable for `B` between 6 and 10.

## Citation

```bibtex
@InProceedings{Ma_2026_CVPR,
    author    = {Ma, Junpeng and Zhou, Sashuai and Li, Guanghao and Gao, Xin and Cao, Yue and Zeng, Hengyu and Yan, Yuxiang and Wang, Zhibin and Song, Jun and Zheng, Bo and Zhang, Shanghang and Pu, Jian},
    title     = {GIFT: Global Irreplaceability Frame Targeting for Efficient Video Understanding},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {25610-25620}
}
```

## Acknowledgements

Built on [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) and
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), and scored with
[SigLIP](https://github.com/google-research/big_vision). The `baselines/`
directory reimplements [AKS](https://github.com/ncTimTang/AKS) and
[BOLT](https://github.com/sming256/BOLT) for comparison.

## License

Released under the Apache 2.0 License, following LLaVA-NeXT. See [LICENSE](LICENSE).
