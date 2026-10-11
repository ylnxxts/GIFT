# GIFT：面向高效视频理解的全局不可替代性关键帧选择

**GIFT**（CVPR 2026）官方代码实现。

[[论文](https://arxiv.org/abs/2603.25072)] · [English](README.md)

GIFT 是一种**免训练**的视频大模型关键帧选择方法。它不再贪心地追问"下一帧选哪个最好"，
而是直接衡量每一帧的**不可替代性**：一帧之所以重要，是因为它既与问题相关，又在整段视频中
找不到更好的替代者。

由于只改变送入模型的帧，GIFT 无需重新训练、不改动任何模型权重，即可接入任意视频大模型。

## 方法

**有向多样性（Directed Diversity）**：对于帧 `Fi`，它的*替代集合*是所有比它更贴合问题的帧。
有向多样性即 `Fi` 到该集合中最近成员的距离。将其与相关性相乘，就得到统一的不可替代性分数——
相关性与多样性不再需要人工权衡系数；而最相关的那一帧因为替代集合为空，永远不会被抑制。

**预算感知精修（Budget-Aware Refinement）**：一次性打分只会留下全局最不可替代的帧，
并压制它们的时序邻居，从而损失连贯性。GIFT 改为每轮选 `B` 帧并将其移出候选池；
移除一帧的同时也移除了它施加的抑制，先前被遮蔽的邻居得以浮现。
预算小时优先锁定关键事件，预算变大时逐步补足事件周边的上下文。

这两步都在 [`gift/irreplaceability.py`](gift/irreplaceability.py) 中，合计约 90 行 PyTorch 代码。

## 实验结果

LLaVA-Video-7B 在 MVBench / LongVideoBench / MLVU / VideoMME 上的平均分，
以 64 帧上限为 100%（论文表 1）：

| 帧预算 | 均匀采样 | BOLT (CVPR'25) | AKS (CVPR'25) | **GIFT** |
|---|---|---|---|---|
| 32 | 96.5 | 100.0 | 98.9 | **100.6** |
| 16 | 93.4 | 97.8 | 97.0 | **99.2** |
| 8  | 90.6 | 93.9 | 94.9 | **97.3** |
| 4  | 85.6 | 89.6 | 91.7 | **93.9** |

在 4 帧预算下 GIFT 恢复了全帧精度的 93.9%，比均匀采样高 8.3 个点。预算越紧，增益越明显。

## 仓库结构

```
gift/                  核心方法
  irreplaceability.py    有向多样性 + 预算感知精修
  selector.py            GIFTFrameSelector
  siglip.py              SigLIP 帧/问题打分器
  base.py                各选择器共用的分块编码逻辑
  registry.py            build_frame_selector(name, budget, device)
baselines/             AKS 与 BOLT 对比基线
llava/                 LLaVA-NeXT（内置）
lmms-eval/             带 GIFT 接入点的 lmms-eval 评测框架
scripts/eval/          开箱即用的评测脚本
tests/                 仅依赖 torch 的冒烟测试
```

对内置第三方代码的每一处改动都用 `GIFT Start` / `GIFT End` 注释包裹，
执行 `grep -rn "GIFT Start"` 即可看到全部接入点：上述四个模型封装（每处约十行），
以及两处与本方法无关的单行修复（数据集路径、VILA 加载函数签名）。

## 安装

```bash
conda create -n gift python=3.10 -y
conda activate gift
pip install -r requirements.txt
pip install flash-attn==2.5.7 --no-build-isolation   # 请匹配你的 CUDA/torch 版本
pip install -e .
```

打分部分只依赖 torch，因此无需下载任何权重即可先验证安装是否正常：

```bash
python -m pytest tests/ -q
```

## 准备模型与数据

GIFT 使用 SigLIP 为帧打分，脚本默认从 Hub 拉取 `google/siglip-so400m-patch14-384`。
离线运行时把 `SIGLIP_PATH` 改成本地权重目录即可。

评测数据由 lmms-eval 在首次运行时自动从 Hub 拉取，
MVBench、LongVideoBench、MLVU、VideoMME 都无需手动准备。

如果机器不能联网，可以在别处下载好数据集，然后把 `HF_HOME` 指过去，
lmms-eval 会在该目录下解析所有视频路径：

```bash
export HF_HOME=/path/to/huggingface
```

只有 LVBench 例外：它在本框架下没有可自动拉取的 Hub 版本，
需要把视频放到 `./data/benchmarks/lvbench/videos/`，
或修改 `lmms-eval/lmms_eval/tasks/lvbench/lvbench.yaml` 里的 `dataset_path`。

## 运行评测

`scripts/eval/` 下每个脚本开头都有一段 `edit me` 配置区，改好路径直接运行即可：

```bash
bash scripts/eval/eval_llava_video.sh
```

```bash
# ------------------------------- edit me -------------------------------------
MODEL_PATH="/path/to/LLaVA-Video-7B-Qwen2"
SIGLIP_PATH="google/siglip-so400m-patch14-384"  # 本地目录或 HF hub id
TASKS="videomme"        # videomme | longvideobench_val_v | mlvu_dev | mvbench
FRAME_POOL=128          # GIFT 打分的稠密候选帧池
BUDGET=32               # 最终送入视频大模型的帧数
SELECTOR="GIFT"         # GIFT | AKS | BOLT | uniform
NUM_PROCESSES=8         # GPU 数量
```

`FRAME_POOL` 是 GIFT 打分的稠密候选帧池，`BUDGET` 是最终送入视频大模型的帧数。
把 `SELECTOR` 在 `GIFT`、`AKS`、`BOLT`、`uniform` 之间切换，即可复现上表任意一行。复现 `uniform` 一行时，请把 `FRAME_POOL` 设为与 `BUDGET` 相同，这样帧是在整段视频上均匀采样，而不是在更密的候选池里采样。

仓库提供了 `llava_video`、`llava_onevision`、`qwen2_vl`、`qwen2_5_vl` 四套脚本。

## 在自己的模型上使用 GIFT

选择器本身不依赖评测框架：

```python
from gift import GIFTFrameSelector

selector = GIFTFrameSelector(select_frame_num=32, device="cuda")
keep = selector(frames, "Who scored the third goal?")  # frames: uint8 [T, H, W, 3]
frames = frames[keep.cpu().numpy()]                    # 已按时序排序的帧下标
```

如果你已经有帧特征和文本特征，可以直接调用打分函数，完全跳过 SigLIP：

```python
from gift import select_keyframes

keep = select_keyframes(frame_embeds, text_embed, selection_num=32, refine_group_size=9)
```

`refine_group_size` 即预算感知精修中的 `B`，论文取 9；`B` 在 6～10 之间性能都很稳定。

## 引用

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

## 致谢

本项目基于 [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) 与
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) 构建，
打分模型为 [SigLIP](https://github.com/google-research/big_vision)。
`baselines/` 目录复现了 [AKS](https://github.com/ncTimTang/AKS) 与
[BOLT](https://github.com/sming256/BOLT) 作为对比方法。

## 许可证

遵循 LLaVA-NeXT 采用 Apache 2.0 许可证，详见 [LICENSE](LICENSE)。
