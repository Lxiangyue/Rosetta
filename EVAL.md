# Evaluation Guide

## 📕 Contents
- [Quick Start](#-quick-start)
- [Examples](#-examples)
- [Benchmark Scripts](#-benchmark-scripts)
- [Model Checkpoints](#-model-checkpoints)
- [Config Selection](#-config-selection)
- [Benchmark Results](#-benchmark-results)

> This guide covers the evaluation interface, benchmark scripts, model checkpoints, and config choices for evaluating Rosetta, MoE, and MoT across language, coding, multimodal understanding, and text-to-image benchmarks.

---

## ⚡ Quick Start

All evaluation scripts share the same interface:

- `EXP`: checkpoint folder, such as `./checkpoints/Rosetta-3.8B-A1B`.
- `CONFIG`: architecture config under `./evaluation/configs/`.
- `ASSETS_BASE`: shared assets directory. Defaults to `./public_assets`.
- `CKPT_DIR`: weights directory. Defaults to `${EXP}/hf_weights`.
- `SAMPLE_OUT`: output directory. Defaults to `${EXP}/outputs`.

### Default 8-GPU Evaluation

By default, the scripts evaluate the final Rosetta checkpoint with 8 GPUs:

```bash
bash scripts/eval/eval_arc_c.sh
```

### Single-GPU Evaluation

Override `HOST_GPU_NUM` to run on one GPU:

```bash
HOST_GPU_NUM=1 bash scripts/eval/eval_arc_c.sh
```

### Multi-Node Evaluation

Create a `hosts.txt` file with one node per line, then launch from the chief node:

```bash
hostfile=hosts.txt HOST_NUM=8 HOST_GPU_NUM=8 \
bash launch/run_multinode.sh scripts/eval/eval_arc_c.sh
```

### Change Model

To evaluate a different model, replace `EXP` with a checkpoint from [Model Checkpoints](#-model-checkpoints), and use the matching `CONFIG` from [Config Selection](#-config-selection):

```bash
EXP=checkpoints/MoE-3.8B-A1B \
CONFIG=evaluation/configs/moe.yaml \
bash scripts/eval/eval_arc_c.sh
```

### Change Benchmark

To evaluate a different benchmark, keep the same `EXP` and `CONFIG`, and replace the script with one from [Benchmark Scripts](#-benchmark-scripts):

```bash
EXP=checkpoints/Rosetta-3.8B-A1B \
CONFIG=evaluation/configs/rosetta.yaml \
bash scripts/eval/eval_mmlu.sh
```

### Output Layout

By default, each benchmark writes outputs under the evaluated model directory, `${EXP}/outputs`. For example:

```text
checkpoints/Rosetta-3.8B-A1B/
├── hf_weights/
└── outputs/
    └── arc_challenge__arc_challenge_1/
        ├── metric_results/
        │   └── arc_challenge.json
        └── results/
            └── all_results.csv
```

---

## 🚀 Examples

### Evaluate Final Models On ARC-Challenge

Rosetta is the default:

```bash
bash scripts/eval/eval_arc_c.sh
```

Evaluate MoE:

```bash
EXP=checkpoints/MoE-3.8B-A1B \
CONFIG=evaluation/configs/moe.yaml \
bash scripts/eval/eval_arc_c.sh
```

Evaluate MoT:

```bash
EXP=checkpoints/MoT-4.5B-A1B \
CONFIG=evaluation/configs/mot.yaml \
bash scripts/eval/eval_arc_c.sh
```

### Evaluate Stage Checkpoints

To reproduce the eval curve across all stages, run the same benchmark across stage checkpoints. Example for Rosetta on MMLU:

```bash
for EXP in checkpoints/Rosetta-3.8B-A1B-init \
           checkpoints/Rosetta-3.8B-A1B-stage1-lm \
           checkpoints/Rosetta-3.8B-A1B-stage2-lm-mmu-warmup \
           checkpoints/Rosetta-3.8B-A1B-stage2-lm-mmu \
           checkpoints/Rosetta-3.8B-A1B; do
    EXP=${EXP} \
    CONFIG=evaluation/configs/rosetta.yaml \
    bash scripts/eval/eval_mmlu.sh
done
```

---

## 📋 Benchmark Scripts

| Benchmark | Task | Script |
|:----------|:-----|:-------|
| ARC-Challenge | Language | `scripts/eval/eval_arc_c.sh` |
| MMLU | Language | `scripts/eval/eval_mmlu.sh` |
| BBH | Language | `scripts/eval/eval_bbh.sh` |
| MBPP | Coding | `scripts/eval/eval_mbpp.sh` |
| MMMU | Multimodal Understanding | `scripts/eval/eval_mmmu.sh` |
| MMBench | Multimodal Understanding | `scripts/eval/eval_mmbench.sh` |
| POPE | Hallucination | `scripts/eval/eval_pope.sh` |
| AI2D | Diagram Understanding | `scripts/eval/eval_ai2d.sh` |
| RealWorldQA | Real-world VQA | `scripts/eval/eval_realworldqa.sh` |
| T2I-CompBench | Image Generation | `scripts/eval/eval_t2i_compbench.sh` |
| COCO | Image Generation | `scripts/eval/eval_coco.sh` |

---

## 📦 Model Checkpoints

We release 15 model checkpoints across 3 architectures and 5 training stages. The final 3 checkpoints are for benchmark reproduction:

| Checkpoint | Capabilities | Total / Active | HuggingFace |
|:-----------|:-------------|:--------------:|:-----------:|
| Rosetta-3.8B-A1B | LM + MMU + T2I | 3.8B / 0.97B | 🤗 [Download](https://huggingface.co/tencent/Rosetta-inference/tree/main/checkpoints/Rosetta-3.8B-A1B) |
| MoE-3.8B-A1B | LM + MMU + T2I | 3.8B / 0.97B | 🤗 [Download](https://huggingface.co/tencent/Rosetta-inference/tree/main/checkpoints/MoE-3.8B-A1B) |
| MoT-4.5B-A1B | LM + MMU + T2I | 4.5B / 0.97B | 🤗 [Download](https://huggingface.co/tencent/Rosetta-inference/tree/main/checkpoints/MoT-4.5B-A1B) |

<details>

<summary><b>Full checkpoint list — all 15 models across 3 architectures × 5 training stages, plus the MoT Stage 3 init checkpoint</b></summary>
<br>

| Checkpoint | Stage | Iter | Total / Active | HuggingFace |
|:-----------|:------|-----:|:--------------:|:-----------:|
| Rosetta-3.8B-A1B-init | Upcycling init | 0 | 3.8B / 0.97B | 🤗 [Download](https://huggingface.co/tencent/Rosetta-inference/tree/main/checkpoints/Rosetta-3.8B-A1B-init) |
| Rosetta-3.8B-A1B-stage1-lm | LM | 35K | 3.8B / 0.97B | 🤗 [Download](https://huggingface.co/tencent/Rosetta-inference/tree/main/checkpoints/Rosetta-3.8B-A1B-stage1-lm) |
| Rosetta-3.8B-A1B-stage2-lm-mmu-warmup | LM+MMU warmup | 3K | 3.8B / 0.97B | 🤗 [Download](https://huggingface.co/tencent/Rosetta-inference/tree/main/checkpoints/Rosetta-3.8B-A1B-stage2-lm-mmu-warmup) |
| Rosetta-3.8B-A1B-stage2-lm-mmu | LM+MMU | 20K | 3.8B / 0.97B | 🤗 [Download](https://huggingface.co/tencent/Rosetta-inference/tree/main/checkpoints/Rosetta-3.8B-A1B-stage2-lm-mmu) |
| Rosetta-3.8B-A1B | LM+MMU+T2I | 400K | 3.8B / 0.97B | 🤗 [Download](https://huggingface.co/tencent/Rosetta-inference/tree/main/checkpoints/Rosetta-3.8B-A1B) |
| MoE-3.8B-A1B-init | Upcycling init | 0 | 3.8B / 0.97B | 🤗 [Download](https://huggingface.co/tencent/Rosetta-inference/tree/main/checkpoints/MoE-3.8B-A1B-init) |
| MoE-3.8B-A1B-stage1-lm | LM | 35K | 3.8B / 0.97B | 🤗 [Download](https://huggingface.co/tencent/Rosetta-inference/tree/main/checkpoints/MoE-3.8B-A1B-stage1-lm) |
| MoE-3.8B-A1B-stage2-lm-mmu-warmup | LM+MMU warmup | 3K | 3.8B / 0.97B | 🤗 [Download](https://huggingface.co/tencent/Rosetta-inference/tree/main/checkpoints/MoE-3.8B-A1B-stage2-lm-mmu-warmup) |
| MoE-3.8B-A1B-stage2-lm-mmu | LM+MMU | 20K | 3.8B / 0.97B | 🤗 [Download](https://huggingface.co/tencent/Rosetta-inference/tree/main/checkpoints/MoE-3.8B-A1B-stage2-lm-mmu) |
| MoE-3.8B-A1B | LM+MMU+T2I | 400K | 3.8B / 0.97B | 🤗 [Download](https://huggingface.co/tencent/Rosetta-inference/tree/main/checkpoints/MoE-3.8B-A1B) |
| MoT-4.5B-A1B-init | Upcycling init | 0 | 4.5B / 0.97B | 🤗 [Download](https://huggingface.co/tencent/Rosetta-inference/tree/main/checkpoints/MoT-4.5B-A1B-init) |
| MoT-4.5B-A1B-stage1-lm | LM | 35K | 4.5B / 0.97B | 🤗 [Download](https://huggingface.co/tencent/Rosetta-inference/tree/main/checkpoints/MoT-4.5B-A1B-stage1-lm) |
| MoT-4.5B-A1B-stage2-lm-mmu-warmup | LM+MMU warmup | 3K | 4.5B / 0.97B | 🤗 [Download](https://huggingface.co/tencent/Rosetta-inference/tree/main/checkpoints/MoT-4.5B-A1B-stage2-lm-mmu-warmup) |
| MoT-4.5B-A1B-stage2-lm-mmu | LM+MMU | 20K | 4.5B / 0.97B | 🤗 [Download](https://huggingface.co/tencent/Rosetta-inference/tree/main/checkpoints/MoT-4.5B-A1B-stage2-lm-mmu) |
| MoT-4.5B-A1B-stage3-init | LM+MMU+T2I | 0 | 4.5B / 0.97B | 🤗 [Download](https://huggingface.co/tencent/Rosetta-inference/tree/main/checkpoints/MoT-4.5B-A1B-stage3-init) |
| MoT-4.5B-A1B | LM+MMU+T2I | 400K | 4.5B / 0.97B | 🤗 [Download](https://huggingface.co/tencent/Rosetta-inference/tree/main/checkpoints/MoT-4.5B-A1B) |

> All models are trained within the Transfusion framework on top of the [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) language backbone, using identical training data and hyperparameters for fair comparison.

> `MoT-4.5B-A1B-stage3-init` is an additional Stage 3 initialization checkpoint used before T2I training. It is useful for validating the expanded MoT architecture and should be evaluated with `evaluation/configs/mot.yaml`.

</details>

---

## 🧩 Config Selection

Use the config that matches the model architecture:

| Checkpoint family | Config |
|:------------------|:-------|
| Rosetta-3.8B-A1B* | `evaluation/configs/rosetta.yaml` |
| MoE-3.8B-A1B* | `evaluation/configs/moe.yaml` |
| MoT checkpoints before Stage 3 | `evaluation/configs/mot_und.yaml` |
| MoT-4.5B-A1B-stage3-init, MoT-4.5B-A1B | `evaluation/configs/mot.yaml` |

> Note: For MoT, use `mot_und.yaml` for `init`, `stage1-lm`, `stage2-lm-mmu-warmup`, and `stage2-lm-mmu`, which only have the understanding stream. Use `mot.yaml` after the Stage 2 to Stage 3 expansion has created the generation stream.

<details>
<summary><b>Full mapping between checkpoint names and configs</b></summary>
<br>

| EXP | CONFIG |
|:----|:-------|
| Rosetta-3.8B-A1B-init | `evaluation/configs/rosetta.yaml` |
| Rosetta-3.8B-A1B-stage1-lm | `evaluation/configs/rosetta.yaml` |
| Rosetta-3.8B-A1B-stage2-lm-mmu-warmup | `evaluation/configs/rosetta.yaml` |
| Rosetta-3.8B-A1B-stage2-lm-mmu | `evaluation/configs/rosetta.yaml` |
| Rosetta-3.8B-A1B | `evaluation/configs/rosetta.yaml` |
| MoE-3.8B-A1B-init | `evaluation/configs/moe.yaml` |
| MoE-3.8B-A1B-stage1-lm | `evaluation/configs/moe.yaml` |
| MoE-3.8B-A1B-stage2-lm-mmu-warmup | `evaluation/configs/moe.yaml` |
| MoE-3.8B-A1B-stage2-lm-mmu | `evaluation/configs/moe.yaml` |
| MoE-3.8B-A1B | `evaluation/configs/moe.yaml` |
| MoT-4.5B-A1B-init | `evaluation/configs/mot_und.yaml` |
| MoT-4.5B-A1B-stage1-lm | `evaluation/configs/mot_und.yaml` |
| MoT-4.5B-A1B-stage2-lm-mmu-warmup | `evaluation/configs/mot_und.yaml` |
| MoT-4.5B-A1B-stage2-lm-mmu | `evaluation/configs/mot_und.yaml` |
| MoT-4.5B-A1B-stage3-init | `evaluation/configs/mot.yaml` |
| MoT-4.5B-A1B | `evaluation/configs/mot.yaml` |

</details>

---

## 📊 Benchmark Results

The following table is the main benchmark table from our paper. It reports the final **Rosetta-3.8B-A1B**, **MoE-3.8B-A1B**, and **MoT-4.5B-A1B** models, trained under identical data and hyperparameters after the full LM+MMU+T2I stage.

<p align="center">
  <img src="assets/table.png" alt="Comprehensive Performance Evaluations" width="80%">
</p>

**Reproduce ARC-Challenge scores:**
```bash
# Rosetta
bash scripts/eval/eval_arc_c.sh

# MoE
EXP=checkpoints/MoE-3.8B-A1B CONFIG=evaluation/configs/moe.yaml bash scripts/eval/eval_arc_c.sh

# MoT
EXP=checkpoints/MoT-4.5B-A1B CONFIG=evaluation/configs/mot.yaml bash scripts/eval/eval_arc_c.sh
```

**Reproduce all benchmarks:**
```bash
# Rosetta
bash scripts/eval/eval_arc_c.sh && \
bash scripts/eval/eval_mmlu.sh && \
bash scripts/eval/eval_bbh.sh && \
bash scripts/eval/eval_mbpp.sh && \
bash scripts/eval/eval_mmmu.sh && \
bash scripts/eval/eval_mmbench.sh && \
bash scripts/eval/eval_pope.sh && \
bash scripts/eval/eval_ai2d.sh && \
bash scripts/eval/eval_realworldqa.sh && \
bash scripts/eval/eval_coco.sh && \
bash scripts/eval/eval_t2i_compbench.sh
```
