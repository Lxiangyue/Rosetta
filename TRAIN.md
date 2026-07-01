# Training Guide

## 📌 Contents

- [Training Roadmap](#training-roadmap)
- [Training From Scratch](#training-from-scratch)
- [Model Architectures](#model-architectures)
- [Core Parameters](#core-parameters)
- [Advanced Parameters](#advanced-parameters)

<a id="training-roadmap"></a>
## 🗺️ Training Roadmap

Rosetta, MoE, and MoT are trained through the same composable pretraining pipeline: **Stage 0** (Upcycle) → **Stage 1** (LM) → **Stage 2** (LM+MMU) → **Stage 3** (LM+MMU+T2I).

Depending on your goals and available resources, you can run this roadmap in two ways:
* **Standard Run (Stage 3):** Most users could start directly from Stage 3 (+T2I) using our pre-released checkpoints, as demonstrated in the repository [README.md](README.md).
* **End-to-End Run (Stage 0–3):** If you wish to reproduce the entire pretraining process from the very beginning, please follow the [Training From Scratch](#training-from-scratch) section below.

<a id="training-from-scratch"></a>
## 🧱 Training From Scratch

### Hardware & Data Scale

* **Demo Run (8 GPUs):** The default configuration is optimized for a single-node setup (8 GPUs) using the provided **example dataset** to verify the pipeline end-to-end. Please note that this example dataset is not the actual training data used in our paper.
* **Paper Production (64–256 GPUs):** Our paper pretraining used 256 GPUs (at Stage 1) and 64 GPUs (at Stage 2 and 3). For comprehensive details about our actual training datasets, please refer to our paper.

### Rosetta

Rosetta first builds a sparse language model, expands experts for multimodal training:

<details>
<summary><b>Rosetta stage entrypoints</b></summary>
<br>

| Stage | Purpose | Rosetta entrypoint |
|:--|:--|:--|
| Stage 0 | Upcycle dense Qwen3-0.6B-Base to sparse model | `scripts/run/run_stage0_upcycle.sh` |
| Stage 1 | Text-only LM training | `scripts/run/run_stage1_lm.sh` |
| Stage 1 -> 2 | Expand 3 experts to 12 experts (3 text + 3 vit + 6 vae) | `scripts/run/run_expand_stage1_to_stage2.sh` |
| Stage 2.1 | MMU projector warmup with frozen backbone | `scripts/run/run_stage2_1_projector.sh` |
| Stage 2.2 | Full MMU + LM training | `scripts/run/run_stage2_2_mmu.sh` |
| Stage 3 | Add T2I training (LM+MMU+T2I) | `scripts/run/run_stage3.sh` |

</details>

```bash
# Download Qwen3-0.6B-Base model weights (if not already downloaded)
hf download Qwen/Qwen3-0.6B-Base --local-dir checkpoints/Qwen3-0.6B-Base

# Stage 0: Upcycle dense Qwen3-0.6B-Base to init, no training
bash scripts/run/run_stage0_upcycle.sh

# Stage 1: LM training (35k steps)
bash scripts/run/run_stage1_lm.sh  # Single-node
hostfile=hosts.txt HOST_NUM=32 HOST_GPU_NUM=8 bash launch/run_multinode.sh scripts/run/run_stage1_lm.sh  # Multi-node

# ### Stage 1 → Stage 2; no training
export STAGE1_CKPT=outputs/stage1_lm/ckpt/0035000
export STAGE2_INIT_CKPT=outputs/stage2_init
bash scripts/run/run_expand_stage1_to_stage2.sh

# Stage 2.1: Projector warmup (3k steps)
bash scripts/run/run_stage2_1_projector.sh  # Single-node
hostfile=hosts.txt HOST_NUM=8 HOST_GPU_NUM=8 bash launch/run_multinode.sh scripts/run/run_stage2_1_projector.sh  # Multi-node

# Stage 2.2: Full LM+MMU training (20k steps)
bash scripts/run/run_stage2_2_mmu.sh  # Single-node
hostfile=hosts.txt HOST_NUM=8 HOST_GPU_NUM=8 bash launch/run_multinode.sh scripts/run/run_stage2_2_mmu.sh  # Multi-node

# Stage 3: Full LM+MMU+T2I training
bash scripts/run/run_stage3.sh # Single-node
hostfile=hosts.txt HOST_NUM=8 HOST_GPU_NUM=8 bash launch/run_multinode.sh scripts/run/run_stage3.sh  # Multi-node
```

### MoE Baseline

Standard MoE (12 routed + 1 shared):

<details>
<summary><b>MoE stage entrypoints</b></summary>
<br>

| Stage | Purpose | MoE entrypoint |
|:--|:--|:--|
| Stage 0 | Upcycle dense Qwen3-0.6B-Base to sparse model | `scripts/run/run_stage0_upcycle_moe.sh` |
| Stage 1 | Text-only LM training | `scripts/run/run_stage1_lm_moe.sh` |
| Stage 2.1 | MMU projector warmup with frozen backbone | `scripts/run/run_stage2_1_projector_moe.sh` |
| Stage 2.2 | Full MMU + LM training | `scripts/run/run_stage2_2_mmu_moe.sh` |
| Stage 3 | Add T2I training (LM+MMU+T2I) | `scripts/run/run_stage3_moe.sh` |

</details>

```bash
# Download Qwen3-0.6B-Base model weights (if not already downloaded)
hf download Qwen/Qwen3-0.6B-Base --local-dir checkpoints/Qwen3-0.6B-Base

# Stage 0: Upcycle to MoE
bash scripts/run/run_stage0_upcycle_moe.sh

# Stage 1: LM training (35k steps)
bash scripts/run/run_stage1_lm_moe.sh  # Single-node
hostfile=hosts.txt HOST_NUM=32 HOST_GPU_NUM=8 bash launch/run_multinode.sh scripts/run/run_stage1_lm_moe.sh  # Multi-node

# Stage 2.1: Projector warmup (3k steps)
bash scripts/run/run_stage2_1_projector_moe.sh  # Single-node
hostfile=hosts.txt HOST_NUM=8 HOST_GPU_NUM=8 bash launch/run_multinode.sh scripts/run/run_stage2_1_projector_moe.sh  # Multi-node

# Stage 2.2: Full MMU training (20k steps)
bash scripts/run/run_stage2_2_mmu_moe.sh  # Single-node
hostfile=hosts.txt HOST_NUM=8 HOST_GPU_NUM=8 bash launch/run_multinode.sh scripts/run/run_stage2_2_mmu_moe.sh  # Multi-node

# Stage 3: Full LM + MMU + T2I training
bash scripts/run/run_stage3_moe.sh # Single-node
hostfile=hosts.txt HOST_NUM=8 HOST_GPU_NUM=8 bash launch/run_multinode.sh scripts/run/run_stage3_moe.sh # Multi-node
```

> **Note:** MoE uses 12 experts directly, no checkpoint expansion needed (unlike Rosetta).

### MoT Baseline

Mixture-of-Transformers uses a single und stream before Stage 3, then expands to separate und/gen streams at Stage 3 T2I training:

<details>
<summary><b>MoT stage entrypoints</b></summary>
<br>

| Stage | Purpose | MoT entrypoint | Note |
|:--|:--|:--|:--|
| Stage 0 | Upcycle dense Qwen3-0.6B-Base to sparse model | `scripts/run/run_stage0_upcycle_mot.sh` | `und`-only |
| Stage 1 | Text-only LM training | `scripts/run/run_stage1_lm_mot.sh` | `und`-only LM |
| Stage 2.1 | MMU projector warmup with frozen backbone | `scripts/run/run_stage2_1_projector_mot.sh` | `und`-only projector warmup |
| Stage 2.2 | Full MMU + LM training | `scripts/run/run_stage2_2_mmu_mot.sh` | `und`-only LM + MMU |
| Stage 2 -> 3 | Prepare Stage 3 MoT initialization | `scripts/run/run_expand_mot_stage2_to_stage3.sh` | prepare `gen` stream |
| Stage 3 | Add T2I training (LM+MMU+T2I) | `scripts/run/run_stage3_mot.sh` | `und-gen` streams |

</details>

```bash
# Download Qwen3-0.6B-Base model weights (if not already downloaded)
hf download Qwen/Qwen3-0.6B-Base --local-dir checkpoints/Qwen3-0.6B-Base

# Stage 0: Upcycle to MoT
bash scripts/run/run_stage0_upcycle_mot.sh

# Stage 1: LM training (35k steps)
bash scripts/run/run_stage1_lm_mot.sh  # Single-node
hostfile=hosts.txt HOST_NUM=32 HOST_GPU_NUM=8 bash launch/run_multinode.sh scripts/run/run_stage1_lm_mot.sh  # Multi-node

# Stage 2.1: Projector warmup (3k steps)
bash scripts/run/run_stage2_1_projector_mot.sh  # Single-node
hostfile=hosts.txt HOST_NUM=8 HOST_GPU_NUM=8 bash launch/run_multinode.sh scripts/run/run_stage2_1_projector_mot.sh  # Multi-node

# Stage 2.2: Full MMU training (20k steps)
bash scripts/run/run_stage2_2_mmu_mot.sh  # Single-node
hostfile=hosts.txt HOST_NUM=8 HOST_GPU_NUM=8 bash launch/run_multinode.sh scripts/run/run_stage2_2_mmu_mot.sh  # Multi-node

# Expand Stage 2 VLM to Stage 3 MoT init (copy und -> gen stream)
bash scripts/run/run_expand_mot_stage2_to_stage3.sh

# Stage 3: Full LM + MMU + T2I training with use-mot enabled
CKPT_DIR=outputs/stage3_init_mot bash scripts/run/run_stage3_mot.sh # Single-node
hostfile=hosts.txt HOST_NUM=8 HOST_GPU_NUM=8 CKPT_DIR=outputs/stage3_init_mot bash launch/run_multinode.sh scripts/run/run_stage3_mot.sh # Multi-node
```

**Note:** MoT does not enable `use-mot` during Stage 1 / Stage 2. Those stages train the und stream as a 7-routed-expert VLM. The gen stream is created only at the Stage 2 → Stage 3 boundary by `expand_mot_stage2_to_stage3.py`; Stage 3 then runs with `train/configs/stage3_mm_mot.yaml` (`use-mot=True`).

---

## 🏗️ Model Architectures

**Rosetta**
- `qwen3-06b-base-upcycling-ours-lm` — Stage 1 LM (3 routed + 1 shared)
- `qwen3-06b-upcycling-ours-mm` — Stage 2/3 MM (3 text + 3 vit + 6 vae + 1 shared)

**MoE Baseline**
- `qwen3-06b-base-upcycling-moe-lm-deepseek` — Stage 1 LM (12 routed + 1 shared)
- `qwen3-06b-upcycling-moe-mm-deepseek` — Stage 2/3 MM (12 routed + 1 shared)

**MoT Baseline**
- `qwen3-06b-base-mot-lm` — Stage 1 LM (7 routed + 1 shared, und stream only)
- `qwen3-06b-mot` — Stage 2 MMU uses the und stream only; Stage 3 uses expanded und/gen streams (und: 7 routed + 1 shared, gen: 6 routed + 1 shared)

---

## ⚙️ Core Parameters

| Parameter | Description |
|:----------|:------------|
| `ASSETS_BASE` | Shared assets directory containing VAE, ViT, tokenizer, and evaluation data. |
| `CKPT_DIR` | Input checkpoint directory for training; checkpoint path for evaluation. |
| `OUTPUT_DIR` | Training output directory. Checkpoints are saved under `OUTPUT_DIR/ckpt/`. |
| `--max-steps` | Number of optimizer steps. |
| `--reproduce` | Enable deterministic training kernels and seed all RNGs. This is slower, but repeated runs in the same environment should produce the same result. |
| `--no-save-optimizer` | Save model weights only. Useful for demo/eval runs; the resulting checkpoints cannot be used with `--resume`. |
| `--gradient-accumulation-steps` | Microbatches accumulated per optimizer step; used to match global batch size on fewer GPUs. |
| `--num-shard` | GPUs per FSDP shard group for `HYBRID_SHARD`; also controls fixed modality allocation groups. |
| `--data-weights` | JSON weights for `lm/mmu/t2i` sampling in Stage 3. |
| `use_modality_routing` | Rosetta: route text, ViT, and VAE tokens to modality-specific experts. |
| `num_text_experts`, `num_vit_experts`, `num_vae_experts` | Rosetta: Expert counts for text, ViT, and VAE routes. Released Stage 2/3 configs use `3/3/6` routed experts. |
| `moe_mixed_mlp`, `moe_topk` | Shared expert count and routed top-k. Released configs use one shared expert and top-2 routing. |
| `--use-orth` | Rosetta: Enable MAOP for shared experts. |
| `--shield-step` | Rosetta: Block non-text gradients to the shared expert during warmup. |
| `--use-lr-diff`, `--vae-lr` | Rosetta: Use a larger learning rate for VAE-related parameters. |
| `use_mot` / `- use-mot` | MoT: enable separated `und` and `gen` streams after Stage 2 → Stage 3 expansion. |


### Rosetta Implement Cores

- **Composable FFN:** `use_modality_routing` routes text, ViT, and VAE tokens to modality-specific experts while keeping a global shared expert.
- **MAOP:** `--use-orth` projects conflicting gradient components away from the shared expert, reducing destructive interference across text, ViT, and VAE objectives.
- **Shield Step:** `--shield-step` blocks non-text gradients to the shared expert during early Stage 3 warmup, protecting language ability while adding T2I training.
- **Differential LR:** `--use-lr-diff`, `--vae-lr`, `--lr` uses defult learning rate 1e-4 for VAE-related parameters and base lr for other parts.

---

## 🛠️ Advanced Parameters

### Auto-resume

Add `--resume` to automatically resume from the latest checkpoint if training is interrupted. But do not use `--resume` from checkpoints produced with `--no-save-optimizer`; those checkpoints intentionally omit optimizer and scheduler state and are meant for evaluation or initialization only.

### FSDP Sharding

- `HYBRID_SHARD`: Shard within each node, replicate across nodes (default, fastest)
- `FULL_SHARD`: Shard across all GPUs (lowest memory, slower)

Set in YAML or via `--sharding-strategy FULL_SHARD --num-shard 32` (for 4×8 GPUs).

### Sequence Packing

Enabled by default. Packs multiple samples into `--max-seq-len` sequences to maximize GPU utilization. Handles variable-length text and multi-resolution images automatically.

### Mixed-Modality Sampling

Set target modality ratios in YAML with `data-weights`, for example `{"lm": 0.15, "mmu": 0.25, "t2i": 0.60}`.

- For large-scale runs, training uses fixed rank allocation: each FSDP shard group is assigned one modality according to these weights. This keeps data shapes and compute paths stable across ranks.

- For small jobs, such as the default 8-GPU demo with `--num-shard 8`, there is only one shard group. The code then falls back to a local weighted schedule across gradient accumulation microbatches. This is expected and keeps the demo memory-friendly. The demo uses `--gradient-accumulation-steps 8`, so `8 GPUs × acc 8 = 64` microbatches per optimizer step, matching a 64-GPU run with `--gradient-accumulation-steps 1`.

---

## 💬 Contact

For questions or issues, please open a GitHub issue or contact us.
