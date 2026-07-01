#!/usr/bin/env bash
set -e

# Stage 2.2: Full MMU Training (MMU + LM)
# Continue from Stage 2.1 projector checkpoint, train projector + backbone
# Default: single-node (8 GPUs). For multi-node, set HOST_NUM=8

export ASSETS_BASE="${ASSETS_BASE:-public_assets}"
export CKPT_DIR="${CKPT_DIR:-outputs/stage2_1_projector/ckpt/0003000}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage2_2_mmu}"

export HOST_NUM="${HOST_NUM:-1}"
export HOST_GPU_NUM="${HOST_GPU_NUM:-8}"
export OUTPUT_PATH="${OUTPUT_DIR}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

[ -n "${MASTER_ADDR}" ] && export RDZV_ENDPOINT="${MASTER_ADDR}:${MASTER_PORT:-29500}"

bash launch/workers/run_train.sh torchrun \
    train/configs/stage2_2_mmu.yaml \
    --ckpt-dir "${CKPT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-shard "${HOST_GPU_NUM}" \
    --max-steps 20000 \
    --warmup-steps 1000 \
    --shield-step 5000 \
    --save-interval 1000 \
    --gradient-accumulation-steps 1 \
    --max-seq-len 8192 \
    --init-save \
    --use-orth \
    "$@"
