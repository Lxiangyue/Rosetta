#!/usr/bin/env bash
set -e

# Use provided paths or defaults to current directory
export ASSETS_BASE="${ASSETS_BASE:-public_assets}"
export CKPT_DIR="${CKPT_DIR:-checkpoints/MoE-3.8B-A1B-stage2-lm-mmu/hf_weights}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage3_mm_moe}"
export HOST_GPU_NUM="${HOST_GPU_NUM:-8}"
export OUTPUT_PATH="${OUTPUT_DIR}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

[ -n "${MASTER_ADDR}" ] && export RDZV_ENDPOINT="${MASTER_ADDR}:${MASTER_PORT:-29500}"

bash launch/workers/run_train.sh torchrun \
    train/configs/stage3_mm_moe.yaml \
    --ckpt-dir "${CKPT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-shard "${HOST_GPU_NUM}" \
    --max-steps 100 \
    --warmup-steps 20 \
    --save-interval 100 \
    --gradient-accumulation-steps 8 \
    --max-seq-len 2048 \
    --init-save \
    "$@"
