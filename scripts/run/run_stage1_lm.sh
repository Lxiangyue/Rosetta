#!/usr/bin/env bash
set -e

# Stage 1: Language Model Training (Text-Only)
# Train upcycled MoE (3 routed + 1 shared) on text-only data
# Default: single-node (8 GPUs). For multi-node, set HOST_NUM=8

export ASSETS_BASE="${ASSETS_BASE:-public_assets}"
export CKPT_DIR="${CKPT_DIR:-outputs/Qwen3-0.6B-Base-upcycling-ours-3e-scale05}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage1_lm}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"

export HOST_NUM="${HOST_NUM:-1}"
export HOST_GPU_NUM="${HOST_GPU_NUM:-8}"
export OUTPUT_PATH="${OUTPUT_DIR}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

[ -n "${MASTER_ADDR}" ] && export RDZV_ENDPOINT="${MASTER_ADDR}:${MASTER_PORT:-29500}"

bash launch/workers/run_train.sh torchrun \
    train/configs/stage1_lm.yaml \
    --ckpt-dir "${CKPT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-shard "${HOST_GPU_NUM}" \
    --max-steps 35000 \
    --warmup-steps 500 \
    --save-interval 1000 \
    --gradient-accumulation-steps 1 \
    --max-seq-len "${MAX_SEQ_LEN}" \
    --init-save \
    "$@"
