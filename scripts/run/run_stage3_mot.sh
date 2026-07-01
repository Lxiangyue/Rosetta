#!/usr/bin/env bash
set -e

# Use provided paths or defaults to current directory
export ASSETS_BASE="${ASSETS_BASE:-public_assets}"
export CKPT_DIR="${CKPT_DIR:-checkpoints/MoT-4.5B-A1B-stage3-init/hf_weights}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage3_mm_mot}"
export HOST_GPU_NUM="${HOST_GPU_NUM:-8}"
export OUTPUT_PATH="${OUTPUT_DIR}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

[ -n "${MASTER_ADDR}" ] && export RDZV_ENDPOINT="${MASTER_ADDR}:${MASTER_PORT:-29500}"

bash launch/workers/run_train.sh torchrun \
    train/configs/stage3_mm_mot.yaml \
    --ckpt-dir "${CKPT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-shard "${HOST_GPU_NUM}" \
    "$@"
