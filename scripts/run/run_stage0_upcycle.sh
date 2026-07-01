#!/usr/bin/env bash
set -e

# Stage 0: Upcycle Qwen3-0.6B-Base (dense) → MoE (3 routed + 1 shared)
# Convert dense checkpoint to MoE format with composable FFN initialization

export ASSETS_BASE="${ASSETS_BASE:-public_assets}"
export SRC_CKPT="${SRC_CKPT:-checkpoints/Qwen3-0.6B-Base}"
export DST_CKPT="${DST_CKPT:-outputs/Qwen3-0.6B-Base-upcycling-ours-3e-scale05}"

if [ -d "${DST_CKPT}" ]; then
    echo "MoE checkpoint already exists at ${DST_CKPT}"
    echo "Skipping conversion."
    exit 0
fi

python scripts/convert_qwen3_dense_to_moe.py \
    --src "${SRC_CKPT}" \
    --dst "${DST_CKPT}" \
    --num-routed-experts 3 \
    --num-shared-experts 1 \
    --expert-scale 0.5
