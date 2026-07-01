#!/usr/bin/env bash
set -e

# Stage 0: Upcycle Qwen3-0.6B-Base (dense) → MoT precursor (7 routed + 1 shared)
# This is the und-only checkpoint used before Stage 3 expands it to MoT und/gen streams.
export ASSETS_BASE="${ASSETS_BASE:-public_assets}"
export SRC_CKPT="${SRC_CKPT:-checkpoints/Qwen3-0.6B-Base}"
export DST_CKPT="${DST_CKPT:-outputs/Qwen3-0.6B-Base-upcycling-mot-7e-scale05}"

if [ -d "${DST_CKPT}" ]; then
    echo "MoT checkpoint already exists at ${DST_CKPT}"
    echo "Skipping conversion."
    exit 0
fi

python scripts/convert_qwen3_dense_to_moe.py \
    --src "${SRC_CKPT}" \
    --dst "${DST_CKPT}" \
    --num-routed-experts 7 \
    --num-shared-experts 1 \
    --expert-scale 0.5
