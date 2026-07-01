#!/usr/bin/env bash
set -e

# Expand Stage 1 checkpoint (3 experts) to Stage 2 init checkpoint (12 experts)
# Maps: 0-2 → text (0-2), vit (3-5), vae (6-11)

export STAGE1_CKPT="${STAGE1_CKPT:-outputs/stage1_lm/ckpt/0035000}"
export STAGE2_INIT_CKPT="${STAGE2_INIT_CKPT:-outputs/stage2_init}"

if compgen -G "${STAGE2_INIT_CKPT}/*.safetensors" > /dev/null; then
    echo "Stage 2 init checkpoint already exists at ${STAGE2_INIT_CKPT}"
    echo "Skipping expansion."
    exit 0
fi

python scripts/expand_ours_stage1_to_stage2.py \
    --src "${STAGE1_CKPT}" \
    --dst "${STAGE2_INIT_CKPT}"
