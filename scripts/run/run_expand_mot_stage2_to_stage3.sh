#!/usr/bin/env bash
set -e

# Expand a Stage 2 und-only MoT/VLM checkpoint into a Stage 3 MoT init checkpoint.
# Stage 3 training then runs with train/configs/stage3_mm_mot.yaml (use-mot=True).

export STAGE2_CKPT="${STAGE2_CKPT:-outputs/stage2_2_mmu_mot/ckpt/0020000}"
export STAGE3_INIT_CKPT="${STAGE3_INIT_CKPT:-outputs/stage3_init_mot}"

if compgen -G "${STAGE3_INIT_CKPT}/*.safetensors" > /dev/null; then
    echo "Stage 3 MoT init checkpoint already exists at ${STAGE3_INIT_CKPT}"
    echo "Skipping expansion."
    exit 0
fi

python scripts/expand_mot_stage2_to_stage3.py \
    --src "${STAGE2_CKPT}" \
    --dst "${STAGE3_INIT_CKPT}"
