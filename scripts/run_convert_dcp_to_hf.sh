#!/usr/bin/env bash
# Convert the FSDP/DCP checkpoint to HuggingFace safetensors format.
# Run with a single GPU – conversion does not need multiple GPUs.
#
# Usage:
#   bash scripts/run_convert_dcp_to_hf.sh
#
# You can also override the checkpoint/output paths via environment variables:
#   CKPT_DIR=<dcp_dir> OUTPUT_DIR=<hf_dir> bash scripts/run_convert_dcp_to_hf.sh

set -euo pipefail
SCRIPT_DIR=$(realpath "$(dirname "$0")")
REPO_DIR=$(realpath "${SCRIPT_DIR}/..")

# ── Paths ───────────────────────────────────────────────────────────────────
EXP="${EXP:-checkpoints/Rosetta-3.8B-A1B}"
ITER="${ITER:-400000}"

ITER_PADDED=$(printf "%07d" "${ITER}")

CKPT_DIR="${CKPT_DIR:-${EXP}/ckpt/iter_${ITER_PADDED}/weights}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXP}/ckpt/iter_${ITER_PADDED}/hf_weights}"

echo "=================================================="
echo "Input  (DCP) : ${CKPT_DIR}"
echo "Output (HF)  : ${OUTPUT_DIR}"
echo "=================================================="

if [ ! -d "${CKPT_DIR}" ]; then
    echo "ERROR: DCP checkpoint directory not found: ${CKPT_DIR}"
    exit 1
fi

if [ -f "${OUTPUT_DIR}/model.safetensors.index.json" ]; then
    echo "HF checkpoint already exists at ${OUTPUT_DIR}."
    echo "Delete it first if you want to re-convert."
    exit 0
fi

mkdir -p "${OUTPUT_DIR}"

cd "${REPO_DIR}"

# Single GPU is enough for the conversion.
torchrun \
    --standalone \
    --nproc_per_node=1 \
    scripts/convert_dcp_to_hf.py \
    --ckpt    "${CKPT_DIR}" \
    --output  "${OUTPUT_DIR}" \
    --device  cuda \
    --shard-size-gb 5

echo ""
echo "Conversion done.  Files in ${OUTPUT_DIR}:"
ls -lh "${OUTPUT_DIR}"
