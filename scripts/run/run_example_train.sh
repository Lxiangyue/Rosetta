#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(realpath "$(dirname "$0")")
REPO_DIR=$(realpath "${SCRIPT_DIR}/../..")
cd "${REPO_DIR}"

export BASE_PATH="${BASE_PATH:-.}"

export ASSETS_BASE="${BASE_PATH}/public_assets"
export HOST_GPU_NUM="${HOST_GPU_NUM:-8}"

DEMO_OUT="${DEMO_OUT:-${BASE_PATH}/outputs/example_train}"
mkdir -p "${DEMO_OUT}"

export DEMO_SEED="${DEMO_SEED:-42}"

ROSETTA_CKPT="${BASE_PATH}/checkpoints/Rosetta-3.8B-A1B-stage2-lm-mmu/hf_weights"
MOE_CKPT="${BASE_PATH}/checkpoints/MoE-3.8B-A1B-stage2-lm-mmu/hf_weights"
MOT_CKPT="${BASE_PATH}/checkpoints/MoT-4.5B-A1B-stage3-init/hf_weights"

check_path() {
    local path="$1"
    local hint="$2"
    if [ ! -e "${path}" ]; then
        echo "[run_example_train] Missing: ${path}"
        echo "[run_example_train] ${hint}"
        exit 1
    fi
}

check_path "${ASSETS_BASE}" "Download: hf download tencent/Rosetta-inference public_assets.zip --local-dir . && unzip -o public_assets.zip && rm public_assets.zip"
check_path "example_data" "Download: bash scripts/download_example_data.sh"
check_path "${ROSETTA_CKPT}" "Download: hf download tencent/Rosetta-inference --include 'checkpoints/Rosetta-3.8B-A1B-stage2-lm-mmu/**' --local-dir ."
check_path "${MOE_CKPT}" "Download: hf download tencent/Rosetta-inference --include 'checkpoints/MoE-3.8B-A1B-stage2-lm-mmu/**' --local-dir ."
check_path "${MOT_CKPT}" "Download: hf download tencent/Rosetta-inference --include 'checkpoints/MoT-4.5B-A1B-stage3-init/**' --local-dir ."


echo "[run_example_train] Running Rosetta 100-step training + ARC evaluation"
CKPT_DIR="${ROSETTA_CKPT}" \
OUTPUT_DIR="${DEMO_OUT}/rosetta" \
bash scripts/run/run_train.sh --no-save-optimizer --reproduce --seed "${DEMO_SEED}" "$@"
OUTPUT_DIR="${DEMO_OUT}/rosetta" ASSETS_BASE="${ASSETS_BASE}" bash scripts/run/run_eval.sh

echo "[run_example_train] Running MoE baseline 100-step training + ARC evaluation"
CKPT_DIR="${MOE_CKPT}" \
OUTPUT_DIR="${DEMO_OUT}/moe" \
bash scripts/run/run_train_moe.sh --no-save-optimizer --reproduce --seed "${DEMO_SEED}" "$@"
OUTPUT_DIR="${DEMO_OUT}/moe" ASSETS_BASE="${ASSETS_BASE}" bash scripts/run/run_eval_moe.sh

echo "[run_example_train] Running MoT baseline 100-step training + ARC evaluation"
CKPT_DIR="${MOT_CKPT}" \
OUTPUT_DIR="${DEMO_OUT}/mot" \
bash scripts/run/run_train_mot.sh --no-save-optimizer --reproduce --seed "${DEMO_SEED}" "$@"
OUTPUT_DIR="${DEMO_OUT}/mot" ASSETS_BASE="${ASSETS_BASE}" bash scripts/run/run_eval_mot.sh

python3 scripts/plot_scores.py --run-dir "${DEMO_OUT}" --out-dir "${DEMO_OUT}"

echo "[run_example_train] Done. Open ${DEMO_OUT}/arc_step0_step100.png to view the result."
