#!/usr/bin/env bash
# Runs a Rosetta torchrun training job on a single node.
# Normally invoked by launch/run_multinode.sh, but can also be called directly for 1-node runs.
#
# Usage (single node, 8 GPUs):
#   INDEX=0 HOST_NUM=1 HOST_GPU_NUM=8 FREE_PORT=23456 \
#       bash launch/workers/run_train.sh torchrun \
#       train/configs/stage3_mm.yaml \
#       --ckpt-dir checkpoints/stage2_vlm \
#       --output-dir outputs/stage3_mm \
#       [extra args...]
#
# For multi-node training, invoke this worker through launch/run_multinode.sh from the
# chief node. See TRAIN.md for hostfile usage.

LAUNCH_DIR=$(realpath "$(dirname "$(dirname "$0")")")
source "${LAUNCH_DIR}/common_env.sh"
source "${LAUNCH_DIR}/common_tools.sh"
set -o pipefail

INDEX=${INDEX:-0}
HOST_NUM=${HOST_NUM:-1}
HOST_GPU_NUM=${HOST_GPU_NUM:-8}
_NAME="[NODE ${INDEX}]"

PROJECT_BASE=$(dirname "${LAUNCH_DIR}")
print_msg "$_NAME PROJECT_BASE: ${PROJECT_BASE}"
cd "${PROJECT_BASE}" || exit 1

export PYTHONPATH="${PROJECT_BASE}:${PYTHONPATH}"
if [ -d "${PROJECT_BASE}/deps" ]; then
    while IFS= read -r -d '' subdir; do
        export PYTHONPATH="$(realpath "${subdir}"):${PYTHONPATH}"
        print_msg "$_NAME Added to PYTHONPATH: ${subdir}"
    done < <(find "${PROJECT_BASE}/deps" -mindepth 1 -maxdepth 1 -type d -print0)
fi

export ASSETS_BASE="${ASSETS_BASE:-${PROJECT_BASE}/public_assets}"
print_msg "$_NAME ASSETS_BASE: ${ASSETS_BASE}"

STARTUP_METHOD="${1:-torchrun}"
if [ "${STARTUP_METHOD}" == "torchrun" ] || [ "${STARTUP_METHOD}" == "python" ]; then
    shift
else
    STARTUP_METHOD=torchrun
fi

CONFIG_FILE="$1"
if [ -z "${CONFIG_FILE}" ]; then
    print_msg error "$_NAME Config file must be specified" >&2
    exit 1
fi
CONFIG_FILE=$(realpath "${CONFIG_FILE}")
if [ ! -f "${CONFIG_FILE}" ]; then
    print_msg error "$_NAME Config file not found: ${CONFIG_FILE}" >&2
    exit 1
fi
print_msg "$_NAME CONFIG: ${CONFIG_FILE}"
shift

TASK_ID="${TASK_ID:-$(date +%Y%m%d-%H-%M-%S)}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_BASE}/outputs}"
LOG_DIR="${OUTPUT_PATH}/logs/${TASK_ID}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${INDEX}.log"
print_msg "$_NAME LOG_FILE: ${LOG_FILE}"

MASTER_ADDR="${CHIEF_IP:-127.0.0.1}"
FREE_PORT="${FREE_PORT:-$(find_free_port)}"

case "${STARTUP_METHOD}" in
    torchrun)
        set -x
        if [ -n "${RDZV_ENDPOINT}" ]; then
            # Multi-node: all nodes run the same command; torchrun auto-assigns ranks.
            torchrun \
                --nproc-per-node "${HOST_GPU_NUM}" \
                --nnodes "${HOST_NUM}" \
                --rdzv-backend c10d \
                --rdzv-endpoint "${RDZV_ENDPOINT}" \
                -m train.pretrain \
                "${CONFIG_FILE}" \
                --task-id "${TASK_ID}" \
                "$@" 2>&1 | tee "${LOG_FILE}"
        else
            # Single-node, or multi-node when launched by launch/run_multinode.sh.
            torchrun \
                --nproc-per-node "${HOST_GPU_NUM}" \
                --nnodes "${HOST_NUM}" \
                --node-rank "${INDEX}" \
                --master-addr "${MASTER_ADDR}" \
                --master-port "${FREE_PORT}" \
                -m train.pretrain \
                "${CONFIG_FILE}" \
                --task-id "${TASK_ID}" \
                "$@" 2>&1 | tee "${LOG_FILE}"
        fi
        ;;
    python)
        set -x
        python3 -m train.pretrain \
            "${CONFIG_FILE}" \
            --task-id "${TASK_ID}" \
            "$@" 2>&1 | tee "${LOG_FILE}"
        ;;
esac
