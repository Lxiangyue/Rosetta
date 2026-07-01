#!/usr/bin/env bash
# Launch a Rosetta worker script on one or more nodes from the chief node.
#
# Usage:
#   hostfile=hosts.txt HOST_NUM=4 HOST_GPU_NUM=8 \
#       bash launch/run_multinode.sh scripts/run/run_stage2_1_projector_moe.sh
#
# The hostfile should contain one node IP/hostname per line. The first selected
# node is used as CHIEF_IP. This script starts one worker per node via ssh; each
# worker then starts torchrun with HOST_GPU_NUM local processes.

SCRIPT_DIR=$(realpath "$(dirname "$0")")
source "${SCRIPT_DIR}/common_env.sh"
source "${SCRIPT_DIR}/common_tools.sh"

HOST_NUM=${HOST_NUM:-1}
HOST_GPU_NUM=${HOST_GPU_NUM:-8}
HOSTFILE=${hostfile:-${HOSTFILE:-}}
OFFSET=${OFFSET:-0}
TASK_ID=${TASK_ID:-$(date +%Y%m%d-%H-%M-%S)}
FREE_PORT=$(find_free_port)
_NAME="[main]"

if [ -z "${HOSTFILE}" ]; then
    print_msg error "$_NAME Please set hostfile=<path> or HOSTFILE=<path>." >&2
    exit 1
fi

if [ ! -f "${HOSTFILE}" ]; then
    print_msg error "$_NAME Hostfile not found: ${HOSTFILE}" >&2
    exit 1
fi

HOSTFILE_NUM=$(wc -l < "${HOSTFILE}")
if [ "${HOST_NUM}" -gt "${HOSTFILE_NUM}" ]; then
    print_msg error "$_NAME HOST_NUM (${HOST_NUM}) is greater than hostfile nodes (${HOSTFILE_NUM})." >&2
    exit 1
fi

if [ $# -lt 1 ]; then
    print_msg error "Usage: $0 <script> [args...]"
    exit 1
fi

SCRIPT=$(realpath "$1"); shift
if [ ! -f "${SCRIPT}" ]; then
    print_msg error "Script not found: ${SCRIPT}"
    exit 1
fi

shell_quote() {
    printf "%q" "$1"
}

mapfile -t NODE_IPS < <(tail -n +$((OFFSET + 1)) "${HOSTFILE}" | head -n "${HOST_NUM}" | awk '{print $1}')
CHIEF_IP=${CHIEF_IP:-${NODE_IPS[0]}}

print_msg "$_NAME Launcher : ssh multi-node"
print_msg "$_NAME Hostfile : ${HOSTFILE}"
print_msg "$_NAME Nodes    : ${NODE_IPS[*]}"
print_msg "$_NAME Script   : ${SCRIPT}"
print_msg "$_NAME GPUs/node: ${HOST_GPU_NUM}"
print_msg "$_NAME Task ID  : ${TASK_ID}"
print_msg "$_NAME Chief IP : ${CHIEF_IP}"
print_msg "$_NAME Port     : ${FREE_PORT}"
echo

PROJECT_BASE=$(dirname "${SCRIPT_DIR}")

pids=()
for idx in "${!NODE_IPS[@]}"; do
    node="${NODE_IPS[$idx]}"
    remote_env="INDEX=$(shell_quote "${idx}") HOST_NUM=$(shell_quote "${HOST_NUM}") HOST_GPU_NUM=$(shell_quote "${HOST_GPU_NUM}") CHIEF_IP=$(shell_quote "${CHIEF_IP}") FREE_PORT=$(shell_quote "${FREE_PORT}") TASK_ID=$(shell_quote "${TASK_ID}")"
    for var in BASE_PATH ASSETS_BASE OUTPUT_PATH CKPT_DIR OUTPUT_DIR SRC_CKPT DST_CKPT MAX_SEQ_LEN DATA_PATH; do
        if [ -n "${!var+x}" ]; then
            remote_env="${remote_env} ${var}=$(shell_quote "${!var}")"
        fi
    done

    remote_args=""
    for arg in "$@"; do
        remote_args="${remote_args} $(shell_quote "${arg}")"
    done
    remote_cmd="cd $(shell_quote "${PROJECT_BASE}") && unset MASTER_ADDR RDZV_ENDPOINT && ${remote_env} bash $(shell_quote "${SCRIPT}")${remote_args}"

    print_msg "$_NAME Launching node ${idx}/${HOST_NUM} on ${node}"
    ssh "${node}" "${remote_cmd}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

exit "${status}"
