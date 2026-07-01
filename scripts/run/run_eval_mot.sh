# Use provided paths or defaults to current directory
export ASSETS_BASE="${ASSETS_BASE:-public_assets}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage3_mm_mot}"

for iter in 0 100 ; do

    ITER_ID=$(printf "%07d" $iter)
    task_id="${TASK_ID:-$(date +%Y%m%d-%H-%M-%S)}"
    CKPT=${OUTPUT_DIR}/ckpt/${ITER_ID}

    EXP=${CKPT} \
    TASK_ID=${task_id} \
    LOG_DIR=${OUTPUT_DIR}/eval_logs/${ITER_ID}/${task_id} \
    CKPT_DIR=${CKPT} \
    SAMPLE_OUT=${OUTPUT_DIR}/eval_outputs/${ITER_ID} \
    CONFIG=evaluation/configs/mot.yaml \
    bash scripts/eval/eval_arc_c.sh

done
