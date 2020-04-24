#!/usr/bin/env bash

set -x

set -x
now=$(date +"%Y-%m-%d_%H:%M:%S")
PARTITION=$1
JOB_NAME=sepc
CONFIG=$2
CHECKPOINT=$3
GPUS_PER_NODE=8
PY_ARGS=${@:5}
srun --partition=${PARTITION} -n8 \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --mpi=pmi2 \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    python -u ../tools/test.py ${CONFIG} ${CHECKPOINT} --eval bbox    --out out.pkl   --launcher="slurm" ${PY_ARGS}
