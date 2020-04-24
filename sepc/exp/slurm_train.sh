#!/usr/bin/env bash
set -x
now=$(date +"%Y-%m-%d_%H:%M:%S")
PARTITION=$1
all=$2
JOB_NAME=sepc
CONFIG=$3
GPUS_PER_NODE=$(($all<8?$all:8))
work_dir=$4
srun --partition=${PARTITION} -n${all} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --mpi=pmi2 \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    python -u ../tools/train.py ${CONFIG} --work_dir $work_dir  --validate   --launcher="slurm"
