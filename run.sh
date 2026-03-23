#!/bin/bash

#SBATCH --job-name=PCDARTS
#SBATCH --output=slurm/slurm-%x-%A_%a.out
#SBATCH --time=5-00:10:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -p gpu-best
#SBATCH --exclude=margpu018,margpu021
#SBATCH --array=[1-5]

STAGGER_SECONDS=5
SLEEP_TIME=$(( (SLURM_ARRAY_TASK_ID - 1) * STAGGER_SECONDS ))
echo "Array task ${SLURM_ARRAY_TASK_ID}: sleeping ${SLEEP_TIME}s before starting"
sleep "${SLEEP_TIME}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi

if [ "${SLURM_ARRAY_JOB_ID}" ] ; then
    JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
    JOB_ID="${SLURM_JOB_ID}"
fi
echo "JOB ID = ${JOB_ID}"
echo "NODE NAME = ${SLURMD_NODENAME}"

echo "DATASET = ${NAS_DATASET}"

python custom_train_search.py --dataset "${NAS_DATASET}"
