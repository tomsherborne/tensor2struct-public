#!/usr/bin/env bash
#SBATCH -o /home/%u/t2s/%A.out
#SBATCH -e /home/%u/t2s/%A.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1  # use 1 GPU
#SBATCH --mem=16000  # memory in Mb
#SBATCH -t 20-00:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=8  # number of cpus to use - there are 32 on each node.
#SBATCH --partition=apollo
##SBATCH --partition=cdtgpucluster
##SBATCH --exclude=charles[11-19]

set -e # fail fast

# Activate Conda
source /home/${USER}/miniconda3/bin/activate tensor2struct

echo "I'm running on ${SLURM_JOB_NODELIST}"
dt=$(date '+%d_%m_%y_%H:%M');
echo ${dt}

# Env variables
export MKL_THREADING_LAYER=GNU
export STUDENT_ID=${USER}
export SCRATCH_HOME="/disk/scratch/${STUDENT_ID}"
export CLUSTER_HOME="/home/${STUDENT_ID}"
export EXP_ROOT="${CLUSTER_HOME}/sp/tensor2struct-public"
export SCRATCH_ROOT="${SCRATCH_HOME}/sp/tensor2struct-public"

# Disable internet based logging
wandb offline

# Replicate the local codebase on scratch (TODO)
# ROOTDIR=$(git rev-parse --show-toplevel)
# EXCLUDE_FILE=${ROOTDIR}/scripts/send_exclude.txt
# rm -rf ./**/.ipynb_checkpoints  # Delete unwanted files
# rm -rf ./**/__pycache__
# rsync -auzh --progress --exclude-from ${EXCLUDE_FILE} "${ROOTDIR}" "${SCRATCH_HOME}/sp/" # Copy t2s to scratch space

RUN_CONFIG="/home/s1833057/sp/tensor2struct-public/configs/spider/run_config/REPLICATION_EXPS/run_en_spider_mbert_dgmaml.jsonnet"

echo "${RUN_CONFIG}"

echo "TRAIN========"
cd ${EXP_ROOT}

# Execute training
python ${EXP_ROOT}/experiments/spider_dg/run.py meta_train ${RUN_CONFIG}

echo "TEST========"

# Prediction script
tensor2struct eval ${RUN_CONFIG}

echo "============"
echo "job finished"
