#!/bin/bash
#SBATCH --account=training2625
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=task1_calib_ansart1
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Task 1, point 1: temperature/top-p calibration grid for the exact
# Gumbel-Max/TextSeal LLR (docs/task1/attempt1.md 19.4). Labeled splits only
# (train+validation, 180 docs) - cheap grid search before committing to one
# combo for the 1320-doc test split.

set -euo pipefail

module load GCC CUDA PyTorch torchvision
export HF_HOME="${HF_HOME:-/p/scratch/training2625/ansart1/loki/hf_cache}"

REPO_ROOT="${REPO_ROOT:-/p/home/jusers/${USER}/jureca/code/cispa_final}"
DATA_DIR="${DATA_DIR:-/p/scratch/training2625/ansart1/loki/watermark_localization}"
cd "$REPO_ROOT/task_1_text_watermark/alexandre"
mkdir -p logs output

echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

python -u calib_pass.py --data-dir "$DATA_DIR" --out-dir output \
    --splits train validation --temps 0.8 0.9 1.0 --top-ps 0.9 0.95 1.0
