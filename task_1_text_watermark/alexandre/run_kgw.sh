#!/bin/bash
#SBATCH --account=training2625
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=task1_att1_ansart1
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Task 1: precompute KGW greenlist masks with CUDA Philox (probe vocab size first).

set -euo pipefail

module load GCC CUDA PyTorch torchvision

REPO_ROOT="${REPO_ROOT:-/p/home/jusers/${USER}/jureca/code/cispa_final}"
DATA_DIR="${DATA_DIR:-/p/scratch/training2625/ansart1/loki/watermark_localization}"
cd "$REPO_ROOT/task_1_text_watermark/alexandre"
mkdir -p logs output

echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

python -u kgw_scores.py --data-dir "$DATA_DIR" --out-dir output --auto \
    --splits train validation test
