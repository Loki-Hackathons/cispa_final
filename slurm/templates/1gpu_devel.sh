#!/bin/bash
#SBATCH --account=training2557
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=task1_att1_${USER}
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Usage:
#   1. Copy to your task directory and edit SCRIPT/ARGS below
#   2. mkdir -p logs
#   3. sbatch run.sh
#   4. Log in slurm/submitted.log and docs/notes-communes.md

set -euo pipefail

module load GCC CUDA PyTorch torchvision

REPO_ROOT="${REPO_ROOT:-/p/home/jusers/${USER}/jureca/code/cispa_final}"
cd "$REPO_ROOT"
source .venv/bin/activate

SCRIPT="shared/smoke_test_gpu.py"
ARGS=""

echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | GPUs: 1"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

python -u "$SCRIPT" $ARGS
