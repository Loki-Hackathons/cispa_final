#!/bin/bash
#SBATCH --account=training2557
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --job-name=task2_att1_${USER}
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Usage: copy to task dir, set SCRIPT and ARGS, then sbatch.

set -euo pipefail

module load GCC CUDA PyTorch torchvision

REPO_ROOT="${REPO_ROOT:-/p/home/jusers/${USER}/jureca/code/cispa_final}"
cd "$REPO_ROOT"
source .venv/bin/activate

SCRIPT="main.py"
ARGS=""

echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | GPUs: 4"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

python -u "$SCRIPT" $ARGS
