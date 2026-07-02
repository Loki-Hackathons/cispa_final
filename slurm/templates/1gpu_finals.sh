#!/bin/bash
#SBATCH --account=training2625
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=task_att_${USER}
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# CISPA Grand Finals template — edit SCRIPT/ARGS, copy to task dir.
# See docs/Hackathon_Setup Finale.md

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
