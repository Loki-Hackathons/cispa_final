#!/bin/bash
#SBATCH --account=training2625
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=t1_base_${USER}
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Task 1 — key-free baseline (fast). Produces a valid submission early.
set -euo pipefail

module load GCC CUDA PyTorch torchvision

REPO_ROOT="${REPO_ROOT:-/p/home/jusers/${USER}/jureca/code/cispa_final}"
cd "$REPO_ROOT"
source .venv/bin/activate

cd task_1_text_watermark/melissa
mkdir -p logs outputs

python -m src.load_data --check
python -m src.baseline
python -m src.evaluate --pred outputs/baseline_val_pred.jsonl --split validation

echo "Baseline submission at outputs/baseline_submission.jsonl"
