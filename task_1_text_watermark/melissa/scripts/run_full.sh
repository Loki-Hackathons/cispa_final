#!/bin/bash
#SBATCH --account=training2625
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=t1_wml_${USER}
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Task 1 — Text Watermark Localization (Melissa): full pipeline on 1 GPU.
# KGW greenlists need a CUDA (Philox) generator, hence a GPU job.
set -euo pipefail

module load GCC CUDA PyTorch torchvision

REPO_ROOT="${REPO_ROOT:-/p/home/jusers/${USER}/jureca/code/cispa_final}"
cd "$REPO_ROOT"
source .venv/bin/activate

cd task_1_text_watermark/melissa
mkdir -p logs outputs

# Path to the dataset YAML with real watermark keys (edit to your scratch path).
export WML_WATERMARK_YAML="${WML_WATERMARK_YAML:-/p/scratch/training2625/ansart1/loki/watermark_keys.yaml}"

echo "== dataset check =="
python -m src.load_data --check

echo "== train fusion calibrator (logreg) =="
python -m src.train_calibrator --model logreg
python -m src.evaluate --pred outputs/val_pred.jsonl --split validation

echo "== generate test submission =="
python -m src.predict --model outputs/calibrator_logreg.pkl

echo "Done. Submit with:"
echo "  python ../../shared/submit.py task_1_text_watermark/melissa/outputs/submission.jsonl --task-id 30-watermark-localization --action submit"
