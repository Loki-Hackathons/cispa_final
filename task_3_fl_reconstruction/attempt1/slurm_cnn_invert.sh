#!/bin/bash
#SBATCH --account=training2625
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=cnn_inv
#SBATCH --output=output/cnninv_%j.out
#SBATCH --error=output/cnninv_%j.err

# CNN feature-map inversion on GPU (A100-friendly).
#
# Quick test (models 2 and 12 only):
#   sbatch --export=ALL,CNN_MODELS="2 12",CNN_OUT=submission_cnninv_2_12.pt slurm_cnn_invert.sh
#
# Full CNN pass (all 6 CNN models):
#   sbatch --export=ALL,CNN_MODELS="2 3 6 7 10 12",CNN_OUT=submission_cnninv_all.pt slurm_cnn_invert.sh
#
# Defaults if env vars unset: models 2 12, 2000 steps.
set -euo pipefail

module purge

export TASK3_DATA_ROOT=/p/scratch/training2625/dougnon1/Loki/FL_Data_Reconstruction
cd /p/scratch/training2625/dougnon1/Loki/cispa_final/task_3_fl_reconstruction/attempt1
source "$TASK3_DATA_ROOT/.venv/bin/activate"

mkdir -p output

CNN_MODELS="${CNN_MODELS:-2 12}"
CNN_OUT="${CNN_OUT:-submission_cnninv.pt}"
CNN_STEPS="${CNN_STEPS:-2000}"
CNN_BASE="${CNN_BASE:-submission_analytic.pt}"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if [ ! -f "$CNN_BASE" ]; then
  echo "Base $CNN_BASE missing; generating analytic baseline..."
  python run.py --out "$CNN_BASE"
fi

echo "CNN inversion: base=$CNN_BASE out=$CNN_OUT models=$CNN_MODELS steps=$CNN_STEPS"
python cnn_invert.py --base "$CNN_BASE" --out "$CNN_OUT" --models $CNN_MODELS --steps "$CNN_STEPS"
python submit.py --check "$CNN_OUT"
echo "done -> $CNN_OUT"
