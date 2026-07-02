#!/bin/bash
#SBATCH --account=training2625
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --job-name=mlp_refine
#SBATCH --output=output/mlp_refine_%j.out
#SBATCH --error=output/mlp_refine_%j.err

# Gradient-matching refine for smooth-activation MLPs (models 1, 4).
#
# Example (refine 1 and 4 on top of the current best base):
#   sbatch --export=ALL,MLP_MODELS="1 4",MLP_OUT=sub_mlp_refine.pt,MLP_STEPS=4000,MLP_BASE=submission_all_m3_5k.pt slurm_mlp_refine.sh
set -euo pipefail

module purge

export TASK3_DATA_ROOT=/p/scratch/training2625/dougnon1/Loki/FL_Data_Reconstruction
cd /p/scratch/training2625/dougnon1/Loki/cispa_final/task_3_fl_reconstruction/attempt1
source "$TASK3_DATA_ROOT/.venv/bin/activate"

mkdir -p output

MLP_MODELS="${MLP_MODELS:-1 4}"
MLP_OUT="${MLP_OUT:-sub_mlp_refine.pt}"
MLP_STEPS="${MLP_STEPS:-4000}"
MLP_BASE="${MLP_BASE:-submission_all_m3_5k.pt}"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "MLP refine: base=$MLP_BASE out=$MLP_OUT models=$MLP_MODELS steps=$MLP_STEPS"
python mlp_reconstruct.py --base "$MLP_BASE" --out "$MLP_OUT" \
  --models $MLP_MODELS --refine --steps "$MLP_STEPS"
python submit.py --check "$MLP_OUT"
echo "done -> $MLP_OUT"
