#!/bin/bash
#SBATCH --account=training2625
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=t3_arr
#SBATCH --output=output/arr_%A_%a.out
#SBATCH --error=output/arr_%A_%a.err
#SBATCH --array=1-12

# One array task per model. Small independent tasks get backfill-scheduled into
# GPU gaps, so we grab slots as they free instead of waiting for one big block.
# Each task writes output/parts/model{i}.pt; assemble with `python merge.py`.
#
# Submit:  sbatch slurm_array.sh
# Fewer/more concurrent: sbatch --array=1-12%4 slurm_array.sh
set -euo pipefail

# Do NOT `module load PyTorch` (it clobbers the task .venv torch C extensions).
module purge

export TASK3_DATA_ROOT=/p/scratch/training2625/dougnon1/Loki/FL_Data_Reconstruction
cd /p/scratch/training2625/dougnon1/Loki/cispa_final/task_3_fl_reconstruction/attempt1
source "$TASK3_DATA_ROOT/.venv/bin/activate"

mkdir -p output/parts

M="$SLURM_ARRAY_TASK_ID"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# --optimize only affects MLP (exact forward); CNN/ViT fall back to analytic
# inside run.py, so this is safe to run for every model. Optimizing CNN/ViT
# against guessed forwards regressed the leaderboard (0.1427 -> 0.0635).
python run.py --optimize --models "$M" --steps 4000 --save-part output/parts
echo "model $M done"
