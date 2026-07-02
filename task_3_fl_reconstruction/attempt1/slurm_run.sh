#!/bin/bash
#SBATCH --account=training2625
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=t3_recon_${USER}
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err

# Task 3 reconstruction job.
#   Analytic-only is CPU/ms — run it on the login node.
#   Use this GPU job for --optimize on hard MLP models (e.g. 1, 4, 5, 8).
#
# Submit:  sbatch slurm_run.sh
set -euo pipefail

# Do NOT `module load PyTorch` here: it conflicts with the task .venv torch
# (ImportError: loaded torch/_C folder instead of C extensions).
module purge

export TASK3_DATA_ROOT=/p/scratch/training2625/dougnon1/Loki/FL_Data_Reconstruction
cd /p/scratch/training2625/dougnon1/Loki/cispa_final/task_3_fl_reconstruction/attempt1
source "$TASK3_DATA_ROOT/.venv/bin/activate"

mkdir -p output

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Analytic pass over everything, then optimize the hard MLP models.
python run.py --out submission.pt
python run.py --base submission.pt --optimize --models 1 4 5 8 --steps 4000 --out submission.pt

python submit.py --check submission.pt
echo "done!"
