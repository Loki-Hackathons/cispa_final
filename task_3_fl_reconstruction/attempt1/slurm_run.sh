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
#   Use this GPU job for --optimize on hard models (e.g. ViT 9, 11).
#
# Submit:  sbatch slurm_run.sh
set -euo pipefail

module load GCC CUDA PyTorch torchvision

cd /p/scratch/training2625/dougnon1/Loki/FL_Data_Reconstruction
source .venv/bin/activate

mkdir -p output

# Analytic pass over everything, then optimize the hard MLP-style models.
python run.py --out submission.pt
python run.py --base submission.pt --optimize --models 1 4 --steps 4000 --out submission.pt

python submit.py --check submission.pt
echo "done!"
