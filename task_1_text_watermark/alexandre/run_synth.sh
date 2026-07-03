#!/bin/bash
#SBATCH --account=training2625
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --job-name=task1_synth_ansart1
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -euo pipefail
module load GCC CUDA PyTorch torchvision
export HF_HOME="${HF_HOME:-/p/scratch/training2625/ansart1/loki/hf_cache}"
REPO_ROOT="${REPO_ROOT:-/p/home/jusers/${USER}/jureca/code/cispa_final}"
cd "$REPO_ROOT/task_1_text_watermark/alexandre"
mkdir -p logs output

echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
python -u gen_synthetic.py --out-dir output --n-per-cell 30 \
    --temperature 0.9 --top-p 0.95
# Move outputs to standard names expected by cv_smm.load_synthetic
mv -f output/synthetic_train.jsonl output/synthetic_train.jsonl
mv -f output/entropy_synth.npz output/entropy_synth.npz
