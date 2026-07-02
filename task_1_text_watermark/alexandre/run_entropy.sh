#!/bin/bash
#SBATCH --account=training2625
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=task1_entropy_ansart1
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Task 1: 7B forward-pass signals over all splits - entropy, realized-token
# logp (exact Gumbel/TextSeal LLR), Unigram/KGW boosted green-mass.
# Produces output/{entropy,logp,unigram_lpg,kgw_lpg}_{train,validation,test}.npz.
# Requires the 7B (--require-primary): logp/green-mass need the real
# generator, not the 0.5B proxy.

set -euo pipefail

module load GCC CUDA PyTorch torchvision
export HF_HOME="${HF_HOME:-/p/scratch/training2625/ansart1/loki/hf_cache}"

REPO_ROOT="${REPO_ROOT:-/p/home/jusers/${USER}/jureca/code/cispa_final}"
DATA_DIR="${DATA_DIR:-/p/scratch/training2625/ansart1/loki/watermark_localization}"
cd "$REPO_ROOT/task_1_text_watermark/alexandre"
mkdir -p logs output

echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

python -u entropy_pass.py --data-dir "$DATA_DIR" --out-dir output \
    --splits train validation test --require-primary
