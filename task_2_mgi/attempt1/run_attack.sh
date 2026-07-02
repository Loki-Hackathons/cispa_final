#!/bin/bash
#SBATCH --account=training2625
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=task2_att1_${USER}
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -euo pipefail

LOKI_ROOT="${LOKI_ROOT:-/p/scratch/training2625/dougnon1/Loki}"
REPO_ROOT="${REPO_ROOT:-${LOKI_ROOT}/cispa_final}"
export ONED_TOKENIZER_ROOT="${ONED_TOKENIZER_ROOT:-${LOKI_ROOT}/1d-tokenizer}"
export MGI_DATA_DIR="${MGI_DATA_DIR:-${LOKI_ROOT}/MGI/data}"
export MGI_OUTPUT_DIR="${MGI_OUTPUT_DIR:-${REPO_ROOT}/task_2_mgi/attempt1/output}"
export PYTHONPATH="${ONED_TOKENIZER_ROOT}:${REPO_ROOT}/shared:${PYTHONPATH:-}"

cd "${REPO_ROOT}"
mkdir -p task_2_mgi/attempt1/output logs

source "${LOKI_ROOT}/MGI/.venv/bin/activate"

# 1d-tokenizer deps (idempotent; use venv pip, not system module pip)
python -m pip install -q einops omegaconf 2>/dev/null || true

echo "Job ID: ${SLURM_JOB_ID:-local} | Node: ${SLURM_NODELIST:-local} | GPU:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || true
echo "ONED_TOKENIZER_ROOT=${ONED_TOKENIZER_ROOT}"
echo "MGI_DATA_DIR=${MGI_DATA_DIR}"
echo ""

python -u task_2_mgi/attempt1/run_attack.py --phase all "$@"

echo ""
echo "Submission: ${MGI_OUTPUT_DIR}/submission.npz"
ls -lh "${MGI_OUTPUT_DIR}/submission.npz" 2>/dev/null || true
