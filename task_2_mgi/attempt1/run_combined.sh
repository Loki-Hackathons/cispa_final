#!/bin/bash
#SBATCH --account=training2625
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --job-name=task2_combined_${USER}
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# End-to-end Task 2 submission builder (correct per-cell pipeline):
#   1. reconstruct_to_g   -> M_G, N_G blocks   (Stage 1, easy)
#   2. attack_combined    -> M_N, G_M, N_M, G_N blocks (Stage 1 x Stage 2)
#      (gated by --selftest; hard ->M/->N kept as swap fallback in assemble)
#   3. build_submission_v2 -> output/submission.npz
#   4. evaluate_submission -> proxy flip-rate / MSE / estimated score
# Then submit from a login node (see README). Does NOT submit by itself.

set -euo pipefail

LOKI_ROOT="${LOKI_ROOT:-/p/scratch/training2625/dougnon1/Loki}"
REPO_ROOT="${REPO_ROOT:-${LOKI_ROOT}/cispa_final}"
export ONED_TOKENIZER_ROOT="${ONED_TOKENIZER_ROOT:-${LOKI_ROOT}/1d-tokenizer}"
export MGI_DATA_DIR="${MGI_DATA_DIR:-${LOKI_ROOT}/MGI/data}"
export MGI_OUTPUT_DIR="${MGI_OUTPUT_DIR:-${REPO_ROOT}/task_2_mgi/attempt1/output}"
export PYTHONPATH="${ONED_TOKENIZER_ROOT}:${REPO_ROOT}/task_2_mgi/attempt1:${REPO_ROOT}/shared:${PYTHONPATH:-}"

cd "${REPO_ROOT}"
mkdir -p task_2_mgi/attempt1/output/blocks logs

source "${LOKI_ROOT}/MGI/.venv/bin/activate"
python -m pip install -q einops omegaconf 2>/dev/null || true

echo "Job ${SLURM_JOB_ID:-local} on ${SLURM_NODELIST:-local}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || true

A="task_2_mgi/attempt1"
BLK="${MGI_OUTPUT_DIR}/blocks"

echo "=== [1/4] reconstruct_to_g (M->G, N->G) ==="
python -u "$A/reconstruct_to_g.py" --classes M N

echo "=== [2/4] attack_combined selftest (must PASS) ==="
python -u "$A/attack_combined.py" --directions N_M --selftest --limit 8

echo "=== [2/4] attack_combined (M_N, G_M, N_M, G_N) ==="
python -u "$A/attack_combined.py" --directions M_N,G_M,N_M,G_N --steps 300 --batch-size 4

echo "=== [3/4] assemble submission (swap fallback for the two hard dirs) ==="
python -u "$A/build_submission_v2.py" \
    --dir M_G=block:"$BLK/M_G_recon.npy" \
    --dir N_G=block:"$BLK/N_G_recon.npy" \
    --dir M_N=block:"$BLK/M_N_combined.npy" \
    --dir G_M=block:"$BLK/G_M_combined.npy" \
    --dir N_M=swap \
    --dir G_N=swap

echo "=== [4/4] evaluate submission against proxy DCB ==="
python -u "$A/evaluate_submission.py" "${MGI_OUTPUT_DIR}/submission.npz" \
    --json-out "${MGI_OUTPUT_DIR}/proxy_eval.json"

echo ""
echo "Submission: ${MGI_OUTPUT_DIR}/submission.npz"
ls -lh "${MGI_OUTPUT_DIR}/submission.npz" 2>/dev/null || true
echo "Proxy eval: ${MGI_OUTPUT_DIR}/proxy_eval.json"
echo "If N_M/G_N combined blocks beat swap in proxy_eval, re-run build_submission_v2"
echo "swapping --dir N_M=block:$BLK/N_M_combined.npy --dir G_N=block:$BLK/G_N_combined.npy"
