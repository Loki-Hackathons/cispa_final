#!/bin/bash
#SBATCH --account=training2625
#SBATCH --partition=dc-gpu-devel
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --job-name=task1_hmm_submit
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Fit HMM with precomputed KGW masks, eval on val, submit to leaderboard.

set -eo pipefail

module load GCC CUDA PyTorch torchvision

REPO="${REPO_ROOT:-/p/home/jusers/${USER}/jureca/code/cispa_final}"
DATA="${DATA_DIR:-/p/scratch/training2625/ansart1/loki/watermark_localization}"
OUT="$REPO/task_1_text_watermark/alexandre/output"
SCRATCH_VENV="$DATA/.venv"

cd "$REPO/task_1_text_watermark/alexandre"
mkdir -p logs output

if [ -f "$SCRATCH_VENV/bin/activate" ]; then
  source "$SCRATCH_VENV/bin/activate"
fi

pip install -q scikit-learn scipy numpy requests 2>/dev/null || true
export PYTHONPATH="$REPO/task_1_text_watermark/alexandre:$REPO"

cd /tmp
python -c "import torch, numpy, sklearn; print('deps ok', torch.__version__)"

python -u "$REPO/task_1_text_watermark/alexandre/run_hmm.py" \
  --data-dir "$DATA" \
  --kgw-dir "$OUT" \
  --out-dir "$OUT" \
  --splits validation test

python -u "$REPO/shared/task1_eval.py" \
  --dataset "$DATA/validation.jsonl" \
  --predictions "$OUT/validation_scores.jsonl" \
  --method "HMM forward-backward: TextSeal+GumbelMax+Unigram+KGW PRF LLRs" \
  --note "KGW CUDA Philox greenlists job 15399747" \
  --out hmm_kgw_v1

cd "$REPO"
set -a && source .env && set +a
python -u shared/submit.py \
  "$OUT/submission.jsonl" \
  --task-id 30-watermark-localization \
  --action submit \
  --owner ansart1 \
  --method "HMM forward-backward: TextSeal+GumbelMax+Unigram+KGW PRF LLRs" \
  --note "KGW CUDA Philox greenlists job 15399747"

echo "=== pipeline complete ==="
