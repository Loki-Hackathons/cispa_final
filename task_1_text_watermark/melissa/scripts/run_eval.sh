#!/bin/bash
#SBATCH --account=training2625
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=task1_melissa_eval
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Train calibrator, eval on val, produce outputs/submission.jsonl.
# No submission to leaderboard — same pattern as alexandre/run_hmm_submit.sh.
#
# Usage (depuis task_1_text_watermark/melissa/) :
#   mkdir -p logs outputs
#   sbatch scripts/run_eval.sh

set -euo pipefail

DATA="${DATA_DIR:-/p/scratch/training2625/dougnon1/Loki/watermark_localization}"
REPO="${REPO_ROOT:-/p/scratch/training2625/dougnon1/Loki/cispa_final}"
SCRATCH_VENV="$DATA/.venv"
MODEL="gboost"

echo "Job ${SLURM_JOB_ID:-local} | $(date)"

module load GCC CUDA PyTorch torchvision 2>/dev/null || true

if [ -f "$SCRATCH_VENV/bin/activate" ]; then
  source "$SCRATCH_VENV/bin/activate"
fi

# Use the module's CUDA-enabled PyTorch, not the venv's CPU-only build.
# (Same pattern as alexandre: torch/torchvision come from `module load`, not pip.)
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

export WML_DATASET_DIR="$DATA"
export WML_WATERMARK_YAML="$DATA/watermark_config.yaml"
export PYTHONPATH="$REPO:$PYTHONPATH"

# No outbound network on JURECA compute nodes: force HF offline (harmless — the correct
# pipeline uses no HF model, only the pinned vendor detector repos).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# The correct signals come from the pinned vendor repos (textseal / lm-watermarking /
# unigram-watermark). They are git submodules and must be synced on a LOGIN node first
# (compute nodes have no network):
#     bash scripts/task1/sync_watermark_repos.sh
VENDOR_CORE="$REPO/task_1_text_watermark/vendor/textseal/textseal/watermarking/core.py"
if [ ! -f "$VENDOR_CORE" ]; then
  echo "ERROR: vendor detector repos missing ($VENDOR_CORE)." >&2
  echo "       Run on a LOGIN node first: bash scripts/task1/sync_watermark_repos.sh" >&2
  exit 1
fi

cd "$REPO/task_1_text_watermark/melissa"
mkdir -p logs outputs

# One-time cleanup when the feature pipeline changes: a stale calibrator trained on the
# old feature layout would crash predict, and stale partial scores would be reused by the
# resume logic. The sentinel makes this happen exactly once after an upgrade.
SENTINEL="outputs/.pipeline_vendor_v1"
if [ ! -f "$SENTINEL" ]; then
  echo "Pipeline upgraded (vendor signals) — clearing stale calibrator + partial scores."
  rm -f outputs/calibrator_*.pkl outputs/submission.partial.jsonl
  touch "$SENTINEL"
fi

echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'CPU')"

CALIB="outputs/calibrator_${MODEL}.pkl"
if [ -f "$CALIB" ]; then
  echo "[1/3] Calibrator already exists ($CALIB) — skipping training."
  echo "[2/3] Skipping validation eval (calibrator unchanged)."
else
  # Fresh model → drop any partial predictions from a previous model.
  rm -f outputs/submission.partial.jsonl
  echo "[1/3] Train calibrator ($MODEL) ..."
  python -u -m src.train_calibrator --model "$MODEL"

  echo "[2/3] Evaluate on validation ..."
  python -u -m src.evaluate --pred outputs/val_pred.jsonl --split validation
fi

echo "[3/3] Generate test submission file ..."
python -u -m src.predict --model "$CALIB"

echo ""
echo "=== Done — outputs/submission.jsonl ready for submission ==="
