#!/bin/bash
# One-shot API submit for an existing submission.jsonl (login node, no GPU).

set -eo pipefail

REPO="${REPO_ROOT:-/p/home/jusers/${USER}/jureca/code/cispa_final}"
OUT="$REPO/task_1_text_watermark/alexandre/output/submission.jsonl"

cd "$REPO"
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

module load GCC 2>/dev/null || true
if [ -f "$REPO/.venv/bin/activate" ]; then source "$REPO/.venv/bin/activate"; fi
pip install -q requests 2>/dev/null || true

python -u shared/submit.py "$OUT" \
  --task-id 30-watermark-localization \
  --action submit \
  --owner ansart1 \
  --method "HMM forward-backward: TextSeal+GumbelMax+Unigram+KGW PRF LLRs" \
  --note "KGW CUDA Philox job 15399747; HMM job 15399857"
