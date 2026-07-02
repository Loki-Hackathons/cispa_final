#!/bin/bash
# Task 1 only — clone watermark_localization + venv + sync detector repos.
set -euo pipefail

LOG="/p/scratch/training2625/ansart1/task1_setup.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Task 1 setup START $(date) ==="

jutil env activate -p training2625 2>/dev/null || true

TEAM="/p/scratch/training2625/ansart1/loki"
DEST="$TEAM/watermark_localization"
REPO="/p/home/jusers/ansart1/jureca/code/cispa_final"

export UV_CACHE_DIR="$TEAM/.uv/cache"
export UV_TOOL_DIR="$TEAM/.uv/tools"
export UV_PYTHON_INSTALL_DIR="$TEAM/.uv/python"
export HF_HOME="$TEAM/.cache"
export HUGGINGFACE_HUB_CACHE="$TEAM/.cache/hub"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME"

if [[ -f "$HOME/.local/bin/env" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/.local/bin/env"
fi

mkdir -p "$TEAM"

# 1. Clone dataset (git — bypasses HF Hub rate limits)
if [[ -d "$DEST/.git" ]]; then
  echo "Dataset repo already present at $DEST"
else
  echo "Cloning SprintML/watermark_localization..."
  cd "$TEAM"
  git clone https://huggingface.co/datasets/SprintML/watermark_localization
fi

cd "$DEST"
if command -v git-lfs >/dev/null 2>&1; then
  git lfs pull || echo "WARN: git lfs pull failed (may be OK if no LFS files)"
fi

# 2. Per-task venv (same as hackathon_setup.sh)
if [[ -d ".venv" ]]; then
  echo ".venv already exists — skipping"
else
  echo "Creating Task 1 venv..."
  uv venv -p 3.12 .venv
  VIRTUAL_ENV="$DEST/.venv" uv pip install -r requirements.txt
  echo "venv OK"
fi

# 3. Pin watermark detector submodules in team repo
if [[ -d "$REPO/.git" ]]; then
  cd "$REPO"
  git pull --ff-only || echo "WARN: git pull failed"
  bash scripts/task1/sync_watermark_repos.sh
else
  echo "WARN: repo not found at $REPO — skip submodule sync"
fi

echo "=== Task 1 setup DONE $(date) ==="
du -sh "$DEST"
echo "--- top-level files ---"
ls -la "$DEST" | head -25
echo "--- jsonl files ---"
find "$DEST" -maxdepth 2 -name '*.jsonl' 2>/dev/null | head -10 || true
