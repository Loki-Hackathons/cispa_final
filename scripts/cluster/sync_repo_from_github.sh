#!/usr/bin/env bash
# Reset cluster clone to match GitHub (discards local edits to tracked files).
# Safe on JURECA: machine-specific dashboard config lives in config_local.py (gitignored).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "==> Fetch origin"
git fetch origin

echo "==> Reset to origin/main (drops local edits to tracked files)"
git reset --hard origin/main

echo "==> Submodules"
git submodule sync --recursive
git submodule update --init --recursive

if [[ -f dashboard/config_local.py.example && ! -f dashboard/config_local.py ]]; then
  cp dashboard/config_local.py.example dashboard/config_local.py
  echo "==> Created dashboard/config_local.py (MODE=live)"
fi

if [[ -x scripts/task1/sync_watermark_repos.sh ]]; then
  bash scripts/task1/sync_watermark_repos.sh
fi

echo
echo "Done. Dashboard: MODE from dashboard/config_local.py on this machine."
git log -1 --oneline
