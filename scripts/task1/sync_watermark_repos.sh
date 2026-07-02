#!/usr/bin/env bash
# Sync Task 1 watermark detector submodules to commits pinned in watermark_config.yaml.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

git config core.longpaths true 2>/dev/null || true

git submodule sync --recursive
git submodule update --init --recursive task_1_text_watermark/vendor/textseal \
  task_1_text_watermark/vendor/lm-watermarking \
  task_1_text_watermark/vendor/unigram-watermark

checkout() {
  local path="$1" commit="$2"
  echo "==> $path @ ${commit:0:7}"
  git -C "$path" config core.longpaths true 2>/dev/null || true
  git -C "$path" fetch --quiet origin 2>/dev/null || true
  git -C "$path" checkout -f "$commit"
  actual="$(git -C "$path" rev-parse HEAD)"
  if [[ "$actual" != "$commit" ]]; then
    echo "ERROR: $path is at $actual, expected $commit" >&2
    exit 1
  fi
}

checkout task_1_text_watermark/vendor/textseal 788fe8bff5cf086f0881928ce9a81aa08c21dff1
checkout task_1_text_watermark/vendor/lm-watermarking 82922516930c02f8aa322765defdb5863d07a00e
checkout task_1_text_watermark/vendor/unigram-watermark b96cdb4d52771e3cbd543a9d9aeeaec8d0790ca2

echo
echo "Submodules pinned:"
git submodule status task_1_text_watermark/vendor/textseal \
  task_1_text_watermark/vendor/lm-watermarking \
  task_1_text_watermark/vendor/unigram-watermark
