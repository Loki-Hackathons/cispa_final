#!/bin/bash
# Run on JURECA after wget of hackathon_setup.sh and teammate.sh.
# Usage: bash scripts/cluster/configure_scripts.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=loki.env
source "${SCRIPT_DIR}/loki.env"

for f in hackathon_setup.sh teammate.sh; do
  if [[ ! -f "$f" ]]; then
    echo "Missing $f in $(pwd). wget it first (see scripts/cluster/README.md)."
    exit 1
  fi
done

patch_var() {
  local file="$1" key="$2" val="$3"
  if grep -q "^${key}=" "$file"; then
    sed -i "s|^${key}=.*|${key}=\"${val}\"|" "$file"
  else
    echo "${key}=\"${val}\"" >> "$file"
  fi
}

patch_var hackathon_setup.sh OWNER "$OWNER"
patch_var hackathon_setup.sh TEAMMATE_1 "$TEAMMATE_1"
patch_var hackathon_setup.sh TEAMMATE_2 "$TEAMMATE_2"
patch_var hackathon_setup.sh TEAMMATE_3 "$TEAMMATE_3"
patch_var hackathon_setup.sh YOUR_FOLDER "$YOUR_FOLDER"
patch_var hackathon_setup.sh TEAM_FOLDER "$TEAM_FOLDER"

patch_var teammate.sh OWNER "$OWNER"
patch_var teammate.sh TEAM_FOLDER "$TEAM_FOLDER"

echo "Configured hackathon_setup.sh and teammate.sh for Team Loki."
echo "Owner: source hackathon_setup.sh"
echo "Teammates: source teammate.sh"
