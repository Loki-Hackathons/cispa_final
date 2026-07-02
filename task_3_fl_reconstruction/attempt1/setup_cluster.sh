#!/usr/bin/env bash
# Source this on JURECA before any Task 3 command (do NOT copy "..." paths from chat).
#
#   source setup_cluster.sh
#
set -euo pipefail

export TASK3_DATA_ROOT="/p/scratch/training2625/dougnon1/Loki/FL_Data_Reconstruction"
export TASK3_REPO="/p/scratch/training2625/dougnon1/Loki/cispa_final"
export TASK3_ATTEMPT="${TASK3_REPO}/task_3_fl_reconstruction/attempt1"

if [[ ! -d "${TASK3_DATA_ROOT}/gradients" ]]; then
  echo "ERROR: gradients not found under TASK3_DATA_ROOT=${TASK3_DATA_ROOT}" >&2
  echo "  expected: ${TASK3_DATA_ROOT}/gradients/model1.pt" >&2
  exit 1
fi

if [[ -f "${TASK3_DATA_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${TASK3_DATA_ROOT}/.venv/bin/activate"
else
  echo "WARN: no venv at ${TASK3_DATA_ROOT}/.venv — activate manually if needed" >&2
fi

cd "${TASK3_ATTEMPT}"
echo "TASK3_DATA_ROOT=${TASK3_DATA_ROOT}"
echo "cwd=$(pwd)"
