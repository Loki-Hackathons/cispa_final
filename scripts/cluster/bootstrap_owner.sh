#!/bin/bash
# One-shot OWNER bootstrap for CISPA Grand Finals (Team Loki, ansart1).
# Run over ssh:  ssh ... "bash -ls" < scripts/cluster/bootstrap_owner.sh
# Idempotent: safe to re-run. Long organizer setup runs inside cluster tmux.
set -uo pipefail

PROJECT_ID="training2625"
OWNER="ansart1"
TEAM_FOLDER="loki"
TEAMMATE_1="dougnon1"
TEAMMATE_2="paoli1"
TEAMMATE_3="abider1"
SCRATCH_BASE="/p/scratch/${PROJECT_ID}/${OWNER}"
REPO_DIR="/p/home/jusers/${OWNER}/jureca/code/cispa_final"
REPO_URL="https://github.com/Loki-Hackathons/cispa_final.git"

log() { echo ""; echo "==== $* ===="; }

log "1/6 Activate project ${PROJECT_ID}"
jutil env activate -p "${PROJECT_ID}" || echo "WARN: jutil activate failed (non-fatal in scripts)"

log "2/6 Scratch folder + organizer scripts"
mkdir -p "${SCRATCH_BASE}"
cd "${SCRATCH_BASE}"
for f in hackathon_setup.sh teammate.sh; do
  if [[ ! -s "$f" ]]; then
    wget -q "https://huggingface.co/datasets/SprintML/hackathon/resolve/main/$f" -O "$f" \
      && echo "downloaded $f" || echo "ERROR: wget $f failed"
  else
    echo "$f already present"
  fi
done

log "3/6 Configure organizer scripts (Team Loki)"
patch_var() {
  local file="$1" key="$2" val="$3"
  if grep -q "^${key}=" "$file"; then
    sed -i "s|^${key}=.*|${key}=\"${val}\"|" "$file"
  fi
}
if [[ -s hackathon_setup.sh ]]; then
  patch_var hackathon_setup.sh OWNER "${OWNER}"
  patch_var hackathon_setup.sh TEAMMATE_1 "${TEAMMATE_1}"
  patch_var hackathon_setup.sh TEAMMATE_2 "${TEAMMATE_2}"
  patch_var hackathon_setup.sh TEAMMATE_3 "${TEAMMATE_3}"
  patch_var hackathon_setup.sh YOUR_FOLDER "${OWNER}"
  patch_var hackathon_setup.sh TEAM_FOLDER "${TEAM_FOLDER}"
  echo "--- variables now set in hackathon_setup.sh:"
  grep -E "^(OWNER|TEAMMATE_[123]|YOUR_FOLDER|TEAM_FOLDER)=" hackathon_setup.sh || true
fi
if [[ -s teammate.sh ]]; then
  patch_var teammate.sh OWNER "${OWNER}"
  patch_var teammate.sh TEAM_FOLDER "${TEAM_FOLDER}"
fi

log "4/6 Clone/update team repo"
mkdir -p "$(dirname "${REPO_DIR}")"
if [[ -d "${REPO_DIR}/.git" ]]; then
  git -C "${REPO_DIR}" pull --ff-only || echo "WARN: git pull failed"
else
  git clone "${REPO_URL}" "${REPO_DIR}" || echo "ERROR: clone failed"
fi

log "5/6 Repo venv (dashboard + submit tools, lightweight)"
cd "${REPO_DIR}" || exit 1
mkdir -p logs output "${SCRATCH_BASE}/${TEAM_FOLDER}/history" 2>/dev/null || true
if [[ ! -d .venv ]]; then
  python3 -m venv .venv 2>/dev/null || module load Python && python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r dashboard/requirements.txt numpy rich && echo "venv deps OK"
deactivate

log "6/6 Organizer setup in tmux (datasets + per-task venvs — long)"
if tmux has-session -t owner_setup 2>/dev/null; then
  echo "tmux session 'owner_setup' already exists — check ${SCRATCH_BASE}/setup.log"
else
  tmux new-session -d -s owner_setup \
    "cd ${SCRATCH_BASE} && source hackathon_setup.sh 2>&1 | tee ${SCRATCH_BASE}/setup.log; echo BOOTSTRAP_DONE >> ${SCRATCH_BASE}/setup.log; exec bash"
  echo "started tmux 'owner_setup' — logs: ${SCRATCH_BASE}/setup.log"
fi

log "BOOTSTRAP SUMMARY"
echo "Scratch:   ${SCRATCH_BASE} (team folder: ${TEAM_FOLDER})"
echo "Repo:      ${REPO_DIR}"
echo "Setup log: ${SCRATCH_BASE}/setup.log  (tail -f from tmux attach -t owner_setup)"
echo "Check later: tail -5 ${SCRATCH_BASE}/setup.log && ls ${SCRATCH_BASE}/${TEAM_FOLDER}/"
