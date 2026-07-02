#!/bin/bash
#SBATCH --account=training2625
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=t1_all_${USER}
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#
# ============================================================================
#  Task 1 - Text Watermark Localization (Melissa)
#  SCRIPT UNIQUE : fait TOUT le pipeline puis soumet ta partie.
#
#  Etapes : venv -> .env -> check data -> train -> eval val -> predict (jsonl)
#           -> validation format -> submit leaderboard.
#
#  Lancer en job GPU (recommande, KGW a besoin du GPU/CUDA Philox) :
#      sbatch task_1_text_watermark/melissa/scripts/run_all_and_submit.sh
#
#  Ou directement sur un noeud avec GPU / en interactif :
#      bash task_1_text_watermark/melissa/scripts/run_all_and_submit.sh
#
#  Options (variables d'env) :
#      MODEL=gboost                 # ou logreg (defaut: gboost)
#      SUBMIT=1                     # 1 = soumet, 0 = genere seulement (defaut: 1)
#      WML_WATERMARK_YAML=/chemin/vers/watermark_keys.yaml   # vraies cles
# ============================================================================
set -euo pipefail

# --- Reglages -----------------------------------------------------------------
MODEL="${MODEL:-gboost}"
SUBMIT="${SUBMIT:-1}"
TASK_ID="30-watermark-localization"

# --- Racine du repo -----------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-/p/home/jusers/${USER}/jureca/code/cispa_final}"
cd "$REPO_ROOT"

# --- Environnement Python -----------------------------------------------------
# (commente ces 2 lignes si tu n'es pas sur JURECA)
module load GCC CUDA PyTorch torchvision 2>/dev/null || true
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

# --- Cles API (.env : CISPA_BASE_URL + CISPA_API_KEY) -------------------------
if [ -f .env ]; then
    set -a; source .env; set +a
else
    echo "ATTENTION: pas de .env trouve a la racine du repo."
    echo "           La soumission echouera sans CISPA_BASE_URL / CISPA_API_KEY."
fi

# --- Cles watermark du dataset (YAML) -----------------------------------------
export WML_WATERMARK_YAML="${WML_WATERMARK_YAML:-/p/scratch/training2625/ansart1/loki/watermark_keys.yaml}"

MEL="task_1_text_watermark/melissa"
cd "$MEL"
mkdir -p logs outputs

echo "==================================================================="
echo " Task 1 pipeline complet | model=$MODEL | submit=$SUBMIT"
echo " YAML cles watermark: $WML_WATERMARK_YAML"
echo "==================================================================="

echo "[1/4] Verification du dataset ..."
python -m src.load_data --check

echo "[2/4] Entrainement du calibrateur ($MODEL) + evaluation val ..."
python -m src.train_calibrator --model "$MODEL"
python -m src.evaluate --pred outputs/val_pred.jsonl --split validation

echo "[3/4] Generation + validation du .jsonl de test ..."
python -m src.predict --model "outputs/calibrator_${MODEL}.pkl"

SUB="$REPO_ROOT/$MEL/outputs/submission.jsonl"
echo "[4/4] Soumission ..."
if [ "$SUBMIT" = "1" ]; then
    cd "$REPO_ROOT"
    python shared/submit.py "$SUB" \
        --task-id "$TASK_ID" --action submit --owner "${USER}"
else
    echo "SUBMIT=0 -> pas de soumission. Fichier pret : $SUB"
    echo "Pour soumettre plus tard :"
    echo "  python shared/submit.py $SUB --task-id $TASK_ID --action submit --owner $USER"
fi

echo "==================================================================="
echo " Termine."
echo "==================================================================="
