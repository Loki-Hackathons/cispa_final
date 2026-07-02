#!/bin/bash
# Login-node setup for the entropy job: fix deps, pre-download the proxy
# model, then submit run_entropy.sh. Run from the alexandre/ task dir.
set -uo pipefail

cd "$(dirname "$0")"
sed -i 's/\r$//' run_entropy.sh entropy_pass.py download_entropy_model.py
chmod +x run_entropy.sh
mkdir -p logs output

export HF_HOME=/p/scratch/training2625/ansart1/loki/hf_cache
mkdir -p "$HF_HOME"

# drop any queued duplicate of this job before resubmitting
for jid in $(squeue -u "$USER" -h -n task1_entropy_ansart1 -o %i); do
    scancel "$jid"
done

module load GCC CUDA PyTorch torchvision 2>/dev/null

# The module env pins an old regex that the latest transformers rejects;
# install a transformers release compatible with the module stack instead.
if ! python -c 'import transformers' 2>/dev/null; then
    echo "installing transformers==4.49.0 (user site)"
    python -m pip install --user --quiet 'transformers==4.49.0' 2>&1 | tail -2
fi
python -c 'import transformers; print("transformers", transformers.__version__)' || exit 1

# 7B only if already cached; otherwise cache the 0.5B proxy (~1 GB download)
if ls "$HF_HOME"/hub/models--Qwen--Qwen2.5-7B-Instruct >/dev/null 2>&1; then
    echo "7B already cached"
else
    echo "caching 0.5B proxy..."
    python download_entropy_model.py || exit 1
fi

sbatch run_entropy.sh
squeue -u "$USER"
echo SETUP_DONE
