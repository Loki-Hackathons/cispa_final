# Task 2 — Member vs Generated Inference (MGI)

**Owner:** Florian (dougnon1)

Official spec: [Task 2 Description.md](../../docs/Task%202%20Description.md)

Metric: mean(DetectorScore × (1−MSE)) over 1800 images. Submit `.npz`.

## Attack pipeline (v2)

Stage-1 proxy attack on DCB autoencoder score `L_A` (MaskGIT VQ-GAN).

| File | Role |
|------|------|
| `config.py` | Paths, hyperparameters, submission layout |
| `proxy_dcb.py` | L_Q, L_R, L_A, calibration τ_G / α |
| `cw_attack.py` | C&W L2 + input diversity |
| `run_attack.py` | Main CLI (smoke / calibrate / attack / assemble) |
| `smoke_test_proxy.py` | Quick validation L_A(G) < L_A(M/N) |
| `proxy_icas.py` | Optional Stage-2 ICAS (post-v1) |
| `run_attack.sh` | SLURM job for JURECA |

## Cluster (JURECA)

```bash
# Prerequisites: 1d-tokenizer cloned, MGI venv, HF weights downloaded
export LOKI_ROOT=/p/scratch/training2625/dougnon1/Loki
cd $LOKI_ROOT/cispa_final

# Smoke test (~2 min)
export ONED_TOKENIZER_ROOT=$LOKI_ROOT/1d-tokenizer
export PYTHONPATH=$ONED_TOKENIZER_ROOT:$PWD/shared
source $LOKI_ROOT/MGI/.venv/bin/activate
python task_2_mgi/attempt1/smoke_test_proxy.py

# Full attack (GPU, ~2-4h)
mkdir -p task_2_mgi/attempt1/logs
sbatch task_2_mgi/attempt1/run_attack.sh

# Submit to API
export CISPA_BASE_URL=http://35.192.205.84
export CISPA_API_KEY=<team_key>
python shared/submit.py task_2_mgi/attempt1/output/submission.npz \
  --task-id 29-mgi --action submit --owner dougnon1
```

## Local CLI

```bash
python task_2_mgi/attempt1/run_attack.py --phase smoke
python task_2_mgi/attempt1/run_attack.py --phase calibrate
python task_2_mgi/attempt1/run_attack.py --phase attack --directions M_G,N_G
python task_2_mgi/attempt1/run_attack.py --phase all
```

Output: `task_2_mgi/attempt1/output/submission.npz` (< 200 MB).
