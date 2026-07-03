# Task 2 — Member vs Generated Inference (MGI)

**Owner:** Florian (dougnon1)

Official spec: [Task 2 Description.md](../../docs/Task%202%20Description.md) · **Roadmap détaillé : [docs/task2/roadmap.md](../../docs/task2/roadmap.md)**

Metric: `mean(DetectorScore × (1−MSE))` over 1800 images (6 directions × 300).
`DetectorScore` is binary and dominates — **make the flip transfer first, minimise
MSE second.** Submit `.npz` (1800 × 256 × 256 × 3, uint8, IDs 0000–1799).

## Detector model (assumed): DCB on RAR (direct training)

- **Stage 1** — autoencoder `L_A = L_R + α·L_Q` on the MaskGIT VQ-GAN (RAR's
  tokenizer). `L_A ≤ τ_G` ⇒ generated (**G**). Our proxy is the *exact* component.
- **Stage 2** — RAR ICAS (`nll_uncond − nll_cond`) on non-generated images.
  `ICAS ≥ τ_MN` ⇒ member (**M**) else non-member (**N**). Order: `ICAS(N) < ICAS(M) < ICAS(G)`.

Per-direction target cells (see `config.DIRECTION_TARGETS`):

| Dir | Slots | Source ref | Stage 1 | Stage 2 | Recipe | Difficulty |
|-----|-------|-----------|---------|---------|--------|------------|
| M→G | 300–599 | M | generate (`L_A`↓) | off | reconstruct | easy |
| N→G | 900–1199 | N | generate (`L_A`↓) | off | reconstruct | easy |
| G→M | 1200–1499 | G | natural (`L_A`↑) | raise (already high) | combined / `L_A`↑ | easy-med |
| M→N | 0–299 | M | keep natural | lower ICAS | combined / perturb | med |
| G→N | 1500–1799 | G | natural (`L_A`↑) | lower ICAS | combined (2 obj) / swap | hard |
| N→M | 600–899 | N | keep natural | raise ICAS >> M-max | combined / swap | hard |

## Files

| File | Role |
|------|------|
| `config.py` | Paths, hyperparams, slot layout, **`DIRECTION_TARGETS`** (per-cell) |
| `proxy_dcb.py` | `L_Q, L_R, L_A`, calibration `τ_G / α` (Stage 1) |
| `proxy_icas.py` | RAR ICAS (Stage 2) |
| `attack_combined.py` | **Unified per-cell attack** (Stage 1 × Stage 2) — use this |
| `evaluate_submission.py` | **Local proxy-DCB scorer** — gate every API submit |
| `cw_attack.py` | C&W L2 on `L_A` (Stage 1 only) — faithful for `*→G` |
| `attack_membership.py` | Differentiable RAR forward + `--selftest` (reused by combined) |
| `reconstruct_to_g.py` | `*→G` blocks (VQ-GAN encode-decode) — safe fallback |
| `build_content_swap.py` | Nearest-target swap — guaranteed-flip fallback for →M / →N |
| `build_submission_v2.py` | **Assemble** best method per direction |
| `run_attack.py` | LEGACY Stage-1 CLI (warns on unsupported directions) |

## Recommended workflow (JURECA GPU node)

```bash
export LOKI_ROOT=/p/scratch/training2625/dougnon1/Loki
cd $LOKI_ROOT/cispa_final
export ONED_TOKENIZER_ROOT=$LOKI_ROOT/1d-tokenizer
export PYTHONPATH=$ONED_TOKENIZER_ROOT:$PWD/task_2_mgi/attempt1:$PWD/shared
source $LOKI_ROOT/MGI/.venv/bin/activate

# 0. sanity: proxy separates G from M/N
python task_2_mgi/attempt1/smoke_test_proxy.py

# 1. easy directions (Stage 1): reconstructions for *->G
python task_2_mgi/attempt1/reconstruct_to_g.py --classes M N

# 2. unified attack (validate replication first!)
python task_2_mgi/attempt1/attack_combined.py --directions N_M --selftest --limit 8
python task_2_mgi/attempt1/attack_combined.py --directions M_N,G_M,G_N,N_M --steps 300

# 3. safe fallbacks for the hard ->M / ->N (guaranteed flip, higher MSE)
python task_2_mgi/attempt1/build_content_swap.py            # all-swap baseline

# 4. assemble best-per-direction
python task_2_mgi/attempt1/build_submission_v2.py \
    --dir M_G=block:output/blocks/M_G_recon.npy \
    --dir N_G=block:output/blocks/N_G_recon.npy \
    --dir M_N=block:output/blocks/M_N_combined.npy \
    --dir G_M=block:output/blocks/G_M_combined.npy \
    --dir N_M=block:output/blocks/N_M_combined.npy \
    --dir G_N=block:output/blocks/G_N_combined.npy

# 5. SCORE LOCALLY before spending an API submit (gate: flip-rate >= 0.9/dir)
python task_2_mgi/attempt1/evaluate_submission.py output/submission.npz

# 6. submit
export CISPA_BASE_URL=http://35.192.205.84
export CISPA_API_KEY=<team_key>
python shared/submit.py output/submission.npz \
  --task-id 29-mgi --action submit --owner dougnon1 \
  --method "combined per-cell attack + recon + swap fallback"
```

Output: `output/submission.npz` (< 200 MB, enforced by `submission_io`).
