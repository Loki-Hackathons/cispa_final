# Task 1 — watermark detector vendors (git submodules)

Three upstream repos cover **four** watermark families. TextSeal and Gumbel-Max share the same codebase.

| Submodule path | Upstream | Commit | Families |
|----------------|----------|--------|----------|
| `textseal/` | [facebookresearch/textseal](https://github.com/facebookresearch/textseal) | `788fe8b` | TextSeal, Gumbel-Max |
| `lm-watermarking/` | [jwkirchenbauer/lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking) | `8292251` | KGW |
| `unigram-watermark/` | [XuandongZhao/Unigram-Watermark](https://github.com/XuandongZhao/Unigram-Watermark) | `b96cdb4` | Unigram |

Keys and detector parameters: [`../watermark_config.yaml`](../watermark_config.yaml).

## Clone / update (pinned commits)

From repo root:

```bash
bash scripts/task1/sync_watermark_repos.sh
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/task1/sync_watermark_repos.ps1
```

Fresh clone of `cispa_final`:

```bash
git clone --recurse-submodules https://github.com/Loki-Hackathons/cispa_final.git
cd cispa_final
bash scripts/task1/sync_watermark_repos.sh
```

## Notes

- **KGW on Windows:** enable long paths (`git config core.longpaths true`) — the lm-watermarking repo has deep figure paths.
- **KGW on GPU:** greenlists must be recomputed with CUDA Philox (`torch.randperm` on GPU), not CPU.
- Do not bump submodule commits without re-checking against `watermark_config.yaml`.
