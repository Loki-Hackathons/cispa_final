"""Combined 7B forward-pass signals for Task 1 (GPU, single pass per doc).

For each document, runs one forward pass and derives four per-token arrays
from the resulting logits:

- entropy: H(p(x_t | x_<t)) in nats (TextSeal 3.2 weighting proxy).
- logp: log p(x_t | x_<t)) for the REALIZED token, i.e. the model's own
  probability of what was actually generated. Feeds the closed-form
  Gumbel-max / TextSeal detector (Aaronson): under the argmax-of-r^(1/p)
  reparameterization, LLR(r, p) = log(1/p) + (1/p - 1)*log(r), replacing the
  binned-empirical LLR with an exact, per-token statistic (smm_scorer.py
  gumbel_exact_llr / textseal_exact_llr).
- unigram_lpg: log P(token in the fixed Unigram greenlist) under the
  strength-boosted softmax (theoretical green rate at this position, not
  just whether the *realized* token happens to be green) - sharper than a
  single fitted Bernoulli rate.
- kgw_lpg: same idea for KGW, but the greenlist is context-dependent
  (CUDA Philox permutation per n-gram, exactly replicating kgw_scores.py).

Boosted-softmax green mass is derived in closed form from the *unboosted*
log-softmax already computed for entropy (no need to rebuild a full
boosted-vocab softmax per position):
    lg_green_boosted = (lg_green + boost) - logaddexp(lg_green + boost, lg_red)
where lg_green = logsumexp(logp0 at green indices) under the unboosted
distribution and lg_red = log(1 - exp(lg_green)).

Position 0 (no context) and the ngram warm-up positions get sentinel -1 for
entropy, +1.0 for logp (impossible value, valid logp <= 0), and NaN for the
green-mass arrays (no permutation defined there).

Output per split: entropy_{split}.npz, logp_{split}.npz,
unigram_lpg_{split}.npz, kgw_lpg_{split}.npz (document_id -> float32 array).

Usage (JURECA GPU node):
    python entropy_pass.py --data-dir <dataset dir> --out-dir output \
        --splits train validation test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PRIMARY_MODEL = "Qwen/Qwen2.5-7B-Instruct"     # pinned generator, revision in watermark_config.yaml
FALLBACK_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # proxy for entropy only (TextSeal fig. 6); logp/green-mass need the real generator

_VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "lm-watermarking"
sys.path.insert(0, str(_VENDOR))
from alternative_prf_schemes import prf_lookup, seeding_scheme_lookup  # noqa: E402

# --- Unigram (fixed greenlist, context-free) ---
from unigram_scan import REAL_KEY, VOCAB_SIZE as UNIGRAM_VOCAB, greenlist_mask  # noqa: E402

UNIGRAM_STRENGTH = 1.0

# --- KGW (context-dependent greenlist) ---
KGW_SEEDING_SCHEME = "ff-anchored_minhash_prf-4-True-1306382177"
KGW_GAMMA = 0.25
KGW_DELTA = 1.5
# Pinned to the vocab size selected by kgw_scores.py --auto on labeled train
# docs (job 15399747, z=9.3 best of 5 candidates - see docs/task1/attempt1.md
# §8). KGW's own greenlist code seeds off len(tokenizer vocab), not the model
# logits width, unlike Unigram (see watermark_config.yaml / vendor comments).
KGW_VOCAB_SIZE = 151665


def _local_snapshot_dir(name: str) -> Path | None:
    """Resolve <HF_HOME>/hub/models--Org--Name/snapshots/<hash> directly,
    bypassing huggingface_hub's cache index (robust to caches copied in from
    elsewhere without a matching refs/ pointer or blob symlinks)."""
    import os
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    repo_dir = hf_home / "hub" / f"models--{name.replace('/', '--')}" / "snapshots"
    if not repo_dir.is_dir():
        return None
    snaps = [d for d in repo_dir.iterdir() if d.is_dir()]
    if not snaps:
        return None
    return max(snaps, key=lambda d: d.stat().st_mtime)


def load_model(device, require_primary=False):
    from transformers import AutoModelForCausalLM
    candidates = [PRIMARY_MODEL] + ([] if require_primary else [FALLBACK_MODEL])
    for name in candidates:
        local_dir = _local_snapshot_dir(name)
        attempts = ([(str(local_dir), True)] if local_dir else []) + [(name, True), (name, False)]
        for path_or_name, local_only in attempts:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    path_or_name, torch_dtype=torch.float16, local_files_only=local_only)
                model.to(device).eval()
                print(f"loaded {name} from {path_or_name!r} (local_only={local_only})",
                      flush=True)
                return model, name
            except Exception as e:  # not cached / no network on compute node
                print(f"cannot load {path_or_name!r} local_only={local_only}: {e}",
                      flush=True)
    raise SystemExit("No model available. Pre-download on a login node with: "
                     "python -c \"from transformers import AutoModelForCausalLM as M; "
                     f"M.from_pretrained('{PRIMARY_MODEL}')\"")


class KgwGreenIndex:
    """Per-ngram green-index lookup, replicating kgw_scores.py exactly
    (CUDA Philox torch.randperm) but returning indices instead of a bool."""

    def __init__(self, vocab_size: int, device):
        self.prf_type, self.context_width, self.self_salt, self.hash_key = \
            seeding_scheme_lookup(KGW_SEEDING_SCHEME)
        self.vocab_size = vocab_size
        self.device = device
        self.rng = torch.Generator(device=device)
        self.greenlist_size = int(vocab_size * KGW_GAMMA)
        self.ngram_len = self.context_width + 1 - int(self.self_salt)

    def green_idx(self, ngram: tuple[int, ...]) -> torch.Tensor:
        seed_window = ngram if self.self_salt else ngram[:-1]
        ids = torch.as_tensor(seed_window, device=self.device)
        prf_key = prf_lookup[self.prf_type](ids, salt_key=self.hash_key)
        self.rng.manual_seed(prf_key % (2**64 - 1))
        perm = torch.randperm(self.vocab_size, device=self.device, generator=self.rng)
        return perm[: self.greenlist_size]


def _boosted_log_pgreen(lg_green: torch.Tensor, boost: float) -> torch.Tensor:
    """Closed-form log P(green) after adding `boost` to green logits,
    derived from the log P(green) under the UNBOOSTED softmax."""
    lg_green = lg_green.clamp(max=-1e-8)  # keep exp(lg_green) < 1 for log1p
    lg_red = torch.log1p(-lg_green.exp())
    numer = lg_green + boost
    return numer - torch.logaddexp(numer, lg_red)


@torch.no_grad()
def doc_pass(model, token_ids, device, kgw_scorer: KgwGreenIndex,
            unigram_green_idx: torch.Tensor) -> dict[str, np.ndarray]:
    n = len(token_ids)
    ids = torch.tensor([token_ids], device=device)
    logits = model(ids).logits[0].float()               # (n, V)
    logp0 = torch.log_softmax(logits, dim=-1)            # unboosted log-softmax

    ent = -(logp0.exp() * logp0).sum(-1)                 # H of p(x_{t+1} | x_<=t)
    entropy = np.full(n, -1.0, dtype=np.float32)
    entropy[1:] = ent[:-1].cpu().numpy().astype(np.float32)

    target_logp = logp0[torch.arange(n - 1, device=device), ids[0, 1:]]
    logp = np.full(n, 1.0, dtype=np.float32)             # sentinel: no context
    logp[1:] = target_logp.cpu().numpy().astype(np.float32)

    # Unigram: fixed greenlist, vectorized over the whole document.
    lp0_shift = logp0[:-1]                                # distribution predicting token t, t>=1
    lg_green_u = torch.logsumexp(lp0_shift[:, unigram_green_idx], dim=-1)
    lpg_u = _boosted_log_pgreen(lg_green_u, UNIGRAM_STRENGTH)
    unigram_lpg = np.full(n, np.nan, dtype=np.float32)
    unigram_lpg[1:] = lpg_u.cpu().numpy().astype(np.float32)

    # KGW: context-dependent greenlist, looped per position (self-salt: the
    # seeding window includes the target, so it needs ngram_len tokens ending
    # AT t, and the *prediction* distribution used is the one for position
    # t, i.e. logp0[t-1]).
    L = kgw_scorer.ngram_len
    kgw_lpg = np.full(n, np.nan, dtype=np.float32)
    for t in range(L - 1, n):
        ngram = tuple(token_ids[t - L + 1:t + 1])
        green_idx = kgw_scorer.green_idx(ngram)
        lg_green = torch.logsumexp(logp0[t - 1, green_idx], dim=-1)
        kgw_lpg[t] = float(_boosted_log_pgreen(lg_green, KGW_DELTA).cpu())

    return {"entropy": entropy, "logp": logp, "unigram_lpg": unigram_lpg,
           "kgw_lpg": kgw_lpg}


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--require-primary", action="store_true",
                    help="fail instead of falling back to the 0.5B proxy "
                         "(logp / green-mass signals need the real generator)")
    args = ap.parse_args()

    model, name = load_model(args.device, require_primary=args.require_primary)
    is_primary = name == PRIMARY_MODEL
    if not is_primary:
        print("WARNING: using fallback model - logp/unigram_lpg/kgw_lpg will "
              "be biased (wrong generator probabilities). Entropy only is "
              "the validated use case for this proxy.", flush=True)
    elif model.config.vocab_size != UNIGRAM_VOCAB:
        raise SystemExit(f"model.config.vocab_size={model.config.vocab_size} "
                         f"!= UNIGRAM_VOCAB={UNIGRAM_VOCAB}: unigram_lpg "
                         "green-index gather would be silently misaligned")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    kgw_scorer = KgwGreenIndex(KGW_VOCAB_SIZE, device)
    unigram_mask = greenlist_mask(REAL_KEY, UNIGRAM_VOCAB)
    unigram_green_idx = torch.as_tensor(np.nonzero(unigram_mask)[0], device=device)

    for split in args.splits:
        records = read_jsonl(Path(args.data_dir) / f"{split}.jsonl")
        out = {k: {} for k in ("entropy", "logp", "unigram_lpg", "kgw_lpg")}
        t0 = time.time()
        for i, rec in enumerate(records):
            sig = doc_pass(model, rec["token_ids"], device, kgw_scorer,
                          unigram_green_idx)
            did = str(rec["document_id"])
            for k in out:
                out[k][did] = sig[k]
            if (i + 1) % 50 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"{split}: {i + 1}/{len(records)} ({rate:.2f} docs/s)",
                      flush=True)
        for k, data in out.items():
            np.savez_compressed(out_dir / f"{k}_{split}.npz", **data)
        print(f"{split}: wrote entropy/logp/unigram_lpg/kgw_lpg_{split}.npz "
              f"({len(records)} docs, {time.time() - t0:.0f}s, model={name})",
              flush=True)


if __name__ == "__main__":
    main()
