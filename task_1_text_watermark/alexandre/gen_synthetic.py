"""Point 2: synthetic labeled data via real watermarked generation.

Organizer-hidden info: which family is active per token, test labels/spans.
Public info (watermark_config.yaml): family names, exact detector params,
keys, tokenizer, pinned repo commits. Since generation is pinned to the same
public repos/params, we can regenerate OUR OWN watermarked spans with the
real model (Qwen2.5-7B-Instruct) + the real keys, giving exactly-labeled
extra training data - not a statistical simulation, the real PRF/greenlist
mechanism run forward instead of detected backward.

Each synthetic "document" = [clean context (32 real tokens, for a stable
n-gram seed at the span start)] + [watermarked continuation, one canonical
length, one scheme]. This augments the H1 pools (fit_smm.collect_pools) that
are the scarce resource (180 real docs / ~800 real spans, 34% short-span low
SNR per docs/task1/attempt1.md). Entropy is recorded alongside (same
TextSeal 3.2 definition as entropy_pass.py) so synthetic docs can feed the
entropy-conditioned emission tables too, from the SAME 7B forward passes
used for generation (no extra GPU cost).

Correctness check (printed at the end, not just assumed): each generated
span's OWN validated detector statistic (detectors.py functions / KGW
self-salt greenlist) vs the theoretical H0 mean - large z confirms the
watermark mechanism actually fired as intended before this data is trusted
for fitting.

Generators:
- gumbelmax / textseal: vendor/textseal generator classes (unmodified PRF
  math, pinned commit 788fe8b), called directly (no HF generate() wrapper).
- kgw: vendor/lm-watermarking WatermarkLogitsProcessor (pinned commit
  8292251), official self-salt rejection-sampling greenlist, matches the
  detector already validated in kgw_scores.py / entropy_pass.py.
- unigram: vendor/unigram-watermark GPTWatermarkLogitsWarper, matches
  unigram_scan.py (REAL_KEY, VOCAB_SIZE=152064 = model logits width).

Sampling: temperature=0.9, top_p=0.95 (ProcessingConfig default in
vendor/textseal/textseal/watermarking/config.py - the most likely value
used for the real dataset, see calib_pass.py docstring for the full
triangulation). KGW/Unigram bias is added to raw logits before temp/top-p
(after_topp=False, textseal default), matching entropy_pass.py's boosted-
softmax convention.

Usage (JURECA GPU node):
    python gen_synthetic.py --out-dir output --n-per-cell 30 \
        --temperature 0.9 --top-p 0.95
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

from entropy_pass import load_model, read_jsonl

_VENDOR = Path(__file__).resolve().parents[1] / "vendor"
_LMWM = _VENDOR / "lm-watermarking"
sys.path.insert(0, str(_LMWM))

CANONICAL_LENGTHS = (31, 47, 63, 95, 159, 320)
CTX_LEN = 32
SCHEMES = ("gumbelmax", "textseal", "kgw", "unigram")

# watermark_config.yaml
TEXTSEAL_KEY_A = 947821031
TEXTSEAL_KEY_B = 1562881159
TEXTSEAL_NGRAM = 3
TEXTSEAL_ALPHA = 0.5
GUMBEL_KEY = 2004683203
GUMBEL_NGRAM = 2
UNIGRAM_KEY = 1873092841
UNIGRAM_FRACTION = 0.5
UNIGRAM_STRENGTH = 1.0
UNIGRAM_VOCAB = 152064
KGW_SEEDING_SCHEME = "ff-anchored_minhash_prf-4-True-1306382177"
KGW_GAMMA = 0.25
KGW_DELTA = 1.5


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_textseal_generator():
    """Load vendor textseal generator classes without the heavy package
    __init__ (which pulls in evaluation/mauve deps) - same trick as
    detectors.py._load_textseal_core."""
    root = _VENDOR / "textseal" / "textseal"
    for pkg_name, pkg_path in [("textseal", root), ("textseal.watermarking", root / "watermarking")]:
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(pkg_path)]
            sys.modules[pkg_name] = pkg
    _load_module("textseal.watermarking.config", root / "watermarking" / "config.py")
    _load_module("textseal.watermarking.core", root / "watermarking" / "core.py")
    _load_module("textseal.watermarking.detector", root / "watermarking" / "detector.py")
    return _load_module("textseal.watermarking.generator", root / "watermarking" / "generator.py")


_gen_mod = _load_textseal_generator()
WatermarkConfig = sys.modules["textseal.watermarking.config"].WatermarkConfig
GumbelmaxGenerator = _gen_mod.GumbelmaxGenerator
TextSealGenerator = _gen_mod.TextSealGenerator

from alternative_prf_schemes import prf_lookup, seeding_scheme_lookup  # noqa: E402
from extended_watermark_processor import WatermarkLogitsProcessor  # noqa: E402

sys.path.insert(0, str(_VENDOR / "unigram-watermark"))
from gptwm import GPTWatermarkBase  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data" / "watermark_localization"


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(-1)


def sample_clean(logits, temperature, top_p):
    probs = torch.softmax(logits / temperature, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > top_p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token).reshape(-1)


def sample_boosted(logits, bias, temperature, top_p):
    """Greenlist-style: add bias to raw logits, then temp+top-p sample
    (after_topp=False convention - textseal GreenlistGenerator default)."""
    return sample_clean(logits + bias, temperature, top_p)


class UnigramBooster:
    def __init__(self, vocab_size, device):
        base = GPTWatermarkBase(fraction=UNIGRAM_FRACTION, strength=UNIGRAM_STRENGTH,
                                vocab_size=vocab_size, watermark_key=UNIGRAM_KEY)
        self.mask = base.green_list_mask.to(device)
        self.strength = UNIGRAM_STRENGTH

    def bias(self, logits):
        return self.strength * self.mask


class KgwBooster:
    """Wraps the official WatermarkLogitsProcessor; returns BOOSTED logits
    (not a separate bias) since self-salt rejection sampling needs input_ids."""

    def __init__(self, vocab_size, device):
        self.wp = WatermarkLogitsProcessor(
            vocab=list(range(vocab_size)), gamma=KGW_GAMMA, delta=KGW_DELTA,
            seeding_scheme=KGW_SEEDING_SCHEME, select_green_tokens=True,
        )

    def boosted(self, input_ids, logits):
        """input_ids: (B, T), logits: (B, V). WatermarkLogitsProcessor loops
        over the batch internally (self-salt rejection sampling is not
        vectorizable across the vocab candidates)."""
        return self.wp(input_ids, logits.clone())


@torch.no_grad()
def generate_batch(model, ctx_ids: torch.Tensor, length: int, scheme: str,
                   temperature: float, top_p: float, device,
                   unigram: UnigramBooster, kgw: KgwBooster):
    """ctx_ids: (B, CTX_LEN). Returns (gen_ids (B, length), entropy (B, CTX_LEN+length))."""
    B = ctx_ids.shape[0]
    wm_gm = WatermarkConfig(secret_key=GUMBEL_KEY, ngram=GUMBEL_NGRAM, watermark_type="gumbelmax", method="uniform")
    wm_ts = WatermarkConfig(secret_key=TEXTSEAL_KEY_A, secret_key_b=TEXTSEAL_KEY_B,
                            ngram=TEXTSEAL_NGRAM, mixing_alpha=TEXTSEAL_ALPHA, watermark_type="textseal")
    gm_gen = GumbelmaxGenerator.__new__(GumbelmaxGenerator)
    gm_gen.wm_args, gm_gen.ngram, gm_gen.secret_key = wm_gm, GUMBEL_NGRAM, GUMBEL_KEY
    ts_gen = TextSealGenerator.__new__(TextSealGenerator)
    ts_gen.wm_args, ts_gen.ngram = wm_ts, TEXTSEAL_NGRAM
    ts_gen.key_a, ts_gen.key_b, ts_gen.mixing_alpha = TEXTSEAL_KEY_A, TEXTSEAL_KEY_B, TEXTSEAL_ALPHA

    outputs = model(ctx_ids, use_cache=True)
    past = outputs.past_key_values
    ctx_logits = outputs.logits  # (B, CTX_LEN, V)
    entropy = torch.full((B, CTX_LEN + length), -1.0, device=device)
    entropy[:, 1:CTX_LEN] = entropy_from_logits(ctx_logits[:, :-1, :])

    full_ids = ctx_ids
    cur_logits = ctx_logits[:, -1, :]
    gen_tokens = []
    for step in range(length):
        entropy[:, CTX_LEN + step] = entropy_from_logits(cur_logits)
        ngram = GUMBEL_NGRAM if scheme == "gumbelmax" else TEXTSEAL_NGRAM
        window = full_ids[:, -ngram:]
        if scheme == "gumbelmax":
            next_tok = gm_gen.sample_next(cur_logits, window, temperature, top_p)
        elif scheme == "textseal":
            next_tok = ts_gen.sample_next(cur_logits, window, temperature, top_p)
        elif scheme == "unigram":
            next_tok = sample_boosted(cur_logits, unigram.bias(cur_logits), temperature, top_p)
        elif scheme == "kgw":
            boosted = kgw.boosted(full_ids, cur_logits)
            next_tok = sample_clean(boosted, temperature, top_p)
        else:
            raise ValueError(scheme)
        gen_tokens.append(next_tok)
        full_ids = torch.cat([full_ids, next_tok.unsqueeze(1)], dim=1)
        out = model(next_tok.unsqueeze(1), use_cache=True, past_key_values=past)
        past = out.past_key_values
        cur_logits = out.logits[:, -1, :]
    gen_ids = torch.stack(gen_tokens, dim=1)  # (B, length)
    return gen_ids, entropy


def sample_clean_contexts(docs, n, ctx_len, rng):
    """n random (label==0) windows of length ctx_len from real train+val docs."""
    pool = []
    for d in docs:
        lab = np.array(d["labels"])
        ids = d["token_ids"]
        clean = lab == 0
        run_start = None
        for i in range(len(lab) + 1):
            if i < len(lab) and clean[i]:
                if run_start is None:
                    run_start = i
            else:
                if run_start is not None and i - run_start >= ctx_len:
                    pool.append((ids, run_start, i))
                run_start = None
    if not pool:
        raise SystemExit("No clean run >= ctx_len found in labeled docs")
    out = []
    for _ in range(n):
        ids, s, e = pool[rng.integers(0, len(pool))]
        start = int(rng.integers(s, e - ctx_len + 1))
        out.append(ids[start:start + ctx_len])
    return out


def self_check(scheme: str, token_ids: list[int], ctx_len: int) -> float:
    """Quick z-like check restricted to the generated span, reusing
    detectors.py's own validated signal functions."""
    import detectors as det
    span = token_ids[max(0, ctx_len - det.TEXTSEAL_NGRAM):]
    if scheme == "gumbelmax":
        r = det.gumbelmax_r(token_ids)[ctx_len:]
        g = -np.log1p(-np.clip(r, 0, 1 - 1e-9))
        return float((g.mean() - 1.0) / (1.0 / np.sqrt(len(g))))
    if scheme == "textseal":
        ra, rb = det.textseal_r(token_ids)
        ra, rb = ra[ctx_len:], rb[ctx_len:]
        g = TEXTSEAL_ALPHA * -np.log1p(-np.clip(ra, 0, 1 - 1e-9)) + \
            (1 - TEXTSEAL_ALPHA) * -np.log1p(-np.clip(rb, 0, 1 - 1e-9))
        return float((g.mean() - 1.0) / (np.sqrt(0.5) / np.sqrt(len(g))))
    if scheme == "unigram":
        mask = det.unigram_mask(UNIGRAM_VOCAB)
        ids = np.asarray(token_ids[ctx_len:])
        green = mask[ids[ids < UNIGRAM_VOCAB]]
        p = green.mean()
        n = len(green)
        return float((p - 0.5) / np.sqrt(0.25 / n)) if n else 0.0
    if scheme == "kgw":
        prf_type, ctxw, self_salt, hash_key = seeding_scheme_lookup(KGW_SEEDING_SCHEME)
        L = ctxw + 1 - int(self_salt)
        greens = []
        for t in range(ctx_len, len(token_ids)):
            if t - L + 1 < 0:
                continue
            window = tuple(token_ids[t - L + 1:t + 1]) if self_salt else tuple(token_ids[t - L:t])
            ids_t = torch.as_tensor(window)
            prf_key = prf_lookup[prf_type](ids_t, salt_key=hash_key)
            rng = torch.Generator()
            rng.manual_seed(int(prf_key) % (2**64 - 1))
            perm = torch.randperm(UNIGRAM_VOCAB, generator=rng)
            green_ids = set(perm[:int(UNIGRAM_VOCAB * KGW_GAMMA)].tolist())
            greens.append(1.0 if token_ids[t] in green_ids else 0.0)
        p = float(np.mean(greens)) if greens else 0.0
        n = len(greens)
        return float((p - KGW_GAMMA) / np.sqrt(KGW_GAMMA * (1 - KGW_GAMMA) / n)) if n else 0.0
    raise ValueError(scheme)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-per-cell", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    model, name = load_model(args.device, require_primary=True)
    print(f"model={name}, vocab_size={model.config.vocab_size}", flush=True)
    if model.config.vocab_size != UNIGRAM_VOCAB:
        raise SystemExit(f"vocab_size mismatch: {model.config.vocab_size} != {UNIGRAM_VOCAB}")

    real_docs = read_jsonl(DATA / "train.jsonl") + read_jsonl(DATA / "validation.jsonl")
    rng = np.random.default_rng(args.seed)

    unigram = UnigramBooster(UNIGRAM_VOCAB, device)
    kgw = KgwBooster(UNIGRAM_VOCAB, device)

    out_docs = []
    entropy_out = {}
    check_z = {s: [] for s in SCHEMES}
    t0 = time.time()
    doc_idx = 0
    for scheme in SCHEMES:
        for L in CANONICAL_LENGTHS:
            ctxs = sample_clean_contexts(real_docs, args.n_per_cell, CTX_LEN, rng)
            ctx_ids = torch.tensor(np.stack(ctxs), device=device, dtype=torch.long)
            gen_ids, entropy = generate_batch(model, ctx_ids, L, scheme, args.temperature,
                                              args.top_p, device, unigram, kgw)
            for b in range(ctx_ids.shape[0]):
                tok = ctx_ids[b].tolist() + gen_ids[b].tolist()
                did = f"synth_{scheme}_{L}_{b}_{doc_idx}"
                doc_idx += 1
                labels = [0] * CTX_LEN + [1] * L
                out_docs.append({"document_id": did, "token_ids": tok, "labels": labels})
                entropy_out[did] = entropy[b].cpu().numpy().astype(np.float32)
                check_z[scheme].append(self_check(scheme, tok, CTX_LEN))
            elapsed = time.time() - t0
            print(f"{scheme:10s} L={L:3d}  n={ctx_ids.shape[0]:3d}  "
                  f"mean_self_z={np.mean(check_z[scheme][-args.n_per_cell:]):.2f}  "
                  f"({elapsed:.0f}s elapsed)", flush=True)

    print("\n=== Self-check summary (mean/min z per scheme, all cells) ===", flush=True)
    for s in SCHEMES:
        zs = np.array(check_z[s])
        print(f"{s:10s} mean_z={zs.mean():7.2f}  min_z={zs.min():7.2f}  n={len(zs)}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "synthetic_train.jsonl", "w", encoding="utf-8") as f:
        for d in out_docs:
            f.write(json.dumps(d) + "\n")
    np.savez_compressed(out_dir / "entropy_synth.npz", **entropy_out)
    print(f"\nWrote {len(out_docs)} synthetic docs to {out_dir / 'synthetic_train.jsonl'} "
          f"+ entropy_synth.npz ({time.time() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
