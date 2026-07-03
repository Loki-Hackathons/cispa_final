"""Synthetic labeled document generation for Task 1 (GPU, JURECA).

Goal (roadmap.md workstream D): the emission tables (binned LLR x entropy,
KGW Bernoulli rates) are estimated from only 180 labeled docs / ~790 spans;
estimation noise in the tail bins directly caps the 0.1%FPR threshold, and
short spans (34% of all spans, audit §17) are the first casualties. We own
everything the organizers used: the exact keys (watermark_config.yaml), the
pinned vendor generation code, and the real generator (Qwen2.5-7B-Instruct,
cached on JURECA). So we can manufacture unlimited extra fit data.

Per document:
1. Prompt = prefix of a random test doc (same domain distribution).
2. Segment plan: clean gaps + 1-3 watermarked spans, lengths drawn from the
   canonical prior {31,47,63,95,159,320}, schemes drawn from the observed
   prevalence (GM/TS/KGW ~41/36/22% + ~2% Unigram), ~10% edge-truncated.
3. Autoregressive generation with the watermark applied DURING the span:
   - Unigram: +strength on greenlist logits (vendor GPTWatermarkLogitsWarper
     semantics, mask over the model logits dim = 152064).
   - KGW: vendor WatermarkLogitsProcessor (self-salt rejection sampling,
     CUDA Philox randperm, vocab 151665 as pinned by kgw_scores --auto).
   - Gumbel-Max / TextSeal: argmax_v r_v^(1/p_v) with the pinned vendor PRF
     (textseal core.prf_uniform); TextSeal routes between the two keys with
     prob alpha per step.
4. Labels logged exactly at generation time (1 while a watermark is active).

After generation, the per-doc detection-side arrays are recomputed with the
SAME semantics as the real pipeline (doc-only context, no prompt):
- kgw_synth.npz    : green/red mask (KgwMaskScorer, kgw_scores.py)
- entropy_synth.npz: doc-only forward-pass predictive entropy (entropy_pass
  convention: position 0 sentinel -1)

Outputs: output/synth.jsonl + output/{kgw,entropy}_synth.npz, consumed by
cv_smm.py configs with use_synth=True (fit pools only; priors and CV eval
stay on real docs).

IMPORTANT - sampling params are unknown (temperature/top-p of the organizer
pipeline). Default temperature=1.0, top_p=1.0. Run validate_synth.py on the
output BEFORE using it in CV: if the synthetic H1 pools mismatch the labeled
ones, regenerate with different --temperature/--top-p until they match.

Usage (JURECA GPU node):
    python gen_synth.py --data-dir <dataset dir> --out-dir output \
        --n-docs 1000 [--temperature 1.0] [--top-p 1.0] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from detectors import (GUMBEL_KEY, GUMBEL_NGRAM, TEXTSEAL_ALPHA,
                       TEXTSEAL_KEY_A, TEXTSEAL_KEY_B, TEXTSEAL_NGRAM,
                       prf_uniform)
from entropy_pass import load_model, read_jsonl
from kgw_scores import KgwMaskScorer
from unigram_scan import REAL_KEY, VOCAB_SIZE as UNIGRAM_VOCAB, greenlist_mask

_VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "lm-watermarking"
sys.path.insert(0, str(_VENDOR))
from extended_watermark_processor import WatermarkLogitsProcessor  # noqa: E402

UNIGRAM_STRENGTH = 1.0
KGW_GAMMA = 0.25
KGW_DELTA = 1.5
KGW_SEEDING_SCHEME = "ff-anchored_minhash_prf-4-True-1306382177"
KGW_VOCAB_SIZE = 151665

# canonical span-length prior (train+val counts) and observed scheme
# prevalence among confidently-assigned labeled spans (audit §17)
SPAN_LENGTHS = np.array([31, 47, 63, 95, 159, 320])
SPAN_LENGTH_P = np.array([64, 133, 163, 145, 158, 73], dtype=float)
SPAN_LENGTH_P /= SPAN_LENGTH_P.sum()
SCHEMES = ["gumbelmax", "textseal", "kgw", "unigram"]
SCHEME_P = np.array([0.40, 0.36, 0.22, 0.02])

PROMPT_LEN = 32
EDGE_TRUNC_P = 0.10          # fraction of spans truncated by a doc boundary


def build_plan(rng: np.random.Generator, target_len: int) -> list[tuple[str | None, int]]:
    """Sequence of (scheme | None, length) segments totaling ~target_len."""
    plan = []
    n_spans = rng.integers(1, 4)
    # leading clean gap (possibly zero -> span truncated at doc start)
    if rng.random() < EDGE_TRUNC_P:
        lead = 0
    else:
        lead = int(rng.integers(30, 200))
    if lead:
        plan.append((None, lead))
    for i in range(n_spans):
        scheme = SCHEMES[rng.choice(len(SCHEMES), p=SCHEME_P)]
        L = int(SPAN_LENGTHS[rng.choice(len(SPAN_LENGTHS), p=SPAN_LENGTH_P)])
        if lead == 0 and i == 0 and rng.random() < 0.5:
            L = max(5, int(L * rng.random()))  # visible part of a cut span
        plan.append((scheme, L))
        plan.append((None, int(rng.integers(30, 200))))
    total = sum(L for _, L in plan)
    if total < target_len:
        plan.append((None, target_len - total))
    # trailing edge truncation: end the doc mid-span occasionally
    if rng.random() < EDGE_TRUNC_P and len(plan) >= 2 and plan[-1][0] is None:
        plan = plan[:-1]
        scheme, L = plan[-1]
        plan[-1] = (scheme, max(5, int(L * rng.random())))
    return plan


class GumbelSampler:
    """argmax_v r_v^(1/p_v) over the (top-p filtered) candidate set, with
    the pinned vendor PRF. TextSeal: per-step key routing with prob alpha."""

    def __init__(self, device):
        self.device = device

    def _r_all(self, context: list[int], ngram: int, key: int,
               cand: torch.Tensor) -> torch.Tensor:
        w = torch.tensor(context[-ngram:], dtype=torch.long,
                         device=self.device).unsqueeze(0).expand(len(cand), ngram)
        return prf_uniform(w, cand, key)

    def pick(self, scheme: str, context: list[int], probs: torch.Tensor,
             cand: torch.Tensor, rng: np.random.Generator) -> int:
        if scheme == "gumbelmax":
            r = self._r_all(context, GUMBEL_NGRAM, GUMBEL_KEY, cand)
        else:  # textseal: route one key per step
            key = TEXTSEAL_KEY_A if rng.random() < TEXTSEAL_ALPHA else TEXTSEAL_KEY_B
            r = self._r_all(context, TEXTSEAL_NGRAM, key, cand)
        r = r.clamp(min=1e-9, max=1.0 - 1e-9)
        score = torch.log(r) / probs.clamp(min=1e-12)
        return int(cand[torch.argmax(score)].item())


def top_p_filter(probs: torch.Tensor, top_p: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (candidate ids, renormalized probs over candidates)."""
    if top_p >= 1.0:
        return torch.arange(len(probs), device=probs.device), probs
    sorted_p, idx = probs.sort(descending=True)
    keep = sorted_p.cumsum(0) - sorted_p < top_p
    keep[0] = True
    cand = idx[keep]
    p = probs[cand]
    return cand, p / p.sum()


@torch.no_grad()
def generate_doc(model, prompt_ids: list[int], plan, device, gumbel: GumbelSampler,
                 unigram_boost: torch.Tensor, kgw_proc, temperature: float,
                 top_p: float, rng: np.random.Generator) -> tuple[list[int], list[int]]:
    """Returns (token_ids, labels) for the generated continuation only."""
    all_ids = list(prompt_ids)
    tokens, labels = [], []
    past = None
    ids_t = torch.tensor([all_ids], device=device)

    for scheme, seg_len in plan:
        for _ in range(seg_len):
            out = model(ids_t, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1].float()

            if scheme == "unigram":
                logits = logits + unigram_boost
            elif scheme == "kgw":
                full_ctx = torch.tensor(all_ids, device=device)
                logits = kgw_proc(full_ctx.unsqueeze(0),
                                  logits.unsqueeze(0))[0]

            if temperature != 1.0:
                logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            cand, p_cand = top_p_filter(probs, top_p)

            if scheme in ("gumbelmax", "textseal"):
                nxt = gumbel.pick(scheme, all_ids, p_cand, cand, rng)
            else:
                nxt = int(cand[torch.multinomial(p_cand, 1)].item())

            all_ids.append(nxt)
            tokens.append(nxt)
            labels.append(0 if scheme is None else 1)
            ids_t = torch.tensor([[nxt]], device=device)
    return tokens, labels


@torch.no_grad()
def doc_entropy(model, token_ids: list[int], device) -> np.ndarray:
    """Doc-only predictive entropy, entropy_pass.py convention."""
    ids = torch.tensor([token_ids], device=device)
    logits = model(ids).logits[0].float()
    logp0 = torch.log_softmax(logits, dim=-1)
    ent = -(logp0.exp() * logp0).sum(-1)
    out = np.full(len(token_ids), -1.0, dtype=np.float32)
    out[1:] = ent[:-1].cpu().numpy().astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--n-docs", type=int, default=1000)
    ap.add_argument("--doc-len", type=int, default=700)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA required (KGW Philox greenlists + 7B).")
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    model, name = load_model(device, require_primary=True)
    if model.config.vocab_size != UNIGRAM_VOCAB:
        raise SystemExit(f"vocab {model.config.vocab_size} != {UNIGRAM_VOCAB}")

    test_docs = read_jsonl(Path(args.data_dir) / "test.jsonl")
    gumbel = GumbelSampler(device)
    uni_mask = torch.as_tensor(greenlist_mask(REAL_KEY, UNIGRAM_VOCAB),
                               device=device)
    unigram_boost = uni_mask.float() * UNIGRAM_STRENGTH
    kgw_proc = WatermarkLogitsProcessor(
        vocab=list(range(KGW_VOCAB_SIZE)), gamma=KGW_GAMMA, delta=KGW_DELTA,
        seeding_scheme=KGW_SEEDING_SCHEME, select_green_tokens=True)
    kgw_mask_scorer = KgwMaskScorer(vocab_size=KGW_VOCAB_SIZE, device=args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kgw_out, ent_out = {}, {}
    t0 = time.time()
    meta = dict(temperature=args.temperature, top_p=args.top_p, seed=args.seed,
                n_docs=args.n_docs, doc_len=args.doc_len)

    with open(out_dir / "synth.jsonl", "w", encoding="utf-8") as f:
        for i in range(args.n_docs):
            src = test_docs[int(rng.integers(len(test_docs)))]
            prompt = src["token_ids"][:PROMPT_LEN]
            plan = build_plan(rng, args.doc_len)
            tokens, labels = generate_doc(
                model, prompt, plan, device, gumbel, unigram_boost, kgw_proc,
                args.temperature, args.top_p, rng)
            did = f"synth_{i}"
            f.write(json.dumps({"document_id": did, "token_ids": tokens,
                                "labels": labels}) + "\n")
            kgw_out[did] = kgw_mask_scorer.score_document(tokens)
            ent_out[did] = doc_entropy(model, tokens, device)
            if (i + 1) % 10 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (args.n_docs - i - 1) / max(rate, 1e-9) / 60
                print(f"{i + 1}/{args.n_docs} docs ({rate:.2f} docs/s, "
                      f"ETA {eta:.0f} min)", flush=True)

    np.savez_compressed(out_dir / "kgw_synth.npz", **kgw_out)
    np.savez_compressed(out_dir / "entropy_synth.npz", **ent_out)
    with open(out_dir / "synth_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {args.n_docs} docs + kgw/entropy npz to {out_dir} "
          f"in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
