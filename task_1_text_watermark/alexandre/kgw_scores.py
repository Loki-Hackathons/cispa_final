"""Precompute KGW green/red masks on GPU (CUDA Philox required).

Replicates the vendor detector (lm-watermarking commit 8292251) with
seeding scheme ff-anchored_minhash_prf-4-True-1306382177 (self_salt=True):
ngrams of length context_width + 1 - self_salt = 4 tokens; the seeding
window is the whole 4-gram INCLUDING the target token (Algorithm 3 self-hash),
and we check whether the last token of the 4-gram is in its own greenlist.

Greenlist generation uses torch.randperm on the CUDA generator — matching
the organizers' generation pipeline. Running this on CPU produces garbage.

Output: one .npz per split mapping document_id -> uint8 mask (1 = green).
Positions t < context_width get 0.25 (H0 mean, uninformative).

Usage (on JURECA GPU node):
    python kgw_scores.py --data-dir <dataset dir> --out-dir <dir> --splits train validation test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "lm-watermarking"
sys.path.insert(0, str(_VENDOR))

from alternative_prf_schemes import prf_lookup, seeding_scheme_lookup  # noqa: E402

SEEDING_SCHEME = "ff-anchored_minhash_prf-4-True-1306382177"
GAMMA = 0.25
VOCAB_SIZE = 151936  # Qwen2.5-7B-Instruct model config vocab size


class KgwMaskScorer:
    def __init__(self, vocab_size: int = VOCAB_SIZE, device: str = "cuda"):
        self.prf_type, self.context_width, self.self_salt, self.hash_key = \
            seeding_scheme_lookup(SEEDING_SCHEME)
        self.vocab_size = vocab_size
        self.device = torch.device(device)
        self.rng = torch.Generator(device=self.device)
        self.greenlist_size = int(vocab_size * GAMMA)
        self.cache: dict[tuple, bool] = {}
        # ngram length as in vendor _score_ngrams_in_passage
        self.ngram_len = self.context_width + 1 - int(self.self_salt)

    def _is_green(self, ngram: tuple[int, ...]) -> bool:
        """ngram: seeding window; with self_salt the target IS the last element."""
        if ngram not in self.cache:
            seed_window = ngram if self.self_salt else ngram[:-1]
            target = ngram[-1]
            ids = torch.as_tensor(seed_window, device=self.device)
            prf_key = prf_lookup[self.prf_type](ids, salt_key=self.hash_key)
            self.rng.manual_seed(prf_key % (2**64 - 1))
            perm = torch.randperm(self.vocab_size, device=self.device, generator=self.rng)
            green = perm[: self.greenlist_size]
            self.cache[ngram] = bool((green == target).any().item())
        return self.cache[ngram]

    def score_document(self, token_ids: list[int]) -> np.ndarray:
        n = len(token_ids)
        out = np.full(n, GAMMA, dtype=np.float32)
        L = self.ngram_len
        for t in range(L - 1, n):
            ngram = tuple(token_ids[t - L + 1:t + 1])
            out[t] = 1.0 if self._is_green(ngram) else 0.0
        return out


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def probe_vocab_sizes(data_dir: Path, device: str, n_docs: int = 30) -> int:
    """Compare candidate vocab sizes on labeled train docs: green fraction
    in watermarked vs clean tokens. The right size shows wm >> gamma.
    Returns the best vocab size by separation z."""
    records = read_jsonl(data_dir / "train.jsonl")[:n_docs]
    best_vs, best_z = None, -np.inf
    for vs in (151643, 151646, 151665, 151936, 152064):
        scorer = KgwMaskScorer(vocab_size=vs, device=device)
        wm, cl = [], []
        for rec in records:
            mask = scorer.score_document(rec["token_ids"])
            lab = np.array(rec["labels"])
            valid = np.arange(len(lab)) >= scorer.ngram_len - 1
            wm.extend(mask[(lab == 1) & valid])
            cl.extend(mask[(lab == 0) & valid])
        wm, cl = np.array(wm), np.array(cl)
        z = (wm.mean() - cl.mean()) / np.sqrt(GAMMA * (1 - GAMMA) * (1 / len(wm) + 1 / len(cl)))
        print(f"vocab={vs}: wm green={wm.mean():.4f} clean green={cl.mean():.4f} z={z:.1f}",
              flush=True)
        if z > best_z:
            best_vs, best_z = vs, z
    print(f"best vocab size: {best_vs} (z={best_z:.1f})", flush=True)
    return best_vs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--probe", action="store_true",
                        help="probe candidate vocab sizes on labeled train docs and exit")
    parser.add_argument("--auto", action="store_true",
                        help="probe first, then run all splits with the best vocab size")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA unavailable — KGW greenlists require CUDA Philox, aborting.")

    vocab_size = args.vocab_size
    if args.probe or args.auto:
        vocab_size = probe_vocab_sizes(Path(args.data_dir), args.device)
        if args.probe:
            return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scorer = KgwMaskScorer(vocab_size=vocab_size, device=args.device)

    for split in args.splits:
        records = read_jsonl(Path(args.data_dir) / f"{split}.jsonl")
        result = {}
        t0 = time.time()
        for i, rec in enumerate(records):
            result[rec["document_id"]] = scorer.score_document(rec["token_ids"])
            if (i + 1) % 20 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"{split}: {i + 1}/{len(records)} docs ({rate:.1f} docs/s, "
                      f"cache={len(scorer.cache)})", flush=True)
        np.savez_compressed(out_dir / f"kgw_{split}.npz", **result)
        print(f"{split}: wrote {out_dir / f'kgw_{split}.npz'} "
              f"({len(records)} docs, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
