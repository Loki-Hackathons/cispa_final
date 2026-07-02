"""Correct, vendor-based per-token features for Task 1.

Melissa's original detectors used a home-grown PRF that does **not** reproduce the
dataset's watermark signals, so the calibrator was learning from noise. This module
instead reuses Alexandre's already-tested implementation, which loads the pinned vendor
repositories (``textseal``, ``lm-watermarking``, ``unigram-watermark``) to reproduce the
exact PRF / greenlist math used to generate the dataset:

  * ``detectors.compute_signals`` â€” TextSeal (dual-key Gumbel), Gumbel-Max, Unigram.
  * ``features.doc_features``     â€” multi-scale windowed z-scores (centered + left +
    right) per scheme; these are the features that separate spans at 0.1 % FPR.
  * ``kgw_scores.KgwMaskScorer``  â€” correct KGW greenlists via CUDA Philox
    (``torch.randperm`` on a CUDA generator). A single scorer instance is reused so its
    greenlist cache is shared **across documents** (contexts repeat), which is both
    faster and exactly what the organizers used.

The vendor code lives in ``task_1_text_watermark/vendor/`` and is only present after
``scripts/task1/sync_watermark_repos.sh`` has been run on a login node (needs network).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

# Alexandre's tested modules (they add the vendor repos to sys.path themselves).
_ALEX_DIR = Path(__file__).resolve().parents[2] / "alexandre"
_VENDOR_DIR = Path(__file__).resolve().parents[2] / "vendor"

if str(_ALEX_DIR) not in sys.path:
    sys.path.insert(0, str(_ALEX_DIR))

try:
    from detectors import H0_MOMENTS, compute_signals  # noqa: F401  (re-exported)
    from features import doc_features, feature_names
    from kgw_scores import KgwMaskScorer, VOCAB_SIZE
except Exception as exc:  # noqa: BLE001
    raise ImportError(
        "Could not import the vendor-based detectors. The pinned watermark repos must be "
        f"synced first (they were not found under {_VENDOR_DIR}).\n"
        "Run on a JURECA *login* node (needs network):\n"
        "    bash scripts/task1/sync_watermark_repos.sh\n"
        f"Original error: {exc}"
    ) from exc


# Number of feature columns depends only on whether KGW is included.
def n_features(with_kgw: bool = True) -> int:
    return len(feature_names(with_kgw=with_kgw))


_KGW_STATE: dict = {}


def _kgw_scorer() -> "KgwMaskScorer | None":
    """Singleton KGW scorer (shared greenlist cache across documents).

    Uses CUDA if available (required to match the dataset's Philox greenlists). By
    default, CPU KGW is disabled because it is not just slower: it produces different
    greenlists and can poison the high-score tail at 0.1% FPR. Set
    `WML_ALLOW_CPU_KGW=1` only for a local smoke test.
    """
    if "scorer" in _KGW_STATE:
        return _KGW_STATE["scorer"]
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            if os.environ.get("WML_ALLOW_CPU_KGW", "0") != "1":
                print("[kgw] CUDA unavailable; KGW disabled to avoid non-Philox false "
                      "signals. Run on a GPU node for correct KGW features.")
                scorer = None
            else:
                print("[kgw] WARNING: CPU KGW enabled for smoke test only; greenlists "
                      "will NOT match the dataset.")
                scorer = KgwMaskScorer(vocab_size=VOCAB_SIZE, device=device)
        else:
            scorer = KgwMaskScorer(vocab_size=VOCAB_SIZE, device=device)
    except Exception as exc:  # noqa: BLE001
        print(f"[kgw] disabled ({exc}); features will omit KGW.")
        scorer = None
    _KGW_STATE["scorer"] = scorer
    return scorer


def doc_feature_matrix(token_ids: Sequence[int], use_kgw: bool = True) -> np.ndarray:
    """Return the (n_tokens, n_features) matrix of correct, standardized signals.

    ``use_kgw`` must be consistent between training and prediction so the column count
    matches. On a GPU node KGW is always included; if the scorer cannot be built the
    matrix silently omits the KGW columns (and ``use_kgw`` effectively becomes False).
    """
    ids = [int(t) for t in token_ids]
    kgw_signal = None
    if use_kgw:
        scorer = _kgw_scorer()
        if scorer is not None:
            kgw_signal = np.asarray(scorer.score_document(ids), dtype=np.float64)
    return doc_features(ids, kgw_signal)


def kgw_available() -> bool:
    return _kgw_scorer() is not None


def kgw_signal(token_ids: Sequence[int]) -> "np.ndarray | None":
    """Correct per-token KGW green signal (Philox), or None if KGW is unavailable.

    Reuses the shared scorer so its greenlist cache is warmed across documents. Used by
    the HMM head, which consumes the raw signal (the calibrator head gets it through the
    windowed-z feature matrix instead).
    """
    scorer = _kgw_scorer()
    if scorer is None:
        return None
    return np.asarray(scorer.score_document([int(t) for t in token_ids]), dtype=np.float64)

