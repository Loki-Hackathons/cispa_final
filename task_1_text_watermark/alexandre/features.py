"""Per-token feature matrix for the Task 1 calibrator.

For each scheme signal (H0-standardized increments), emit multi-scale window
z-scores anchored at the token: centered, left-only, and right-only windows.
Left/right variants let the classifier resolve span boundaries instead of
bleeding high scores onto neighboring clean tokens.
"""

from __future__ import annotations

import numpy as np

from detectors import H0_MOMENTS, compute_signals

CENTERED_WINDOWS = (8, 16, 32, 64, 128)
SIDE_WINDOWS = (16, 32, 64)

SCHEMES = ("textseal", "gumbelmax", "unigram", "kgw")


def _window_z(centered: np.ndarray, sigma0: float, w: int, mode: str) -> np.ndarray:
    """Window z-score per token. mode: 'c' centered, 'l' ends at token, 'r' starts at token."""
    n = len(centered)
    w = min(w, n)
    cumsum = np.concatenate([[0.0], np.cumsum(centered)])
    idx = np.arange(n)
    if mode == "c":
        starts = np.clip(idx - w // 2, 0, n - w)
    elif mode == "l":
        starts = np.clip(idx - w + 1, 0, n - w)
    else:
        starts = np.clip(idx, 0, n - w)
    sums = cumsum[starts + w] - cumsum[starts]
    return sums / (sigma0 * np.sqrt(w))


def doc_features(token_ids: list[int], kgw_signal: np.ndarray | None = None) -> np.ndarray:
    """Feature matrix (n_tokens, n_features)."""
    signals = compute_signals(token_ids)
    if kgw_signal is not None:
        signals["kgw"] = kgw_signal

    cols = []
    for name in SCHEMES:
        if name not in signals:
            continue
        mu0, var0 = H0_MOMENTS[name]
        sigma0 = np.sqrt(var0)
        centered = signals[name] - mu0
        cols.append(centered / sigma0)  # own-token standardized increment
        for w in CENTERED_WINDOWS:
            cols.append(_window_z(centered, sigma0, w, "c"))
        for w in SIDE_WINDOWS:
            cols.append(_window_z(centered, sigma0, w, "l"))
            cols.append(_window_z(centered, sigma0, w, "r"))
    return np.column_stack(cols)


def feature_names(with_kgw: bool) -> list[str]:
    names = []
    for scheme in SCHEMES:
        if scheme == "kgw" and not with_kgw:
            continue
        names.append(f"{scheme}_own")
        names += [f"{scheme}_c{w}" for w in CENTERED_WINDOWS]
        for w in SIDE_WINDOWS:
            names += [f"{scheme}_l{w}", f"{scheme}_r{w}"]
    return names
