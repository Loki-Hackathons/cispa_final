"""FC1 analytic extraction with per-row isolation confidence.

When several batch images activate the same fc1 neuron, the Boenisch identity
recovers a weighted average of their feature maps, not a single clean target.
This module scores how likely each analytic row came from one image, so we can
prefer clean rows before CNN inversion and flag models (e.g. model 6) whose
targets are mostly contaminated.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

import config
import utils


@dataclass
class Fc1Candidates:
    """All valid fc1 analytic rows for one CNN model."""

    model_idx: int
    activation: str
    feature_shape: tuple[int, int, int]
    feats: torch.Tensor          # (N, C, H, W) raw fc1 inputs
    preview: torch.Tensor        # (N, 3, 64, 64) RGB preview
    quality: torch.Tensor        # (N,) visual proxy
    confidence: torch.Tensor     # (N,) isolation confidence in [0, 1]
    gb_abs: torch.Tensor         # (N,) |dL/db_i| for each row
    components: dict[str, torch.Tensor]  # per-metric breakdown for diagnostics


def _reshape_fc1_rows(grad: dict, rows: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
    gr = grad["gradients"]
    c, h, w = tuple(grad["feature_shape"])
    if c * h * w != rows.shape[1]:
        c = gr["conv.weight"].shape[0]
        h, w = utils.infer_square(rows.shape[1], c)
        if c * h * w != rows.shape[1]:
            raise RuntimeError(
                f"cannot reshape fc1 rows {rows.shape[1]} into a feature map"
            )
    return rows.reshape(rows.shape[0], c, h, w).float(), c, h, w


def _row_uniqueness(feats: torch.Tensor) -> torch.Tensor:
    """1 - max cosine similarity to any other row (higher = more isolated)."""
    n = feats.shape[0]
    if n <= 1:
        return torch.ones(n)
    flat = feats.reshape(n, -1).float()
    flat = F.normalize(flat, dim=1)
    sim = flat @ flat.t()
    sim.fill_diagonal_(-1.0)
    return (1.0 - sim.max(dim=1).values).clamp(0, 1)


def _feature_sparsity(feats: torch.Tensor, activation: str) -> torch.Tensor:
    """Fraction of near-zero activations; high sparsity often means a cleaner ReLU map."""
    n = feats.shape[0]
    x = feats.reshape(n, -1).float()
    if activation == "relu":
        peak = x.max(dim=1).values.clamp_min(1e-8)
        return (x < 0.05 * peak.unsqueeze(1)).float().mean(dim=1)
    # tanh/sigmoid: measure dynamic range instead
    lo = x.min(dim=1).values
    hi = x.max(dim=1).values
    span = (hi - lo).clamp_min(1e-8)
    return ((x - lo.unsqueeze(1)) / span.unsqueeze(1) > 0.9).float().mean(dim=1)


def _gb_sweetness(gb_abs: torch.Tensor) -> torch.Tensor:
    """Penalize tiny |gb| (noisy) and very large |gb| (many summed contributors)."""
    if gb_abs.numel() == 0:
        return gb_abs
    logg = torch.log(gb_abs.clamp_min(config.EPS))
    lo, hi = float(logg.min()), float(logg.max())
    if hi - lo < 1e-6:
        return torch.ones_like(gb_abs)
    norm = (logg - lo) / (hi - lo)
    # Peak trust around the 35th-65th percentile band.
    return (1.0 - (norm - 0.5).abs() * 2.0).clamp(0, 1)


def _norm_sanity(feats: torch.Tensor, gb_abs: torch.Tensor) -> torch.Tensor:
    """||gW|| / |gb| = ||x|| for isolated rows; down-rank extreme norms."""
    flat = feats.reshape(feats.shape[0], -1).float()
    norms = flat.norm(dim=1) / gb_abs.clamp_min(config.EPS)
    if norms.numel() <= 1:
        return torch.ones_like(norms)
    logn = torch.log(norms.clamp_min(1e-8))
    med = float(logn.median())
    mad = float((logn - med).abs().median().clamp_min(1e-3))
    z = ((logn - med).abs() / (3.0 * mad)).clamp(0, 1)
    return 1.0 - z


def isolation_confidence(
    feats: torch.Tensor,
    gb_abs: torch.Tensor,
    activation: str,
    quality: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-row isolation confidence in [0, 1] plus component scores."""
    uniqueness = _row_uniqueness(feats)
    sparsity = _feature_sparsity(feats, activation)
    gb_sweet = _gb_sweetness(gb_abs)
    norm_ok = _norm_sanity(feats, gb_abs)

    if quality is None:
        preview = utils.features_to_image(
            feats.reshape(feats.shape[0], -1),
            feats.shape[1], feats.shape[2], feats.shape[3],
        )
        quality = utils.quality_score(preview)
    q_norm = (quality - quality.min()) / (quality.max() - quality.min() + 1e-8)

    # Weights favor uniqueness + gb band; quality is a mild tie-breaker only.
    conf = (
        0.35 * uniqueness
        + 0.25 * sparsity
        + 0.25 * gb_sweet
        + 0.10 * norm_ok
        + 0.05 * q_norm
    ).clamp(0, 1)
    return conf, {
        "uniqueness": uniqueness,
        "sparsity": sparsity,
        "gb_sweet": gb_sweet,
        "norm_ok": norm_ok,
        "quality_norm": q_norm,
    }


def extract_fc1_candidates(model_idx: int) -> Fc1Candidates:
    """Extract all valid fc1 analytic rows with confidence scores."""
    grad = utils.load_gradient(model_idx)
    gr = grad["gradients"]
    gW, gb = gr["fc1.weight"], gr["fc1.bias"]
    valid = gb.abs() > config.EPS
    if valid.sum() == 0:
        raise RuntimeError(f"model{model_idx}: no fc1 analytic rows")

    rows = gW[valid] / gb[valid].unsqueeze(1)
    feats, c, h, w = _reshape_fc1_rows(grad, rows)
    preview = utils.features_to_image(rows, c, h, w)
    quality = utils.quality_score(preview)
    gb_abs = gb[valid].abs().float()
    confidence, components = isolation_confidence(
        feats, gb_abs, grad["activation"], quality=quality,
    )
    return Fc1Candidates(
        model_idx=model_idx,
        activation=grad["activation"],
        feature_shape=(c, h, w),
        feats=feats,
        preview=preview,
        quality=quality,
        confidence=confidence,
        gb_abs=gb_abs,
        components=components,
    )


def combined_score(
    quality: torch.Tensor,
    confidence: torch.Tensor,
    confidence_weight: float = 0.55,
) -> torch.Tensor:
    """Blend visual proxy with isolation confidence for candidate ranking."""
    w = float(confidence_weight)
    q = (quality - quality.min()) / (quality.max() - quality.min() + 1e-8)
    return (1.0 - w) * q + w * confidence


def select_fc1_candidates(
    cands: Fc1Candidates,
    k: int = config.BATCH,
    use_confidence: bool = True,
    confidence_weight: float = 0.55,
    sim_threshold: float = config.DEDUP_SIM_THRESHOLD,
) -> torch.Tensor:
    """Pick k distinct row indices; returns LongTensor of shape (k,)."""
    if use_confidence:
        scores = combined_score(cands.quality, cands.confidence, confidence_weight)
    else:
        scores = cands.quality

    fp = utils._fingerprints(cands.preview)
    order = torch.argsort(scores, descending=True)
    chosen: list[int] = []
    chosen_fp: list[torch.Tensor] = []
    for idx in order.tolist():
        if len(chosen) >= k:
            break
        if chosen_fp:
            sims = torch.stack(chosen_fp) @ fp[idx]
            if float(sims.max()) > sim_threshold:
                continue
        chosen.append(idx)
        chosen_fp.append(fp[idx])

    if len(chosen) < k:
        chosen_set = set(chosen)
        for idx in order.tolist():
            if idx not in chosen_set:
                chosen.append(idx)
                chosen_set.add(idx)
            if len(chosen) >= k:
                break
    return torch.tensor(chosen[:k], dtype=torch.long)


def model_confidence_report(cands: Fc1Candidates, k: int = config.BATCH) -> dict:
    """Summary stats for one model; useful to spot contamination before GPU work."""
    conf = cands.confidence
    sel = select_fc1_candidates(cands, k=k, use_confidence=True)
    sel_conf = conf[sel]
    sel_q = cands.quality[sel]
    return {
        "model": cands.model_idx,
        "activation": cands.activation,
        "feature_shape": cands.feature_shape,
        "n_candidates": int(conf.numel()),
        "conf_mean_all": float(conf.mean()),
        "conf_median_all": float(conf.median()),
        "conf_p90_all": float(torch.quantile(conf, 0.9)),
        "conf_high_frac": float((conf > 0.5).float().mean()),
        "conf_mean_selected": float(sel_conf.mean()),
        "conf_min_selected": float(sel_conf.min()),
        "quality_mean_selected": float(sel_q.mean()),
        "selected_indices": sel,
    }


def print_confidence_table(model_ids: list[int], k: int = config.BATCH) -> None:
    """Print a compact comparison table for several CNN models."""
    print(
        f"{'model':>5} | {'act':7} | {'shape':>11} | {'n':>5} | "
        f"{'conf_all':>8} | {'conf_sel':>8} | {'high%':>5} | {'q_sel':>6}"
    )
    print("-" * 78)
    for i in model_ids:
        try:
            cands = extract_fc1_candidates(i)
            rep = model_confidence_report(cands, k=k)
            sh = f"{rep['feature_shape'][1]}x{rep['feature_shape'][2]}"
            print(
                f"{i:>5} | {rep['activation']:7} | {sh:>11} | "
                f"{rep['n_candidates']:>5} | {rep['conf_mean_all']:8.3f} | "
                f"{rep['conf_mean_selected']:8.3f} | "
                f"{100 * rep['conf_high_frac']:4.0f}% | "
                f"{rep['quality_mean_selected']:6.3f}"
            )
        except Exception as e:
            print(f"{i:>5} | ERROR: {e}")
