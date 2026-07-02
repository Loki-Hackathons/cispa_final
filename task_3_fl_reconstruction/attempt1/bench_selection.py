"""Objectively benchmark row-SELECTION strategies on synthetic trap gradients.

The whole game for the trap models is: from ~1000 analytic rows (a mix of clean
single-image reconstructions, blurry multi-image mixtures, and near-dead rows),
pick the 128 that maximise the one-to-one matched SSIM.  We cannot see the real
ground truth, so we settle this on SYNTHETIC data where we DO know the images and
exactly which neurons isolated a single image.

This never touches the leaderboard and cannot overfit the 30% public split: it
only compares scoring functions on controllable synthetic isolation, and we bake
in whichever wins.

Run:  python bench_selection.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
import utils

torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))


def _log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Fast batched SSIM matrix (no Python loop over rows)
# --------------------------------------------------------------------------- #
def _gauss_win(size=11, sigma=1.5):
    c = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g)


@torch.no_grad()
def ssim_matrix_fast(a: torch.Tensor, b: torch.Tensor, chunk: int = 64):
    """(Na,Nb) SSIM between grayscale-reduced a,b in [0,1]. Batched via conv."""
    def gray(x):
        return x.mean(dim=1, keepdim=True).float().clamp(0, 1)
    A, B = gray(a), gray(b)                               # (Na,1,H,W),(Nb,1,H,W)
    win = _gauss_win().to(A)[None, None]
    pad = win.shape[-1] // 2

    def stats(x):
        mu = F.conv2d(x, win, padding=pad)
        mu2 = mu * mu
        s = F.conv2d(x * x, win, padding=pad) - mu2
        return mu, mu2, s
    muA, muA2, sA = stats(A)
    muB, muB2, sB = stats(B)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    Na, Nb = A.shape[0], B.shape[0]
    out = torch.zeros(Na, Nb)
    for i in range(0, Na, chunk):
        ai = A[i:i + chunk]                               # (ca,1,H,W)
        ca = ai.shape[0]
        # cross terms: mu_a*mu_b and E[a*b] for every (i,j)
        # expand to (ca,Nb,H,W)
        muab = muA[i:i + chunk].unsqueeze(1) * muB.unsqueeze(0)      # (ca,Nb,1,H,W)? 
        muab = (muA[i:i + chunk].squeeze(1).unsqueeze(1)
                * muB.squeeze(1).unsqueeze(0))            # (ca,Nb,H,W)
        # E[a*b] via conv of products — compute per chunk pair
        aexp = ai.squeeze(1).unsqueeze(1)                 # (ca,1,H,W)
        bexp = B.squeeze(1).unsqueeze(0)                  # (1,Nb,H,W)
        prod = (aexp * bexp).reshape(ca * Nb, 1, *A.shape[-2:])
        Eab = F.conv2d(prod, win, padding=pad).reshape(ca, Nb, *A.shape[-2:])
        sab = Eab - muab
        muA2c = muA2[i:i + chunk].squeeze(1).unsqueeze(1)  # (ca,1,H,W)
        sAc = sA[i:i + chunk].squeeze(1).unsqueeze(1)
        muB2e = muB2.squeeze(1).unsqueeze(0)
        sBe = sB.squeeze(1).unsqueeze(0)
        smap = ((2 * muab + c1) * (2 * sab + c2)) / \
               ((muA2c + muB2e + c1) * (sAc + sBe + c2))
        out[i:i + chunk] = smap.mean(dim=(2, 3))
    return out


# --------------------------------------------------------------------------- #
# Synthetic trap MLP (Boenisch Algorithm 1) -> realistic PARTIAL isolation
# --------------------------------------------------------------------------- #
def trap_row_init(out_f: int, in_f: int, scale: float, seed: int) -> torch.Tensor:
    """Algorithm 1: half components negative |N|, half positive = -scale*neg.

    Negatives dominate (scale<1) so w.x is usually negative -> ReLU rarely fires
    -> a neuron tends to activate for at most a few batch items (sparse), which
    is what makes single-image leakage happen.  Vectorised (no per-row loop).
    """
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(out_f, in_f, generator=g).abs()
    # Random half of each row negative: sort a per-row random key, threshold.
    key = torch.rand(out_f, in_f, generator=g)
    thresh = key.median(dim=1, keepdim=True).values
    neg = key <= thresh
    w = torch.where(neg, -z, scale * z)
    return w


def natural_batch(n: int, seed: int) -> torch.Tensor:
    """Low-frequency [0,1] images (natural-ish statistics: smooth + some edges)."""
    g = torch.Generator().manual_seed(seed)
    low = F.interpolate(torch.rand(n, 3, 8, 8, generator=g), size=(64, 64),
                        mode="bilinear", align_corners=False)
    mid = F.interpolate(torch.rand(n, 3, 16, 16, generator=g), size=(64, 64),
                        mode="bilinear", align_corners=False)
    x = (0.7 * low + 0.3 * mid).clamp(0, 1)
    return x


def build_synthetic(offset: float = 2.3, seed: int = 0):
    """Return dict with rows/images/is_clean/own_active/own_margin for a trap MLP.

    `offset` controls sparsity: bias_i = -offset * std(preact_i), so larger
    offset => neurons fire for fewer images => more genuinely isolated rows.
    This produces a realistic MIX (isolated + mixtures + dead), unlike forcing a
    fixed activation quantile.
    """
    x = natural_batch(config.BATCH, seed=seed + 7)
    W0 = trap_row_init(1024, config.IMG_FLAT, scale=0.10, seed=seed + 1)
    raw = x.flatten(1) @ W0.t()                          # (B, 1024)
    b0 = -(raw.mean(0) + offset * raw.std(0))            # per-neuron sparse bias
    pre = raw + b0

    net = nn.Sequential(nn.Linear(config.IMG_FLAT, 1024), nn.ReLU(),
                        nn.Linear(1024, 200))
    with torch.no_grad():
        net[0].weight.copy_(W0)
        net[0].bias.copy_(b0)
    y = torch.randint(0, 200, (config.BATCH,))
    loss = F.cross_entropy(net(x.flatten(1)), y)
    gW, gb = torch.autograd.grad(loss, [net[0].weight, net[0].bias])

    valid = gb.abs() > config.EPS
    rows = gW[valid] / gb[valid].unsqueeze(1)

    n_active = (pre > 0).sum(dim=0)[valid]              # images per neuron
    is_clean = (n_active == 1)
    W0v, b0v = W0[valid], b0[valid]
    margin = ((W0v * rows).sum(dim=1) + b0v)            # own pre-activation
    own_active = margin > 0
    own_margin = margin / W0v.norm(dim=1).clamp_min(1e-8)
    return {"rows": rows, "x": x, "is_clean": is_clean,
            "own_active": own_active, "own_margin": own_margin,
            "n_active": n_active}


# --------------------------------------------------------------------------- #
# Scoring strategies (all label-free; usable at attack time)
# --------------------------------------------------------------------------- #
def rows_to_img(rows: torch.Tensor) -> torch.Tensor:
    x = rows.reshape(rows.shape[0], *config.IMG_SHAPE)
    return x.clamp(0, 1).float()


def score_quality(imgs, rows, own):
    return utils.quality_score(imgs)


def score_sharpness(imgs, rows, own):
    """High-frequency energy: mixtures are blurry (low), clean images sharper."""
    return utils.total_variation(imgs) * imgs.reshape(imgs.shape[0], -1).std(dim=1)


def score_rownorm(imgs, rows, own):
    return rows.reshape(rows.shape[0], -1).norm(dim=1)


def score_own(imgs, rows, own):
    q = utils.quality_score(imgs)
    qn = (q - q.min()) / (q.max() - q.min() + 1e-8)
    return own.float() * 10.0 + qn


def score_own_sharp(imgs, rows, own):
    s = score_sharpness(imgs, rows, own)
    sn = (s - s.min()) / (s.max() - s.min() + 1e-8)
    return own.float() * 10.0 + sn


def _norm01(t):
    return (t - t.min()) / (t.max() - t.min() + 1e-8)


STRATEGIES = {
    "quality(current)": lambda d, imgs: score_quality(imgs, d["rows"], d["own_active"]),
    "sharpness": lambda d, imgs: score_sharpness(imgs, d["rows"], d["own_active"]),
    "own_active+quality": lambda d, imgs: score_own(imgs, d["rows"], d["own_active"]),
    "own_margin": lambda d, imgs: d["own_margin"],
    "own_margin+quality": lambda d, imgs: _norm01(d["own_margin"]) + 0.5 * _norm01(
        utils.quality_score(imgs)),
    "margin*sharp": lambda d, imgs: _norm01(d["own_margin"]) * _norm01(
        score_sharpness(imgs, d["rows"], d["own_active"])),
}


# --------------------------------------------------------------------------- #
def dedup_pick(imgs, scores, k=config.BATCH, sim=0.92):
    fp = utils._fingerprints(imgs)
    order = torch.argsort(scores, descending=True)
    chosen, chosen_fp = [], []
    for idx in order.tolist():
        if len(chosen) >= k:
            break
        if chosen_fp:
            if float((torch.stack(chosen_fp) @ fp[idx]).max()) > sim:
                continue
        chosen.append(idx)
        chosen_fp.append(fp[idx])
    if len(chosen) < k:
        for idx in order.tolist():
            if idx not in set(chosen):
                chosen.append(idx)
            if len(chosen) >= k:
                break
    return torch.tensor(chosen[:k])


def matched_ssim_from(M_sel: torch.Tensor) -> float:
    """Greedy one-to-one matched SSIM from a (k, n_gt) precomputed SSIM matrix."""
    total, used = 0.0, set()
    best_vals = M_sel.max(dim=1).values
    for i in torch.argsort(best_vals, descending=True).tolist():
        row = M_sel[i].clone()
        for j in used:
            row[j] = -1
        j = int(row.argmax())
        used.add(j)
        total += float(M_sel[i, j])
    return total / M_sel.shape[0]


def main():
    _log("Synthetic trap-MLP selection benchmark (higher matched-SSIM = better)\n")
    for offset in (1.5, 2.3, 3.0):
        d = build_synthetic(offset=offset, seed=0)
        rows, x, is_clean = d["rows"], d["x"], d["is_clean"]
        imgs = rows_to_img(rows)
        n_clean = int(is_clean.sum())
        _log(f"--- offset={offset}: {rows.shape[0]} valid rows, "
             f"{n_clean} truly isolated (single-image) ---")

        M = ssim_matrix_fast(imgs, x)                    # (Nvalid, 128)

        oracle_idx = dedup_pick(imgs, M.max(dim=1).values)
        _log(f"  {'ORACLE (best rows)':24s}: {matched_ssim_from(M[oracle_idx]):.3f}")

        if n_clean > 0:
            clean_idx = torch.nonzero(is_clean, as_tuple=False).squeeze(1)
            if clean_idx.numel() < config.BATCH:
                extra = torch.nonzero(~is_clean, as_tuple=False).squeeze(1)
                clean_idx = torch.cat([clean_idx,
                                       extra[: config.BATCH - clean_idx.numel()]])
            _log(f"  {'clean-rows(oracle gate)':24s}: "
                 f"{matched_ssim_from(M[clean_idx[:config.BATCH]]):.3f}")

        for name, fn in STRATEGIES.items():
            sc = fn(d, imgs)
            idx = dedup_pick(imgs, sc)
            _log(f"  {name:24s}: {matched_ssim_from(M[idx]):.3f}")
        _log("")


if __name__ == "__main__":
    main()
