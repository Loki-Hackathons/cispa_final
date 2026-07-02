"""Source separation for first-layer / fc1 gradients.

Two complementary recovery strategies, both label-free and leaderboard-free:

1. `isolated_recovery` (ReLU / trap-weight regime).
   The analytic row  r_i = gW_i / gb_i  equals a single private image x_b when
   exactly one sample activated neuron i (Boenisch eq. 6).  Many neurons isolate
   the *same* image, so those rows are (near) identical, while rows that mix two
   or more images are unique outliers.  We therefore cluster the rows and treat
   tight, populous clusters as high-confidence single-image reconstructions,
   averaging their members to denoise.  Cluster size + tightness is a far better
   'is this a real image' signal than the old per-row heuristics, and it stops us
   from filling the 128 slots with high-contrast mixture rows.

2. `image_subspace` (smooth-activation regime: sigmoid / tanh / gelu).
   Here every sample contributes to every neuron, so no row is clean.  But the
   weight-gradient matrix  G = A X  (A: neurons x B mixing, X: B x pixels images)
   has rank <= B, and its row space equals span{images}.  We expose that
   B-dimensional subspace via SVD.  NOTE: recovering the *individual* images
   inside it is blind source separation (rotation + scale ambiguous) and is not
   solved by box/TV priors alone — for the smooth models the reliable lever is
   exact-forward gradient matching (rebuild.gradient_match), warm-started from
   whatever the analytic path yields.  `image_subspace` is kept as a diagnostic
   / potential ICA hook, not a standalone reconstructor.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import config
import utils


# --------------------------------------------------------------------------- #
# 1. Isolated-image recovery via row clustering
# --------------------------------------------------------------------------- #
def analytic_rows(gW: torch.Tensor, gb: torch.Tensor,
                  eps: float = config.EPS) -> tuple[torch.Tensor, torch.Tensor]:
    """r_i = gW_i / gb_i for neurons with |gb_i| > eps.  Returns (rows, valid)."""
    valid = gb.abs() > eps
    if int(valid.sum()) == 0:
        return torch.empty(0, gW.shape[1]), valid
    rows = gW[valid] / gb[valid].unsqueeze(1)
    return rows, valid


def _greedy_cluster(fp: torch.Tensor, sim_threshold: float) -> list[list[int]]:
    """Greedy single-link-ish clustering on unit fingerprints (cosine).

    O(N^2) but N <= 1024 here.  Assigns each row to the first cluster whose
    seed is within `sim_threshold`, else starts a new cluster.  Order is by
    descending row norm so the most 'confident' rows seed clusters first.
    """
    n = fp.shape[0]
    seeds: list[int] = []
    members: list[list[int]] = []
    for idx in range(n):
        if seeds:
            sims = fp[torch.tensor(seeds)] @ fp[idx]
            j = int(torch.argmax(sims))
            if float(sims[j]) >= sim_threshold:
                members[j].append(idx)
                continue
        seeds.append(idx)
        members.append([idx])
    return members


def own_margin(rows: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Normalised self-activation margin  (w_i . r_i + b_i) / ||w_i||.

    For a truly isolated neuron, its analytic row r_i IS the single image that
    fired it, so feeding r_i back through the KNOWN weights reactivates that
    neuron strongly (large positive margin).  Mixture / spurious rows give small
    or negative margins.  This is the single best label-free 'is this a real
    single-image row' signal in our synthetic benchmark (bench_selection.py):
    +0.09 matched-SSIM over the old quality score in the heavy-mixture regime,
    and it uses only known weights -> no leaderboard, no overfit risk.
    """
    m = (W * rows).sum(dim=1) + b
    return m / W.norm(dim=1).clamp_min(1e-8)


@torch.no_grad()
def isolated_recovery(
    rows: torch.Tensor,
    feature_shape: tuple[int, int, int],
    to_rgb,
    sim_threshold: float = 0.90,
    own_active: torch.Tensor | None = None,
    row_priority: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cluster analytic rows into single-image reconstructions.

    Args:
      rows: (N, D) analytic rows (D == prod(feature_shape)).
      feature_shape: (C, H, W) to reshape rows before rgb mapping.
      to_rgb: callable(rows_subset) -> (M,3,64,64) in [0,1]  (family-specific).
      sim_threshold: cosine cut for merging rows into one image cluster.
      own_active: optional (N,) bool/float ReLU 'this is real' signal.
      row_priority: optional (N,) score (e.g. own_margin). When given it is the
        PRIMARY ranking (seed order + cluster confidence), which the synthetic
        benchmark shows beats norm/quality ranking on mixture-heavy models.

    Returns (images, confidence): images (K,3,64,64) one per cluster, sorted by
    descending confidence; confidence (K,) in ~[0,1].
    """
    n = rows.shape[0]
    if n == 0:
        return torch.empty(0, *config.IMG_SHAPE), torch.empty(0)

    c, h, w = feature_shape
    imgs = to_rgb(rows)                                  # (N,3,64,64) in [0,1]
    fp = utils._fingerprints(imgs)                       # (N,d) unit vectors

    # Seed order: prefer high own-margin rows (likely isolated) when available,
    # else the most energetic reconstructions.
    if row_priority is not None:
        order = torch.argsort(row_priority, descending=True)
        prio_n = (row_priority - row_priority.min()) / (
            row_priority.max() - row_priority.min() + 1e-8)
    else:
        order = torch.argsort(rows.reshape(n, -1).norm(dim=1), descending=True)
        prio_n = None
    fp_ord = fp[order]
    clusters_ord = _greedy_cluster(fp_ord, sim_threshold)
    clusters = [[int(order[j]) for j in cl] for cl in clusters_ord]

    reps: list[torch.Tensor] = []
    confs: list[float] = []
    quality = utils.quality_score(imgs)
    qn = (quality - quality.min()) / (quality.max() - quality.min() + 1e-8)

    for cl in clusters:
        members = torch.tensor(cl, dtype=torch.long)
        # Denoise: average the cluster's rgb reconstructions (identical for a
        # perfectly isolated image; averages out measurement noise otherwise).
        reps.append(imgs[members].mean(dim=0, keepdim=True))

        size = len(cl)
        if size > 1:
            cen = F.normalize(fp[members].mean(dim=0), dim=0)
            tight = float((fp[members] @ cen).clamp(-1, 1).mean())
        else:
            tight = 0.5
        size_score = min(1.0, size / 8.0)
        q = float(qn[members].mean())
        if prio_n is not None:
            # Own-margin dominates; size/tightness/quality are light tiebreaks.
            p = float(prio_n[members].max())
            conf = 0.7 * p + 0.15 * size_score + 0.1 * tight + 0.05 * q
        else:
            act = float(own_active[members].float().mean()) if own_active is not None else 1.0
            conf = 0.4 * size_score + 0.25 * tight + 0.2 * q + 0.15 * act
        confs.append(conf)

    images = torch.cat(reps, dim=0)
    confidence = torch.tensor(confs)
    o = torch.argsort(confidence, descending=True)
    return images[o].contiguous().float(), confidence[o].contiguous()


# --------------------------------------------------------------------------- #
# 2. Subspace unmixing for smooth activations
# --------------------------------------------------------------------------- #
@torch.no_grad()
def image_subspace(gW: torch.Tensor, rank: int = config.BATCH,
                   energy: float = 0.999) -> torch.Tensor:
    """Orthonormal basis (K, D) of the image span from the weight-gradient SVD.

    Rows of gW live in span{images}; the top right-singular vectors span the
    same subspace.  We keep min(rank, effective_rank(energy)) components.
    """
    gW = gW.float()
    # Economy SVD: gW = U diag(S) Vh, Vh rows are basis vectors in pixel space.
    U, S, Vh = torch.linalg.svd(gW, full_matrices=False)
    cum = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum().clamp_min(1e-12)
    k_energy = int(torch.searchsorted(cum, torch.tensor(energy)).item()) + 1
    k = max(1, min(rank, k_energy, Vh.shape[0]))
    return Vh[:k].contiguous()                           # (k, D)


def effective_rank(gW: torch.Tensor, energy: float = 0.999) -> int:
    """How many singular values hold `energy` of the gradient's mass.

    A rank far below B=128 on a ReLU model means many neurons never isolated an
    image (few sources leaked); rank ~= B on a smooth model means all images mix
    into a full-rank subspace (analytic rows are hopeless -> use gradient
    matching).  Purely diagnostic.
    """
    return image_subspace(gW, rank=gW.shape[0], energy=energy).shape[0]


# --------------------------------------------------------------------------- #
# Fill helpers: diverse plausible priors beat white noise for leftover slots.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def diversify_fill(recovered: torch.Tensor, k: int,
                   seed: int = config.SEED) -> torch.Tensor:
    """Return exactly k images, padding a short `recovered` set with augmented
    variants of its own best images (flips + mild brightness/gamma jitter).

    Duplicates never help the matched-SSIM metric, but a *natural-looking*
    augmented image can still match a different, similar ground-truth image far
    better than white noise (which scores ~0).  So we fill with diverse variants
    rather than noise whenever we have at least one real reconstruction.
    """
    n = recovered.shape[0]
    if n >= k:
        return recovered[:k].contiguous().float()
    if n == 0:
        # No signal at all -> smooth low-frequency noise (still >~ white noise).
        base = torch.rand(k, 3, 8, 8)
        return F.interpolate(base, size=(64, 64), mode="bilinear",
                             align_corners=False).clamp(0, 1).float()

    g = torch.Generator().manual_seed(seed)
    out = [recovered]
    for made in range(k - n):
        j = int(torch.randint(0, n, (1,), generator=g).item())
        x = recovered[j:j + 1].clone()
        if torch.rand((), generator=g) < 0.5:
            x = torch.flip(x, dims=[3])                  # horizontal flip
        gamma = float(torch.empty(()).uniform_(0.8, 1.25, generator=g))
        bright = float(torch.empty(()).uniform_(-0.06, 0.06, generator=g))
        x = (x.clamp(0, 1) ** gamma + bright).clamp(0, 1)
        out.append(x)
    return torch.cat(out, dim=0)[:k].contiguous().float()
