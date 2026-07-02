"""KGW (red-green list) watermark features, per token.

Kirchenbauer et al.: at each step a green list of size ``gamma * V`` is chosen by seeding
an RNG with a hash of the previous token(s), and green tokens get a logit boost.

**Critical correctness note (spec):** in this dataset the greenlists were generated with
``torch.randperm`` on a **CUDA (Philox)** generator. Recomputing on CPU gives effectively
random greenlists, so ~1/3 of KGW-watermarked tokens look unwatermarked. We therefore
build greenlists on the **GPU** when available (``cfg.kgw_use_cuda``). A CPU fallback
exists only so the code runs locally — it will NOT reproduce the real greenlists.

Features per token:
- ``green`` : 1 if token is in the (context-seeded) green list
- ``local_green_frac`` : green fraction over a local window
- ``running_z`` : normalised excess green count up to that position (KGW z-statistic)
"""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

from ..config import WatermarkConfig


def _kgw_key(cfg: WatermarkConfig) -> int:
    keys = cfg.keys or {}
    for name in ("kgw", "red_green", "redgreen"):
        if name in keys:
            val = keys[name]
            if isinstance(val, int):
                return val
            # KGW real config uses a string seeding_scheme, not a scalar key: hash it
            # to a stable int (approximate; exact KGW needs the vendor detector).
            return int.from_bytes(hashlib.sha256(str(val).encode()).digest()[:8], "little")
    return 0


def _seed_for_context(context: Sequence[int], key: int) -> int:
    h = hashlib.sha256()
    h.update(str(key).encode())
    for tok in context:
        h.update(int(tok).to_bytes(8, "little", signed=False))
    return int.from_bytes(h.digest()[:8], "little")


def _greenset_cuda(seed: int, vocab: int, green_size: int, device) -> set:
    import torch

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    perm = torch.randperm(vocab, generator=gen, device=device)
    return set(perm[:green_size].tolist())


def _greenset_cpu(seed: int, vocab: int, green_size: int) -> set:
    import torch

    gen = torch.Generator()  # CPU generator — does NOT match CUDA Philox
    gen.manual_seed(seed)
    perm = torch.randperm(vocab, generator=gen)
    return set(perm[:green_size].tolist())


def _greenlist(seed: int, vocab: int, green_size: int, device) -> "object":
    """Green token ids as a tensor on ``device`` (no host transfer).

    Same permutation as the ``_greenset_*`` helpers — we keep the ids on the GPU so
    membership can be tested with a GPU reduction instead of building a Python ``set``
    and transferring 38k ids to the host per context.
    """
    import torch

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    perm = torch.randperm(vocab, generator=gen, device=device)
    return perm[:green_size]


def kgw_features(token_ids: Sequence[int], cfg: WatermarkConfig,
                 window: int = 25) -> tuple[list[float], list[float], list[float]]:
    """Return (green 0/1, local green fraction, running z-score) per token.

    The greenlist membership test runs entirely on the GPU: for each token we do
    ``(greenlist == target).any()`` on-device and only copy the final ``green`` vector
    back to the host once per document. This is numerically identical to the previous
    Python-``set`` implementation (same permutation, same green size) but keeps the
    38k-element work on the GPU instead of the CPU.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001 - torch unavailable locally
        n = len(token_ids)
        return [0.0] * n, [0.0] * n, [0.0] * n

    key = _kgw_key(cfg)
    vocab = int(cfg.vocab_size)
    gamma = float(cfg.kgw_gamma)
    green_size = max(1, int(round(gamma * vocab)))
    k = cfg.context_width

    use_cuda = bool(cfg.kgw_use_cuda) and torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    if not use_cuda:
        print("[kgw] WARNING: CUDA unavailable — greenlists will NOT match the dataset "
              "(Philox). Run on a GPU for correct KGW features.")

    n = len(token_ids)
    toks = torch.as_tensor([int(t) for t in token_ids], device=device)
    green_t = torch.zeros(n, device=device)
    cache: dict[int, "object"] = {}  # seed -> greenlist id tensor on device
    for t in range(k, n):
        context = tuple(token_ids[t - k:t])
        seed = _seed_for_context(context, key)
        gl = cache.get(seed)
        if gl is None:
            gl = _greenlist(seed, vocab, green_size, device)
            cache[seed] = gl
        # Membership stays on-device; assigning a 0-dim bool tensor avoids a host sync.
        green_t[t] = (gl == toks[t]).any().to(green_t.dtype)
    green = green_t.detach().cpu().tolist()  # single GPU->host transfer per document

    # Local green fraction.
    frac = [0.0] * n
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        seg = green[lo:hi]
        frac[i] = sum(seg) / max(1, len(seg))

    # Running z-score of green count under H0 ~ Binomial(m, gamma).
    running_z = [0.0] * n
    green_count = 0
    for i in range(n):
        green_count += green[i]
        m = i + 1
        mean = gamma * m
        std = math.sqrt(max(1e-9, gamma * (1 - gamma) * m))
        running_z[i] = (green_count - mean) / std
    return green, frac, running_z
