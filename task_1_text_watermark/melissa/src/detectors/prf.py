"""Shared pseudo-random function (PRF) for sampling-based watermarks.

Follows the Aaronson/Kirchner + Fernandez et al. construction used by TextSeal
(App. A.1): a deterministic hash of ``(candidate_token, context_window, secret_key)``
normalised to a uniform value in ``[0, 1]``. Without the real secret key it still
produces a stable, uniform-looking value so the rest of the pipeline runs; on the
cluster pass the real key from the dataset YAML.
"""

from __future__ import annotations

from typing import Sequence

# Large primes for hashing (bit dispersion). Kept explicit for reproducibility.
_P2 = 2_654_435_761
_P3 = 2_246_822_519
_P4 = 3_266_489_917
_PMIX = 0x9E3779B97F4A7C15
_MASK = (1 << 64) - 1
_CONTEXT_PRIMES = (
    5_915_587_277, 1_500_450_271, 3_267_000_013, 5_754_853_343, 4_093_082_899,
    9_576_890_767, 3_628_273_133, 2_860_486_313, 5_463_458_053, 3_367_900_313,
)


def _xorshift64(x: int) -> int:
    x = (x * _PMIX) & _MASK
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & _MASK
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & _MASK
    x ^= x >> 31
    return x & _MASK


def seed_from_context(context: Sequence[int], key: int) -> int:
    """Deterministic 64-bit seed from a context window + secret key."""
    acc = (_P3 * int(key)) & _MASK
    for i, tok in enumerate(context):
        p = _CONTEXT_PRIMES[i % len(_CONTEXT_PRIMES)]
        acc = (acc + int(tok) * p) & _MASK
    return _xorshift64((acc * _P4) & _MASK)


def prf_uniform(token: int, context: Sequence[int], key: int) -> float:
    """Uniform ``R_v = PRF(token, context, key) ∈ [0, 1)`` for a candidate token."""
    h = (_P2 * int(token) + seed_from_context(context, key)) & _MASK
    h = _xorshift64(h)
    return (h & ((1 << 53) - 1)) / float(1 << 53)
