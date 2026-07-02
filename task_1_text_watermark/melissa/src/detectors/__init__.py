"""Per-token watermark detectors (Gumbel-Max, TextSeal, Unigram, KGW) + entropy.

Each detector maps a document's ``token_ids`` to a per-token float aligned 1:1 with the
tokens. Detectors read secret keys / params from ``config.WatermarkConfig`` (populated
from the dataset YAML on the cluster). Where a key is missing they degrade gracefully to
a deterministic key-free PRF so the pipeline still runs.
"""

from .prf import prf_uniform, seed_from_context
from .gumbel import gumbel_scores
from .textseal import textseal_scores
from .unigram import unigram_features
from .kgw import kgw_features
from .entropy import entropy_scores

__all__ = [
    "prf_uniform",
    "seed_from_context",
    "gumbel_scores",
    "textseal_scores",
    "unigram_features",
    "kgw_features",
    "entropy_scores",
]
