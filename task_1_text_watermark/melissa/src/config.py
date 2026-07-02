"""Central configuration for the Text Watermark Localization solution.

All paths and knobs live here so scripts stay clean and reproducible. On JURECA,
override paths with environment variables (see ``melissa/scripts/``) instead of
editing this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------
# Root of this solution (the ``melissa/`` directory).
MELISSA_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.environ.get("WML_OUTPUT_DIR", MELISSA_ROOT / "outputs"))
CACHE_DIR = Path(os.environ.get("WML_CACHE_DIR", MELISSA_ROOT / "data_cache"))

# HuggingFace dataset id (or a local path to a downloaded copy).
DATASET_ID = os.environ.get("WML_DATASET", "SprintML/watermark_localization")

# Tokenizer used by the organizers. token_ids are authoritative; we only load the
# tokenizer for vocab size / optional entropy proxy — never to retokenize `text`.
TOKENIZER_ID = os.environ.get("WML_TOKENIZER", "Qwen/Qwen2.5-7B-Instruct")

# YAML shipped with the dataset: watermark keys, detector params, repos+commits.
# Set this on the cluster (it is not available on the local machine).
WATERMARK_YAML = os.environ.get("WML_WATERMARK_YAML", "")  # e.g. /p/scratch/.../keys.yaml

# Optional proxy LM for entropy weighting (TextSeal §3.2). Smaller = cheaper.
ENTROPY_PROXY_MODEL = os.environ.get("WML_ENTROPY_MODEL", "Qwen/Qwen2.5-0.5B")

# Submission / API.
TASK_ID = os.environ.get("WML_TASK_ID", "30-watermark-localization")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = int(os.environ.get("WML_SEED", "1234"))

# ---------------------------------------------------------------------------
# Detector / watermark defaults (overridden by WATERMARK_YAML when present)
# ---------------------------------------------------------------------------


@dataclass
class WatermarkConfig:
    """Watermark + detector parameters.

    Defaults follow the papers; the dataset YAML overrides them via ``from_yaml``.
    """

    context_width: int = 3          # k previous tokens seed the PRF (spec fixes k=3)
    vocab_size: int = 152064        # Qwen2.5 vocab (approx; refined from tokenizer)
    gamma: float = 0.5              # KGW / Unigram green-list fraction
    # Secret keys per family. Filled from YAML on the cluster.
    keys: dict = field(default_factory=dict)
    # TextSeal dual-key routing probability (early fusion weight).
    textseal_alpha: float = 0.1
    # KGW greenlists were generated with torch.randperm on a CUDA (Philox) generator.
    kgw_use_cuda: bool = True

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> "WatermarkConfig":
        import yaml  # local import: optional dependency

        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        cfg = cls()
        for k, v in raw.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        # Common alternative layouts: keys nested under "keys" / "watermarks".
        cfg.keys = raw.get("keys", raw.get("watermarks", cfg.keys))
        return cfg


def load_watermark_config() -> WatermarkConfig:
    """Load the watermark config from the dataset YAML if available, else defaults."""
    if WATERMARK_YAML and os.path.isfile(WATERMARK_YAML):
        return WatermarkConfig.from_yaml(WATERMARK_YAML)
    return WatermarkConfig()


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
