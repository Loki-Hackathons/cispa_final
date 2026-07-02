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

# HuggingFace dataset id (used only as a fallback if the local .jsonl files are absent).
DATASET_ID = os.environ.get("WML_DATASET", "SprintML/watermark_localization")

# Primary data source: a directory holding train.jsonl / validation.jsonl / test.jsonl
# (+ watermark_config.yaml), exactly as cloned from the HF dataset repo. This is how the
# team reads the data on JURECA. Default = the team scratch clone; local fallback below.
_DEFAULT_DATASET_DIRS = [
    "/p/scratch/training2625/ansart1/loki/watermark_localization",
    str(MELISSA_ROOT.parent.parent / "data" / "watermark_localization"),
]
DATASET_DIR = os.environ.get(
    "WML_DATASET_DIR",
    next((d for d in _DEFAULT_DATASET_DIRS if Path(d).is_dir()), _DEFAULT_DATASET_DIRS[0]),
)

# Tokenizer used by the organizers. token_ids are authoritative; we only load the
# tokenizer for vocab size / optional entropy proxy — never to retokenize `text`.
TOKENIZER_ID = os.environ.get("WML_TOKENIZER", "Qwen/Qwen2.5-7B-Instruct")

# YAML shipped with the dataset: watermark keys, detector params, repos+commits.
# Committed copy lives at task_1_text_watermark/watermark_config.yaml.
_DEFAULT_YAML = MELISSA_ROOT.parent / "watermark_config.yaml"
WATERMARK_YAML = os.environ.get(
    "WML_WATERMARK_YAML", str(_DEFAULT_YAML) if _DEFAULT_YAML.is_file() else ""
)

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
    vocab_size: int = 151643        # Qwen2.5 vocab (unigram greenlist size)
    gamma: float = 0.5              # Unigram green-list fraction
    kgw_gamma: float = 0.25         # KGW green-list fraction (differs from unigram!)
    gumbel_ngram: int = 2           # Gumbel-Max context width
    textseal_ngram: int = 3         # TextSeal context width
    # Secret keys per family. Filled from YAML on the cluster.
    keys: dict = field(default_factory=dict)
    # TextSeal dual-key routing probability (early fusion weight).
    textseal_alpha: float = 0.5
    # KGW greenlists were generated with torch.randperm on a CUDA (Philox) generator.
    kgw_use_cuda: bool = True

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> "WatermarkConfig":
        """Parse the dataset's watermark_config.yaml (schema_version 1).

        Normalises the nested ``watermarks:`` block into the flat fields the detectors
        expect (scalar keys for gumbel/unigram, dual key for textseal, gammas/ngrams).
        """
        import yaml  # local import: optional dependency

        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        cfg = cls()
        wm = raw.get("watermarks", {}) or {}

        gm = wm.get("gumbelmax", {}) or {}
        ts = wm.get("textseal", {}) or {}
        kgw = wm.get("kgw", {}) or {}
        uni = wm.get("unigram", {}) or {}

        # Keys in the format the detectors consume.
        cfg.keys = {
            "gumbel": int(gm.get("secret_key", 0)),
            "textseal": {"key1": int(ts.get("key_a", 0)), "key2": int(ts.get("key_b", 0))},
            "unigram": int(uni.get("watermark_key", 0)),
            "kgw": kgw.get("seeding_scheme", ""),  # KGW uses a seeding scheme, not a scalar
        }
        cfg.gumbel_ngram = int(gm.get("ngram", cfg.gumbel_ngram))
        cfg.textseal_ngram = int(ts.get("ngram", cfg.textseal_ngram))
        cfg.context_width = cfg.textseal_ngram
        cfg.textseal_alpha = float(ts.get("mixing_alpha", cfg.textseal_alpha))
        cfg.gamma = float(uni.get("fraction", cfg.gamma))
        cfg.kgw_gamma = float(kgw.get("gamma", cfg.kgw_gamma))
        return cfg


def load_watermark_config() -> WatermarkConfig:
    """Load the watermark config from the dataset YAML if available, else defaults."""
    if WATERMARK_YAML and os.path.isfile(WATERMARK_YAML):
        return WatermarkConfig.from_yaml(WATERMARK_YAML)
    return WatermarkConfig()


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
