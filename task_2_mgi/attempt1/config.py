"""Paths and hyperparameters for Task 2 MGI adversarial attack."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

# --- dataset / submission ---------------------------------------------------
BASE_IMAGES = 900
IMAGE_SIZE = 256
TOTAL_IMAGES = 1800
CLASS_SIZE = 300

# Submission slot layout: (name, slot_start, source_start, count, attack_mode)
# attack_mode (LEGACY run_attack.py / cw_attack.py, Stage-1 only):
#   "original" | "to_G" | "from_G"
# WARNING: this legacy layout does NOT drive Stage-2 (M/N). It leaves M_N and
# N_M as unmodified originals (score 0) and cannot distinguish G_M from G_N.
# The authoritative per-cell targets live in DIRECTION_TARGETS below and are
# implemented by attack_combined.py. cw_attack/run_attack are kept only for the
# "*->G" directions (Stage-1 is sufficient there).
DIRECTION_BLOCKS: list[tuple[str, int, int, int, str]] = [
    ("M_N", 0, 0, CLASS_SIZE, "original"),
    ("M_G", 300, 0, CLASS_SIZE, "to_G"),
    ("N_M", 600, 300, CLASS_SIZE, "original"),
    ("N_G", 900, 300, CLASS_SIZE, "to_G"),
    ("G_M", 1200, 600, CLASS_SIZE, "from_G"),
    ("G_N", 1500, 600, CLASS_SIZE, "from_G"),
]

# Authoritative detector-cell targets per direction:
#   (source_class, slot_start, stage1_target, stage2_target)
#   stage1_target: "generate" (L_A < tau_G) | "natural" (L_A > tau_G)
#   stage2_target: "raise" (member) | "lower" (non-member) | "off"
DIRECTION_TARGETS: dict[str, tuple[str, int, str, str]] = {
    "M_N": ("M", 0, "natural", "lower"),
    "M_G": ("M", 300, "generate", "off"),
    "N_M": ("N", 600, "natural", "raise"),
    "N_G": ("N", 900, "generate", "off"),
    "G_M": ("G", 1200, "natural", "raise"),
    "G_N": ("G", 1500, "natural", "lower"),
}

# Directions whose Stage-1-only attack (cw_attack.py) is a faithful full solution
# (Stage 2 is irrelevant because Stage 1 already assigns the class).
STAGE1_ONLY_DIRECTIONS = ("M_G", "N_G")
# Directions that REQUIRE attack_combined.py (Stage-2 controlled) or a content
# swap fallback; the legacy cw_attack pipeline cannot solve them correctly.
STAGE2_REQUIRED_DIRECTIONS = ("M_N", "N_M", "G_M", "G_N")

P0_DIRECTIONS = ("M_G", "N_G", "G_M", "G_N")
V1_ATTACK_DIRECTIONS = P0_DIRECTIONS

# Must match assemble + attack deployable validation (baseline npz used q=80).
DEFAULT_SUBMISSION_JPEG_QUALITY = 80

DEFAULT_CLUSTER_SCRATCH = Path("/p/scratch/training2625/dougnon1/Loki")


def _find_hf_tokenizer_ckpt() -> Path | None:
    cache_roots = [
        DEFAULT_CLUSTER_SCRATCH / ".cache/hub",
        Path.home() / ".cache/huggingface/hub",
    ]
    for root in cache_roots:
        snap_dir = root / "models--fun-research--TiTok/snapshots"
        if not snap_dir.is_dir():
            continue
        for snap in sorted(snap_dir.iterdir()):
            ckpt = snap / "maskgit-vqgan-imagenet-f16-256.bin"
            if ckpt.is_file():
                return ckpt
    return None


def default_data_dir() -> Path:
    env = os.environ.get("MGI_DATA_DIR")
    if env:
        return Path(env)
    cluster = DEFAULT_CLUSTER_SCRATCH / "MGI/data"
    if cluster.is_dir():
        return cluster
    return REPO_ROOT / "data" / "MGI"


def default_tokenizer_ckpt() -> Path:
    env = os.environ.get("TOKENIZER_CKPT")
    if env:
        return Path(env)
    found = _find_hf_tokenizer_ckpt()
    if found is not None:
        return found
    return DEFAULT_CLUSTER_SCRATCH / "MGI/model/maskgit-vqgan-imagenet-f16-256.bin"


def default_oned_tokenizer_root() -> Path:
    env = os.environ.get("ONED_TOKENIZER_ROOT")
    if env:
        return Path(env)
    cluster = DEFAULT_CLUSTER_SCRATCH / "1d-tokenizer"
    if cluster.is_dir():
        return cluster
    return REPO_ROOT / "external" / "1d-tokenizer"


def default_output_dir() -> Path:
    env = os.environ.get("MGI_OUTPUT_DIR")
    if env:
        return Path(env)
    out = SCRIPT_DIR / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out


@dataclass
class AttackConfig:
    lr: float = 5e-3
    max_steps: int = 200
    max_steps_retry: int = 500
    c: float = 10.0
    kappa: float = 0.05
    k_aug: int = 3
    early_stop: bool = True
    eps: float = 1e-6
    resize_prob: float = 0.4
    diversity_prob: float = 0.5
    device: str = "cuda"
    # Validate success on uint8 (+ JPEG) before checkpointing. None = uint8 only.
    submission_jpeg_quality: int | None = DEFAULT_SUBMISSION_JPEG_QUALITY


@dataclass
class PathsConfig:
    data_dir: Path = field(default_factory=default_data_dir)
    tokenizer_ckpt: Path = field(default_factory=default_tokenizer_ckpt)
    oned_tokenizer_root: Path = field(default_factory=default_oned_tokenizer_root)
    output_dir: Path = field(default_factory=default_output_dir)
    calibration_path: Path | None = None
    submission_path: Path | None = None

    def __post_init__(self) -> None:
        if self.calibration_path is None:
            self.calibration_path = self.output_dir / "calibration.json"
        if self.submission_path is None:
            self.submission_path = self.output_dir / "submission.npz"


def setup_oned_tokenizer_path(oned_root: Path | None = None) -> Path:
    root = oned_root or default_oned_tokenizer_root()
    root_str = str(root.resolve())
    if root.is_dir() and root_str not in os.sys.path:
        os.sys.path.insert(0, root_str)
    return root
