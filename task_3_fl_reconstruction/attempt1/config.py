"""Central configuration for Task 3 (FL gradient reconstruction).

All paths are overridable via environment variables so the same code runs on
the JURECA cluster and on a laptop for offline testing.

On JURECA, run ``source setup_cluster.sh`` (sets TASK3_DATA_ROOT + venv).
Do not set TASK3_DATA_ROOT to chat placeholders like ``.../FL_Data_Reconstruction``.
"""
from __future__ import annotations

import os

# Where models/ and gradients/ live (JURECA team scratch by default).
DATA_ROOT = os.environ.get(
    "TASK3_DATA_ROOT",
    "/p/scratch/training2625/dougnon1/Loki/FL_Data_Reconstruction",
)

MODELS_DIR = os.path.join(DATA_ROOT, "models")
GRADIENTS_DIR = os.path.join(DATA_ROOT, "gradients")


def ensure_data_root() -> None:
    """Fail with a clear message if TASK3_DATA_ROOT is wrong or data is missing."""
    root = DATA_ROOT
    if "..." in root:
        raise FileNotFoundError(
            f"TASK3_DATA_ROOT contains '...' (placeholder): {root!r}\n"
            "On JURECA run:  source setup_cluster.sh\n"
            "Or export the full path:\n"
            "  export TASK3_DATA_ROOT="
            "/p/scratch/training2625/dougnon1/Loki/FL_Data_Reconstruction"
        )
    probe = os.path.join(GRADIENTS_DIR, "model1.pt")
    if not os.path.isfile(probe):
        raise FileNotFoundError(
            f"Task 3 data not found: {probe}\n"
            f"  TASK3_DATA_ROOT={root}\n"
            f"  gradients dir exists: {os.path.isdir(GRADIENTS_DIR)}\n"
            "On JURECA run:  source setup_cluster.sh\n"
            "Wrong folder (no data): .../dougnon1/FL_Data_Reconstruction/ "
            "(missing Loki/)\n"
            "Correct folder: .../dougnon1/Loki/FL_Data_Reconstruction/"
        )

NUM_MODELS = 12
BATCH = 128                 # images per model (fixed by the task)
IMG_SHAPE = (3, 64, 64)     # required submission shape per image
IMG_FLAT = IMG_SHAPE[0] * IMG_SHAPE[1] * IMG_SHAPE[2]  # 12288

OUT_DIR = os.environ.get("TASK3_OUT_DIR", "output")
SUBMISSION_PATH = os.environ.get("TASK3_SUBMISSION_PATH", "submission.pt")

# Numerical floor for "bias gradient is non-zero" (analytic extraction).
EPS = 1e-8

# Selection heuristics.
DEDUP_SIM_THRESHOLD = 0.92  # cosine similarity above which two imgs are "the same"
CONFIDENCE_WEIGHT = 0.55    # blend for fc1 row selection (see fc1_analytic.py)

# Optimized images replace analytic ones only if they beat them on observed-
# gradient reproduction by at least this margin. A better gradient fit does not
# guarantee better SSIM, so we demand a clear improvement before switching.
SELECT_MARGIN = 0.02

SEED = 1234
