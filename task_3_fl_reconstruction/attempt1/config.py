"""Central configuration for Task 3 (FL gradient reconstruction).

All paths are overridable via environment variables so the same code runs on
the JURECA cluster and on a laptop for offline testing.
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

SEED = 1234
