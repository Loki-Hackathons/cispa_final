"""Phase 6 — produce the test submission from a trained calibrator.

Loads a calibrator saved by ``train_calibrator.py``, scores every test token with the
**correct, vendor-based** watermark features (TextSeal / Gumbel-Max / Unigram + KGW),
applies the selected span smoothing, and writes ``outputs/submission.jsonl``.

Runs single-process on purpose: the KGW greenlists are built on the GPU and a single
``KgwMaskScorer`` keeps one greenlist cache shared across all documents (contexts recur),
which is both correct and fast. Scores are saved progressively to
``outputs/submission.partial.jsonl`` so the run is resumable — re-running skips documents
already present in the partial file.

Usage:
    python -m src.predict --model outputs/calibrator_gboost.pkl
    # then submit:
    #   python ../../shared/submit.py \
    #       task_1_text_watermark/melissa/outputs/submission.jsonl \
    #       --task-id 30-watermark-localization --action submit
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

from . import config
from .correct_features import doc_feature_matrix
from .load_data import load_split
from .pipeline import validate_submission, write_submission
from .postprocess import postprocess


def _load_partial(partial_path: str, expected: dict[str, int]) -> dict[str, list]:
    """Read already-computed, correctly-sized documents from the partial file."""
    done: dict[str, list] = {}
    if not os.path.exists(partial_path):
        return done
    with open(partial_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            did = str(obj.get("document_id"))
            scores = obj.get("scores")
            if did in expected and isinstance(scores, list) and len(scores) == expected[did]:
                done[did] = scores  # later lines overwrite earlier duplicates
    return done


def run(model_path: str, out_path: str | None = None) -> str:
    config.ensure_dirs()
    with open(model_path, "rb") as fh:
        art = pickle.load(fh)
    model, scaler = art["model"], art["scaler"]
    smooth_radius = art.get("smooth_radius", 0)
    smooth_sigma = art.get("smooth_sigma", 0.0)
    use_kgw = art.get("use_kgw", True)

    test = load_split("test")
    expected = {d.document_id: d.n_tokens for d in test}
    total = len(test)

    out_path = out_path or str(config.OUTPUT_DIR / "submission.jsonl")
    partial_path = str(config.OUTPUT_DIR / "submission.partial.jsonl")

    # Resume from any previously saved partial results.
    smoothed = _load_partial(partial_path, expected)
    if smoothed:
        print(f"[predict] resuming: {len(smoothed)}/{total} docs already in {partial_path}")
    todo = [d for d in test if d.document_id not in smoothed]
    print(f"[predict] {len(todo)} docs to score (use_kgw={use_kgw}, {len(smoothed)} cached)")

    n_done = len(smoothed)
    with open(partial_path, "a", encoding="utf-8") as pf:
        for d in todo:
            feat = doc_feature_matrix(d.token_ids, use_kgw=use_kgw)
            raw = model.predict_proba(scaler.transform(feat))[:, 1]
            if smooth_radius and smooth_radius > 0:
                sc = postprocess([float(x) for x in raw],
                                 smooth_radius=smooth_radius, smooth_sigma=smooth_sigma)
            else:
                sc = [min(1.0, max(0.0, float(x))) for x in raw]
            smoothed[d.document_id] = sc
            pf.write(json.dumps({"document_id": d.document_id, "scores": sc}) + "\n")
            pf.flush()
            os.fsync(pf.fileno())
            n_done += 1
            if n_done % 50 == 0 or n_done == total:
                print(f"[predict] {n_done}/{total} documents scored", flush=True)

    write_submission(smoothed, out_path)
    validate_submission(out_path, expected)
    print(f"[predict] wrote {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate + validate test submission.")
    parser.add_argument("--model", required=True, help="Path to calibrator .pkl")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run(args.model, args.out)
