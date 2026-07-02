"""Phase 6 — produce the test submission from the selected scorer.

Loads the scorer chosen by ``train_calibrator.py`` (``outputs/scorer.pkl``) — either a
calibrator head (logreg / gboost on the vendor windowed-z features) or the span-aware HMM
— scores every test token, applies the selected moving-average smoothing, and writes
``outputs/submission.jsonl``.

Runs single-process on purpose: the KGW greenlists are built on the GPU and a single
``KgwMaskScorer`` keeps one greenlist cache shared across all documents (contexts recur),
which is both correct and fast. Scores are saved progressively to
``outputs/submission.partial.jsonl`` so the run is resumable — re-running skips documents
already present in the partial file.

Usage:
    python -m src.predict --model outputs/scorer.pkl
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
from .correct_features import doc_feature_matrix, kgw_signal
from .load_data import load_split
from .pipeline import validate_submission, write_submission
from .postprocess import apply_smoothing


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
    scorer = art.get("scorer", "calibrator")
    window = art.get("smooth_window", 1)
    use_kgw = art.get("use_kgw", True)

    if scorer == "calibrator":
        model, scaler = art["model"], art["scaler"]

        def _raw(token_ids):
            feat = doc_feature_matrix(token_ids, use_kgw=use_kgw)
            return model.predict_proba(scaler.transform(feat))[:, 1]
    elif scorer == "hmm":
        from hmm_scorer import posterior_scores
        hmm = art["hmm"]

        def _raw(token_ids):
            ex = {"kgw": kgw_signal(token_ids)} if use_kgw else None
            return posterior_scores(hmm, list(token_ids), ex)
    else:
        raise ValueError(f"unknown scorer type in artifact: {scorer!r}")

    print(f"[predict] scorer={art.get('model_kind', scorer)} window={window} "
          f"use_kgw={use_kgw}")

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
    print(f"[predict] {len(todo)} docs to score ({len(smoothed)} cached)")

    n_done = len(smoothed)
    with open(partial_path, "a", encoding="utf-8") as pf:
        for d in todo:
            raw = _raw(d.token_ids)
            sc = apply_smoothing([float(x) for x in raw], window)
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
    parser.add_argument("--model", required=True, help="Path to scorer .pkl")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run(args.model, args.out)
