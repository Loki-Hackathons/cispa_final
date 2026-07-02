"""Phase 6 — produce the test submission from a trained calibrator.

Loads a calibrator saved by ``train_calibrator.py``, scores every test token, applies the
same span smoothing, writes ``outputs/submission.jsonl`` and validates the format.

Usage:
    python -m src.predict --model outputs/calibrator_logreg.pkl
    # then submit:
    #   python ../../shared/submit.py \
    #       task_1_text_watermark/melissa/outputs/submission.jsonl \
    #       --task-id 30-watermark-localization --action submit
"""

from __future__ import annotations

import argparse
import pickle

from . import config
from .load_data import load_split
from .pipeline import build_matrix, scores_to_docs, write_submission, validate_submission
from .postprocess import postprocess


def run(model_path: str, out_path: str | None = None) -> str:
    config.ensure_dirs()
    cfg = config.load_watermark_config()
    with open(model_path, "rb") as fh:
        art = pickle.load(fh)
    model, scaler = art["model"], art["scaler"]

    print("[predict] building test features ...")
    test = load_split("test")
    use_entropy_lm = art.get("use_entropy_lm", True)
    if config.DISABLE_PROXY_LM:
        use_entropy_lm = False
        print("[predict] proxy LM disabled, using novelty proxy only")
    Xte, _, idx_te = build_matrix(
        test, cfg, use_detectors=True,
        use_entropy_lm=use_entropy_lm, with_labels=False,
        progress_every=100, progress_label="predict",
    )
    print(f"[predict] scoring {Xte.shape[0]} tokens ...")
    raw = model.predict_proba(scaler.transform(Xte))[:, 1]
    pred_by_doc = scores_to_docs(idx_te, raw)
    smoothed = {
        d: postprocess(s, smooth_radius=art.get("smooth_radius", 3),
                       smooth_sigma=art.get("smooth_sigma", 1.5))
        for d, s in pred_by_doc.items()
    }

    out_path = out_path or str(config.OUTPUT_DIR / "submission.jsonl")
    write_submission(smoothed, out_path)
    expected = {d.document_id: d.n_tokens for d in test}
    validate_submission(out_path, expected)
    print(f"[predict] wrote {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate + validate test submission.")
    parser.add_argument("--model", required=True, help="Path to calibrator .pkl")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run(args.model, args.out)
