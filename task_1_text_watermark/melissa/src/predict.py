"""Phase 6 — produce the test submission from a trained calibrator.

Loads a calibrator saved by ``train_calibrator.py``, scores every test token, applies the
same span smoothing, writes ``outputs/submission.jsonl`` and validates the format.

Feature extraction is parallelised across documents (each document is independent) and
scores are saved progressively to ``outputs/submission.partial.jsonl`` so the run is
resumable: re-running skips documents already present in the partial file. Parallelism
does not change any score — the per-document features are computed by the exact same
code, and KGW's CUDA ``randperm`` is deterministic per seed regardless of the process.

Usage:
    python -m src.predict --model outputs/calibrator_logreg.pkl [--workers 8]
    # then submit:
    #   python ../../shared/submit.py \
    #       task_1_text_watermark/melissa/outputs/submission.jsonl \
    #       --task-id 30-watermark-localization --action submit
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from . import config
from .features import extract_features
from .load_data import load_split
from .pipeline import write_submission, validate_submission
from .postprocess import postprocess

# ---------------------------------------------------------------------------
# Worker side (one process per CPU): extract the feature matrix for one document.
# ---------------------------------------------------------------------------
_WORKER: dict = {}


def _init_worker(use_detectors: bool, use_entropy_lm: bool) -> None:
    _WORKER["cfg"] = config.load_watermark_config()
    _WORKER["use_detectors"] = use_detectors
    _WORKER["use_entropy_lm"] = use_entropy_lm


def _extract_doc(task: tuple[str, list, list]) -> tuple[str, np.ndarray]:
    doc_id, token_ids, pieces = task
    feats = extract_features(
        token_ids, pieces, cfg=_WORKER["cfg"],
        use_detectors=_WORKER["use_detectors"],
        use_entropy_lm=_WORKER["use_entropy_lm"],
    )
    return doc_id, np.asarray(feats, dtype=np.float64)


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


def run(model_path: str, out_path: str | None = None, workers: int | None = None) -> str:
    config.ensure_dirs()
    cfg = config.load_watermark_config()
    with open(model_path, "rb") as fh:
        art = pickle.load(fh)
    model, scaler = art["model"], art["scaler"]
    smooth_radius = art.get("smooth_radius", 3)
    smooth_sigma = art.get("smooth_sigma", 1.5)

    use_entropy_lm = art.get("use_entropy_lm", True)
    if config.DISABLE_PROXY_LM:
        use_entropy_lm = False
        print("[predict] proxy LM disabled, using novelty proxy only")

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

    # Worker count: WML_PREDICT_WORKERS overrides; else the SLURM CPU allocation;
    # else all CPUs (capped at 16).
    if workers is None:
        workers = int(os.environ.get("WML_PREDICT_WORKERS", "0"))
    if workers <= 0:
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    if workers <= 0:
        workers = min(16, max(1, os.cpu_count() or 1))
    print(f"[predict] {len(todo)} docs to score with {workers} worker(s) "
          f"({len(smoothed)} cached)")

    n_done = len(smoothed)

    def _score_and_save(pf, doc_id: str, feat: np.ndarray) -> None:
        nonlocal n_done
        raw = model.predict_proba(scaler.transform(feat))[:, 1]
        sc = postprocess([float(x) for x in raw],
                         smooth_radius=smooth_radius, smooth_sigma=smooth_sigma)
        smoothed[doc_id] = sc
        pf.write(json.dumps({"document_id": doc_id, "scores": sc}) + "\n")
        pf.flush()
        os.fsync(pf.fileno())
        n_done += 1
        if n_done % 100 == 0 or n_done == total:
            print(f"[predict] {n_done}/{total} documents scored", flush=True)

    if todo:
        # Append so we never lose already-saved rows; the final file is rebuilt (deduped)
        # from `smoothed`, so duplicate scratch lines here are harmless.
        with open(partial_path, "a", encoding="utf-8") as pf:
            if workers > 1:
                tasks = [(d.document_id, d.token_ids, d.token_pieces) for d in todo]
                # CUDA (KGW) requires the 'spawn' start method to be fork-safe.
                ctx = mp.get_context("spawn")
                with ProcessPoolExecutor(
                    max_workers=workers, mp_context=ctx,
                    initializer=_init_worker, initargs=(True, use_entropy_lm),
                ) as ex:
                    futures = [ex.submit(_extract_doc, t) for t in tasks]
                    for fut in as_completed(futures):
                        doc_id, feat = fut.result()
                        _score_and_save(pf, doc_id, feat)
            else:
                for d in todo:
                    feat = np.asarray(
                        extract_features(d.token_ids, d.token_pieces, cfg=cfg,
                                         use_detectors=True, use_entropy_lm=use_entropy_lm),
                        dtype=np.float64,
                    )
                    _score_and_save(pf, d.document_id, feat)

    write_submission(smoothed, out_path)
    validate_submission(out_path, expected)
    print(f"[predict] wrote {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate + validate test submission.")
    parser.add_argument("--model", required=True, help="Path to calibrator .pkl")
    parser.add_argument("--out", default=None)
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel feature-extraction processes "
                             "(default: all CPUs, or WML_PREDICT_WORKERS).")
    args = parser.parse_args()
    run(args.model, args.out, workers=args.workers)
