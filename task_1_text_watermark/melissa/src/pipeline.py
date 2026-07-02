"""Shared pipeline helpers: build feature matrices and score documents.

Used by ``baseline.py``, ``train_calibrator.py`` and ``predict.py`` so feature
construction stays identical across train / eval / test.
"""

from __future__ import annotations

import json
from typing import Iterable

import numpy as np

from .config import WatermarkConfig, load_watermark_config
from .features import extract_features
from .load_data import Document


def build_matrix(docs: Iterable[Document], cfg: WatermarkConfig | None = None,
                 use_detectors: bool = True, use_entropy_lm: bool = True,
                 with_labels: bool = True):
    """Concatenate per-token features across docs.

    Returns (X, y, doc_index) where doc_index[i] = (document_id, position) so scores can
    be scattered back per document. ``y`` is None when labels are unavailable.
    """
    if cfg is None:
        cfg = load_watermark_config()
    rows: list[list[float]] = []
    ys: list[int] = []
    doc_index: list[tuple[str, int]] = []
    have_labels = with_labels
    for d in docs:
        feats = extract_features(
            d.token_ids, d.token_pieces, cfg=cfg,
            use_detectors=use_detectors, use_entropy_lm=use_entropy_lm,
        )
        for pos, f in enumerate(feats):
            rows.append(f)
            doc_index.append((d.document_id, pos))
        if d.labels is not None:
            ys.extend(d.labels)
        else:
            have_labels = False
    X = np.asarray(rows, dtype=np.float64)
    y = np.asarray(ys, dtype=np.int64) if (have_labels and ys) else None
    return X, y, doc_index


def scores_to_docs(doc_index: list[tuple[str, int]], scores: np.ndarray
                   ) -> dict[str, list[float]]:
    """Scatter a flat score vector back into {document_id: [scores...]}."""
    out: dict[str, list[float]] = {}
    for (doc_id, pos), s in zip(doc_index, scores):
        out.setdefault(doc_id, [])
        # positions are appended in order, so this stays aligned
        out[doc_id].append(float(s))
    return out


def write_submission(pred_by_doc: dict[str, list[float]], path: str) -> None:
    """Write predictions as competition .jsonl (one object per document)."""
    with open(path, "w", encoding="utf-8") as fh:
        for doc_id, scores in pred_by_doc.items():
            fh.write(json.dumps({"document_id": doc_id, "scores": scores}) + "\n")


def validate_submission(path: str, expected_docs: dict[str, int]) -> None:
    """Assert the .jsonl satisfies every submission rule; raise on the first violation.

    ``expected_docs`` maps document_id -> token count for the test split.
    """
    seen: set[str] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            obj = json.loads(line)
            if set(obj.keys()) != {"document_id", "scores"}:
                raise ValueError(f"line {ln}: keys must be exactly document_id + scores")
            did = str(obj["document_id"])
            if did in seen:
                raise ValueError(f"line {ln}: duplicate document_id {did}")
            seen.add(did)
            scores = obj["scores"]
            exp = expected_docs.get(did)
            if exp is None:
                raise ValueError(f"line {ln}: unknown document_id {did}")
            if len(scores) != exp:
                raise ValueError(f"line {ln}: {did} has {len(scores)} scores, expected {exp}")
            for v in scores:
                if not isinstance(v, (int, float)) or not np.isfinite(v) or not (0.0 <= v <= 1.0):
                    raise ValueError(f"line {ln}: {did} has an invalid score {v!r}")
    missing = set(expected_docs) - seen
    if missing:
        raise ValueError(f"missing {len(missing)} test documents, e.g. {list(missing)[:5]}")
    print(f"[validate] OK — {len(seen)} documents, format valid.")
