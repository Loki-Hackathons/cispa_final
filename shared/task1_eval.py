"""Task 1 (Text Watermark Localization) — local eval + viewer export.

No API call: computes TPR @ 0.1% FPR against the labeled val set and exports
a token-level bundle (ground truth labels + our predicted scores) for the
dashboard's Task 1 viewer. Every run is logged to history (kind=local_eval).

Usage:
    python shared/task1_eval.py \\
        --dataset path/to/val.jsonl \\
        --predictions output/val_scores.jsonl \\
        --method "entropy-weighted KGW+TextSeal ensemble" \\
        --note "raised KGW weight after CUDA Philox fix"
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from history import history_path, log_event, log_failure

try:
    from sklearn.metrics import roc_curve
except ImportError:
    roc_curve = None


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _load_jsonl(path: str) -> dict[str, dict]:
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[rec["document_id"]] = rec
    return records


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float = 0.001) -> float:
    if roc_curve is None:
        die("scikit-learn required (pip install scikit-learn)")
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")  # need both classes present
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(target_fpr, fpr, tpr))


def viz_dir() -> Path:
    d = history_path().parent / "task1_viz"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_bundle(
    dataset_path: str,
    predictions_path: str,
    method: str | None = None,
    note: str | None = None,
) -> dict:
    dataset = _load_jsonl(dataset_path)
    predictions = _load_jsonl(predictions_path)

    missing = set(dataset) - set(predictions)
    if missing:
        print(f"WARNING: {len(missing)} documents have no predictions (skipped): "
              f"{list(missing)[:5]}...")

    documents = []
    all_labels, all_scores = [], []

    for doc_id, doc in dataset.items():
        if "labels" not in doc:
            continue  # test set has no ground truth — nothing to visualize
        pred = predictions.get(doc_id)
        if pred is None:
            continue
        scores = pred["scores"]
        labels = doc["labels"]
        if len(scores) != len(labels):
            print(f"WARNING: {doc_id} length mismatch (labels={len(labels)}, "
                  f"scores={len(scores)}) — skipped")
            continue

        documents.append({
            "document_id": doc_id,
            "token_pieces": doc["token_pieces"],
            "labels": labels,
            "scores": scores,
        })
        all_labels.extend(labels)
        all_scores.extend(scores)

    if not documents:
        die("No overlapping labeled documents between dataset and predictions")

    tpr = tpr_at_fpr(np.array(all_labels), np.array(all_scores))

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": method,
        "note": note,
        "tpr_at_0.1pct_fpr": tpr,
        "n_documents": len(documents),
        "n_tokens": len(all_labels),
        "documents": documents,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1 local eval + viewer export")
    parser.add_argument("--dataset", required=True, help="Labeled JSONL (train/val)")
    parser.add_argument("--predictions", required=True, help="Our predictions JSONL")
    parser.add_argument("--method", default=None, help="Approach used")
    parser.add_argument("--note", default=None, help="Free-form note")
    parser.add_argument("--out", default=None, help="Bundle name (default: timestamp)")
    args = parser.parse_args()

    try:
        bundle = build_bundle(args.dataset, args.predictions, args.method, args.note)
    except (Exception, SystemExit) as e:
        log_failure("local_eval", "task_1", e, method=args.method)
        raise

    name = args.out or datetime.now().strftime("%Y%m%d_%H%M%S")
    if not name.endswith(".json"):
        name += ".json"
    out_path = viz_dir() / name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False)

    tpr = bundle["tpr_at_0.1pct_fpr"]
    print(f"TPR @ 0.1% FPR: {tpr:.6f}  ({bundle['n_documents']} docs, {bundle['n_tokens']} tokens)")
    print(f"Viewer bundle saved: {out_path}")

    log_event("local_eval", "task_1", score=tpr, file=str(out_path),
              method=args.method, note=args.note,
              extra={"n_documents": bundle["n_documents"], "n_tokens": bundle["n_tokens"]})


if __name__ == "__main__":
    main()
