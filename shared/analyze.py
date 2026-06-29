"""Analysis utility — local L2 for adversarial .npz, API mode for true scores."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from submit import get_logits
from team_state import update_task

try:
    import torch
except ImportError:
    torch = None


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_dataset(path: str) -> dict:
    if torch is None:
        die("torch required for dataset loading")
    if not os.path.exists(path):
        die(f"Dataset not found: {path}")
    return torch.load(path, weights_only=False)


def analyze_local_npz(submission_path: str, dataset: dict) -> dict:
    """L2 analysis for adversarial .npz submissions (regional Task 1 format)."""
    sub = np.load(submission_path)
    adv_images = sub["images"]
    original_images = dataset["images"].numpy()

    n, c, h, w = original_images.shape
    l2_norm_factor = float(np.sqrt(c * h * w))
    diffs = adv_images - original_images
    l2_per_image = np.linalg.norm(diffs.reshape(len(diffs), -1), axis=1)
    l2_norm = np.clip(l2_per_image / l2_norm_factor, 0.0, 1.0)

    avg_norm = float(np.mean(l2_norm))
    print(f"Local L2 (normalized avg): {avg_norm:.6f} — lower bound only")
    print("Use --mode api for true leaderboard score.")

    return {
        "mode": "local",
        "timestamp": datetime.now().isoformat(),
        "submission_file": submission_path,
        "avg_l2_normalized": avg_norm,
        "score_lower_bound": avg_norm,
        "per_image": [
            {"image_id": i, "l2_normalized": float(l2_norm[i])}
            for i in range(len(l2_norm))
        ],
    }


def analyze_api_npz(
    submission_path: str,
    dataset: dict,
    task_id: str,
    owner: str | None = None,
) -> dict:
    """Query API logits and compute adversarial L2 score (regional Task 1 metric)."""
    logits_data = get_logits(submission_path, task_id, owner=owner)
    sub = np.load(submission_path)
    adv_images = sub["images"]
    original_images = dataset["images"].numpy()
    true_labels = dataset["labels"].numpy()

    results = sorted(logits_data["results"], key=lambda x: x["image_id"])
    n, c, h, w = original_images.shape
    l2_norm_factor = float(np.sqrt(c * h * w))

    per_image = []
    scores = []
    successes = 0

    print(f"\n{'ID':>3} | {'True':>4} | {'Pred':>4} | {'Status':>7} | {'Score':>6}")
    print("-" * 50)

    for i, res in enumerate(results):
        img_id = res["image_id"]
        logits = np.array(res["logits"])
        pred = int(np.argmax(logits))
        true = int(true_labels[img_id])
        misclassified = pred != true

        diff = adv_images[i] - original_images[i]
        l2_raw = float(np.linalg.norm(diff))
        l2_norm = float(np.clip(l2_raw / l2_norm_factor, 0.0, 1.0))
        score = l2_norm if misclassified else 1.0
        scores.append(score)
        if misclassified:
            successes += 1

        status = "SUCCESS" if misclassified else "FAILED"
        print(f"{img_id:3d} | {true:4d} | {pred:4d} | {status:>7} | {score:6.4f}")
        per_image.append({
            "image_id": img_id,
            "true_label": true,
            "predicted_label": pred,
            "misclassified": misclassified,
            "l2_normalized": l2_norm,
            "score": score,
        })

    leaderboard_score = float(np.mean(scores))
    success_rate = successes / len(scores) * 100
    print(f"\nLeaderboard score: {leaderboard_score:.6f}")
    print(f"Success rate: {successes}/{len(scores)} ({success_rate:.1f}%)")

    update_task(
        task_id,
        owner=owner or os.environ.get("USER"),
        last_score=leaderboard_score,
        last_query_ts=datetime.now().isoformat(),
    )

    return {
        "mode": "api",
        "timestamp": datetime.now().isoformat(),
        "submission_file": submission_path,
        "leaderboard_score": leaderboard_score,
        "success_rate": success_rate,
        "num_successes": successes,
        "per_image": per_image,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CISPA submission")
    parser.add_argument("file", help="Submission file (.npz)")
    parser.add_argument("--task-id", required=True, help="API task ID")
    parser.add_argument("--mode", choices=["local", "api"], default="local")
    parser.add_argument("--dataset", default=None, help="natural_images.pt for local/api npz")
    parser.add_argument("--output", default=None, help="Save analysis JSON here")
    parser.add_argument("--owner", default=None)
    args = parser.parse_args()

    if not args.dataset:
        die("--dataset required for .npz adversarial analysis")

    dataset = load_dataset(args.dataset)

    if args.mode == "local":
        analysis = analyze_local_npz(args.file, dataset)
    else:
        analysis = analyze_api_npz(args.file, dataset, args.task_id, owner=args.owner)

    out = args.output or f"logs/analysis_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
