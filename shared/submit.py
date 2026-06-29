"""Generic submission utility for CISPA hackathon API."""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from team_state import append_score, update_task

SUBMIT_COOLDOWN = int(os.environ.get("CISPA_SUBMIT_COOLDOWN", "300"))   # 5 min
QUERY_COOLDOWN = int(os.environ.get("CISPA_QUERY_COOLDOWN", "900"))     # 15 min


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def api_config(task_id: str) -> tuple[str, str, str]:
    base_url = _env("CISPA_BASE_URL")
    api_key = _env("CISPA_API_KEY")
    if not base_url or not api_key:
        die("Set CISPA_BASE_URL and CISPA_API_KEY in .env or environment")
    return base_url.rstrip("/"), api_key, task_id


def get_logits(
    query_path: str,
    task_id: str,
    log_path: str | None = None,
    owner: str | None = None,
) -> dict:
    base_url, api_key, task_id = api_config(task_id)
    if not os.path.isfile(query_path):
        die(f"File not found: {query_path}")

    print(f"Querying logits: {query_path} (task={task_id})")
    with open(query_path, "rb") as f:
        files = {"npz": (os.path.basename(query_path), f, "application/octet-stream")}
        resp = requests.post(
            f"{base_url}/{task_id}/logits",
            files=files,
            headers={"X-API-Key": api_key},
            timeout=120,
        )
    resp.raise_for_status()
    data = resp.json()
    print(f"OK — logits for {len(data.get('results', []))} items")

    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    update_task(
        task_id,
        owner=owner or os.environ.get("USER"),
        last_query_ts=datetime.now().isoformat(),
    )
    return data


def submit_file(
    file_path: str,
    task_id: str,
    log_path: str | None = None,
    owner: str | None = None,
    content_type: str = "application/octet-stream",
) -> dict:
    base_url, api_key, task_id = api_config(task_id)
    if not os.path.isfile(file_path):
        die(f"File not found: {file_path}")

    size = os.path.getsize(file_path)
    max_size = 200 * 1024 * 1024
    if size > max_size:
        die(f"File too large: {size / 1e6:.1f} MB (max 200 MB)")

    print(f"Submitting: {file_path} ({size / 1e6:.2f} MB, task={task_id})")
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, content_type)}
        resp = requests.post(
            f"{base_url}/submit/{task_id}",
            headers={"X-API-Key": api_key},
            files=files,
            timeout=(10, 180),
        )

    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}

    if resp.status_code == 413:
        die("Upload rejected: file too large (HTTP 413)")
    resp.raise_for_status()

    score = body.get("score")
    print(f"OK — response: {body}")
    if score is not None:
        print(f"Score: {score}")

    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now().isoformat(),
            "file": file_path,
            "response": body,
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

    attempt = body.get("attempt")
    history = append_score(task_id, score)
    update_task(
        task_id,
        owner=owner or os.environ.get("USER"),
        last_submit_ts=datetime.now().isoformat(timespec="seconds"),
        last_score=score,
        attempt=attempt,
        score_history=history,
    )
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit to CISPA hackathon API")
    parser.add_argument("file", help="Submission file path")
    parser.add_argument("--task-id", required=True, help="API task ID")
    parser.add_argument(
        "--action",
        choices=["submit", "logits", "both"],
        default="submit",
    )
    parser.add_argument("--log-dir", default="./logs/api")
    parser.add_argument("--owner", default=None, help="judoor username for team_state")
    parser.add_argument("--no-save-logs", action="store_true")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(args.file).stem
    logits_log = None if args.no_save_logs else f"{args.log_dir}/logits_{stem}_{ts}.json"
    submit_log = None if args.no_save_logs else f"{args.log_dir}/submit_{stem}_{ts}.json"

    if args.action in ("logits", "both"):
        get_logits(args.file, args.task_id, logits_log, owner=args.owner)
        if args.action == "both":
            print("Waiting 5s before submit...")
            time.sleep(5)

    if args.action in ("submit", "both"):
        submit_file(args.file, args.task_id, submit_log, owner=args.owner)


if __name__ == "__main__":
    main()
