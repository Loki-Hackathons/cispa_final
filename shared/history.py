"""Append-only submission/test history (JSONL) — one event per line.

Auto-fed by submit.py / analyze.py. Read from CLI, dashboard, or laptop
(after scripts/sync_history.ps1). Path override: CISPA_HISTORY_FILE.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Shared team file on JURECA (finals scratch); repo-local fallback for dev
_CLUSTER_PATH = Path("/p/scratch/training2625/ansart1/loki/history/submissions.jsonl")
_LOCAL_PATH = Path(__file__).resolve().parent.parent / "history" / "submissions.jsonl"


def history_path() -> Path:
    env = os.environ.get("CISPA_HISTORY_FILE")
    if env:
        return Path(env)
    if _CLUSTER_PATH.parent.parent.exists():
        return _CLUSTER_PATH
    return _LOCAL_PATH


def log_event(
    kind: str,
    task_id: str | None = None,
    *,
    score: float | None = None,
    file: str | None = None,
    owner: str | None = None,
    note: str | None = None,
    method: str | None = None,
    status: str = "success",
    extra: dict | None = None,
) -> dict:
    """Append one event — EVERY attempt is logged, including failures.

    kind: submit | logits | analysis | local_eval | experiment | ...
    method: short description of the approach used (e.g. "KGW+TextSeal ensemble, entropy weighting").
    status: success | error | local — always set, so failed attempts are never silently dropped.
    """
    event: dict = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "kind": kind,
        "task_id": task_id,
        "owner": owner or os.environ.get("USER") or os.environ.get("USERNAME"),
        "status": status,
    }
    if score is not None:
        event["score"] = float(score)
    if file:
        event["file"] = str(file)
    if note:
        event["note"] = note
    if method:
        event["method"] = method
    if extra:
        event["extra"] = extra

    path = history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def log_failure(
    kind: str,
    task_id: str | None,
    error: Exception | str,
    *,
    owner: str | None = None,
    file: str | None = None,
    method: str | None = None,
) -> dict:
    """Log a failed attempt. Call from except blocks so failures are never lost."""
    return log_event(
        kind, task_id, owner=owner, file=file, method=method,
        status="error", extra={"error": str(error)},
    )


def read_events(
    limit: int = 0,
    task_id: str | None = None,
    kind: str | None = None,
) -> list[dict]:
    """Newest first. limit=0 means all."""
    path = history_path()
    if not path.exists():
        return []
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if task_id and ev.get("task_id") != task_id:
                continue
            if kind and ev.get("kind") != kind:
                continue
            events.append(ev)
    events.reverse()
    return events[:limit] if limit else events


def best_scores() -> dict[str, dict]:
    """Best (max) scored event per task."""
    best: dict[str, dict] = {}
    for ev in read_events():
        task = ev.get("task_id")
        score = ev.get("score")
        if task is None or score is None:
            continue
        if task not in best or score > best[task]["score"]:
            best[task] = ev
    return best


def _print_table(events: list[dict]) -> None:
    if not events:
        print(f"No events in {history_path()}")
        return
    print(f"{'TIME':<20} {'KIND':<10} {'TASK':<8} {'STATUS':<8} {'SCORE':>10}  METHOD / FILE / NOTE")
    print("-" * 110)
    for ev in events:
        score = f"{ev['score']:.6f}" if ev.get("score") is not None else "—"
        parts = [p for p in (ev.get("method"), ev.get("file"), ev.get("note")) if p]
        if ev.get("status") == "error" and ev.get("extra", {}).get("error"):
            parts.append(f"ERROR: {ev['extra']['error']}")
        detail = "  |  ".join(parts)
        status = ev.get("status", "success")
        print(f"{ev.get('ts', ''):<20} {ev.get('kind', ''):<10} "
              f"{(ev.get('task_id') or '—'):<8} {status:<8} {score:>10}  {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submission/test history")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="Show events (newest first)")
    p_list.add_argument("-n", type=int, default=30, help="Max events (0 = all)")
    p_list.add_argument("--task", default=None)
    p_list.add_argument("--kind", default=None)
    p_list.add_argument("--json", action="store_true", help="Raw JSON output")

    p_log = sub.add_parser("log", help="Log a manual event")
    p_log.add_argument("--kind", default="experiment")
    p_log.add_argument("--task", default=None)
    p_log.add_argument("--score", type=float, default=None)
    p_log.add_argument("--file", default=None)
    p_log.add_argument("--method", default=None, help="Approach used, e.g. 'entropy-weighted ensemble'")
    p_log.add_argument("--note", default=None)
    p_log.add_argument("--status", default="success", choices=["success", "error", "local"])

    sub.add_parser("best", help="Best score per task")

    args = parser.parse_args()
    if args.cmd == "list":
        events = read_events(limit=args.n, task_id=args.task, kind=args.kind)
        if args.json:
            json.dump(events, sys.stdout, indent=2, ensure_ascii=False)
            print()
        else:
            _print_table(events)
    elif args.cmd == "log":
        ev = log_event(args.kind, args.task, score=args.score, file=args.file,
                       method=args.method, note=args.note, status=args.status)
        print(f"Logged: {json.dumps(ev, ensure_ascii=False)}")
    elif args.cmd == "best":
        for task, ev in sorted(best_scores().items()):
            print(f"{task:<12} {ev['score']:.6f}  at {ev['ts']}  ({ev.get('file', '')})")


if __name__ == "__main__":
    main()
