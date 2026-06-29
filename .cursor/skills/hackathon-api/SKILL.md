---
name: hackathon-api
description: Submit results and analyze API feedback for CISPA hackathon tasks. Use when submitting to leaderboard, querying logits, checking cooldowns, or parsing per-image results.
---

# CISPA Hackathon API

## When to use

- Submitting a solution file to the leaderboard
- Querying logits (rate-limited)
- Analyzing per-image API feedback
- Checking API cooldown status

## Configuration

Store in `.env` on cluster (never commit):

```bash
CISPA_BASE_URL=http://HOST:PORT
CISPA_API_KEY=your_key
CISPA_SUBMIT_COOLDOWN=300    # 5 min (regional reference)
CISPA_QUERY_COOLDOWN=900     # 15 min (regional reference)
```

## Submit

```bash
python shared/submit.py <file> --task-id <TASK_ID> --action submit
python shared/submit.py <file> --task-id <TASK_ID> --action logits
python shared/submit.py <file> --task-id <TASK_ID> --action both
```

Updates `team_state.json` with timestamp, score, and `score_history` (last 5).

Always pass `--owner <judoor_user>`.

## Analyze

For adversarial `.npz` submissions (regional Task 1 format):

```bash
# Local — L2 lower bound only
python shared/analyze.py submission.npz --task-id <TASK_ID> --mode local --dataset natural_images.pt

# API — true leaderboard score
python shared/analyze.py submission.npz --task-id <TASK_ID> --mode api --dataset natural_images.pt
```

Output saved to `logs/analysis_<mode>_<timestamp>.json`.

## Cooldowns

Check before submitting:

```bash
python shared/dashboard.py
```

Or read `team_state.json` — `last_submit_ts` and `last_query_ts` per task.

Coordinate on Discord when a teammate is about to submit.

## Typical API response (adversarial)

```json
{
  "results": [
    {"image_id": 0, "logits": [0.1, 0.8, ...]},
    ...
  ]
}
```

Score logic (Task 1 style): misclassified → normalized L2; classified → 1.0. Average over all images.

## Logs

API responses saved to `logs/api/` when using `--log-dir` (default). Keep these — they drive iterative epsilon tuning (see regional `attempt3/logs/`).

## Task-specific notes

- Adapt `--task-id` and file format per task when subject is released
- Update `docs/subject/subject.md` with endpoints and metrics
- CSV submissions may need different `content_type` in `submit.py` — check task spec
