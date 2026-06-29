"""Thin W&B init helper for training tasks."""

import os

import wandb

from team_state import update_task


def init(
    task_name: str,
    config: dict | None = None,
    project: str = "cispa-finals",
    *,
    task_id: str | None = None,
    owner: str | None = None,
) -> wandb.Run:
    entity = os.environ.get("WANDB_ENTITY")
    run = wandb.init(
        project=project,
        entity=entity,
        name=task_name,
        config=config or {},
    )
    if task_id:
        update_task(
            task_id,
            owner=owner or os.environ.get("USER"),
            wandb_url=run.url,
        )
    return run
