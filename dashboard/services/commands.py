"""Copy-paste shell commands for dashboard UI."""

from dashboard.models import CommandChip, FailedJob, SlurmJob, TaskStatus


def job_commands(job: SlurmJob) -> list[CommandChip]:
    jid = job.job_id
    return [
        CommandChip(label="tail out", command=f"tail -f logs/slurm_{jid}.out"),
        CommandChip(label="tail err", command=f"tail -f logs/slurm_{jid}.err"),
        CommandChip(label="scancel", command=f"scancel {jid}"),
    ]


def failed_job_commands(job_id: str) -> list[CommandChip]:
    return [
        CommandChip(label="tail err", command=f"tail -n 80 logs/slurm_{job_id}.err"),
        CommandChip(label="sacct", command=f"sacct -j {job_id} --format=JobID,State,ExitCode,Elapsed -P"),
    ]


def task_commands(task: TaskStatus) -> list[CommandChip]:
    owner = task.owner or "$USER"
    tid = task.task_id
    return [
        CommandChip(
            label="submit",
            command=(
                f"python shared/submit.py output/submission.npz "
                f"--task-id {tid} --action submit --owner {owner}"
            ),
        ),
        CommandChip(
            label="analyze",
            command=(
                f"python shared/analyze.py output/submission.npz --mode api "
                f"--task-id {tid} --dataset <path>"
            ),
        ),
        CommandChip(
            label="logits",
            command=(
                f"python shared/submit.py output/submission.npz "
                f"--task-id {tid} --action logits --owner {owner}"
            ),
        ),
    ]


def attach_job_commands(jobs: list[SlurmJob]) -> list[SlurmJob]:
    return [j.model_copy(update={"commands": job_commands(j)}) for j in jobs]


def attach_task_commands(tasks: list[TaskStatus]) -> list[TaskStatus]:
    return [t.model_copy(update={"commands": task_commands(t)}) for t in tasks]


def attach_failed_commands(failed: list[FailedJob]) -> list[FailedJob]:
    return [
        f.model_copy(update={"commands": failed_job_commands(f.job_id)}) for f in failed
    ]
