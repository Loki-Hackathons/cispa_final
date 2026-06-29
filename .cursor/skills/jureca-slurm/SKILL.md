---
name: jureca-slurm
description: Write and submit SLURM jobs on JURECA for CISPA hackathon. Use when creating run scripts, requesting GPUs, monitoring jobs, or debugging cluster issues.
---

# JURECA SLURM

## When to use

- Writing a new `run.sh` or SLURM script
- Choosing GPU count (1, 2, or 4)
- Monitoring or cancelling jobs
- Interactive GPU debugging

## Cluster constants

| Setting | Value |
|---------|-------|
| Host | `jureca.fz-juelich.de` |
| Account | `training2557` |
| Partition (short) | `dc-gpu-devel` |
| Partition (long) | `dc-gpu` |
| GPUs per node | 4× A100 40GB |

## SLURM template

Copy from `slurm/templates/1gpu_devel.sh`, `2gpu_devel.sh`, or `4gpu_devel.sh`.

```bash
#!/bin/bash
#SBATCH --account=training2557
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --gres=gpu:2          # 1, 2, or 4
#SBATCH --cpus-per-task=32    # 16 for 1gpu, 32 for 2gpu, 128 for 4gpu
#SBATCH --time=02:00:00
#SBATCH --job-name=task1_attempt1
#SBATCH --output=logs/slurm_%j.out

module load GCC CUDA PyTorch torchvision
source .venv/bin/activate
python -u main.py "$@"
```

## Commands

```bash
sbatch run.sh                  # submit
squeue -A training2557         # all team jobs
squeue -u $USER                # your jobs
scancel <job_id>               # cancel
```

Log every submission in `slurm/submitted.log` and `docs/notes-communes.md`.

## Job naming

```bash
#SBATCH --job-name=task1_att3_${USER}   # task{N}_att{M}_{user}
```

## Job progress (dashboard)

At the start of `main.py` (jobs >10 min):

```python
from job_progress import bind_job, report, complete
bind_job("task_1", attempt=3)
```

See skill `job-progress`.

## Multi-GPU in code

SLURM allocates GPUs; code must use them:

- **Training:** `nn.DataParallel(model)` when `torch.cuda.device_count() > 1`
- **Independent work:** split items across GPUs, `torch.cuda.set_device(gpu_id)` per thread

## Interactive session

```bash
salloc -p dc-gpu-devel -t 20 -N 1 -A training2557
srun --pty bash -i
module load GCC CUDA PyTorch torchvision
source .venv/bin/activate
nvidia-smi
```

## Coordination

- Coordinate GPU budget in `docs/notes-communes.md` before `sbatch`
- Run `python shared/dashboard.py` for live queue view
- Always work inside `tmux`

## Known issue

GPU cgroup constraint may be broken on JURECA. Use `CUDA_VISIBLE_DEVICES` explicitly if multi-process jobs interfere.
