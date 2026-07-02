#!/bin/bash
# Poll the entropy SLURM job from WSL until it leaves the queue, then show
# outputs. Usage: bash poll_entropy.sh <job_id>
set -u
JOB="$1"
for i in $(seq 1 90); do
    st=$(ssh jureca "squeue -j $JOB -h -o %T" 2>/dev/null)
    echo "poll $i: ${st:-gone}"
    if [ -z "$st" ]; then
        echo JOB_LEFT_QUEUE
        break
    fi
    sleep 60
done
ssh jureca "cd code/cispa_final/task_1_text_watermark/alexandre && ls -la output/entropy_*.npz 2>/dev/null; echo ---LOG; tail -25 logs/slurm_${JOB}.out 2>/dev/null; echo ---ERR; tail -8 logs/slurm_${JOB}.err 2>/dev/null; true"
