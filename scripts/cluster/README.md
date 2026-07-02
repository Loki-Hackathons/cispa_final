# Cluster setup — CISPA Grand Finals (Team Loki)

Official organizer guide: [`docs/Hackathon_Setup Finale.md`](../../docs/Hackathon_Setup%20Finale.md)

**Important:** Finals use project **`training2625`** (not `training2557` from regional).

Only **one person (owner)** runs `hackathon_setup.sh`. Others run `teammate.sh`.

## Prerequisites (JuDoor — manual)

1. Join project **`training2625`** on [JuDoor](https://judoor.fz-juelich.de) → wait for PI approval
2. **JURECA** appears under **Systems** → sign User Agreement + upload SSH public key
3. Wait ~15 min after adding SSH key
4. SSH works with TOTP prompt (not `Permission denied (publickey)`)

## Owner bootstrap (ansart1 — run once on JURECA)

```bash
ssh -i ~/.ssh/id_ed25519 \
  -o Ciphers=aes256-ctr \
  -o MACs=hmac-sha2-256-etm@openssh.com \
  ansart1@jureca.fz-juelich.de

jutil env activate -p training2625
tmux new -s hackathon

# Download organizer scripts
cd /p/scratch/training2625
mkdir -p ansart1 && cd ansart1
wget https://huggingface.co/datasets/SprintML/hackathon/resolve/main/hackathon_setup.sh -O hackathon_setup.sh
wget https://huggingface.co/datasets/SprintML/hackathon/resolve/main/teammate.sh -O teammate.sh

# Edit hackathon_setup.sh — use values from scripts/cluster/loki.env:
#   OWNER=ansart1
#   TEAMMATE_1=dougnon1
#   TEAMMATE_2=paoli1
#   TEAMMATE_3=abider1
#   YOUR_FOLDER=ansart1
#   TEAM_FOLDER=loki

source hackathon_setup.sh
```

This creates `/p/scratch/training2625/ansart1/loki/`, downloads datasets, creates per-task `.venv` folders.

## Clone team repo (owner, after setup)

```bash
mkdir -p /p/home/jusers/ansart1/jureca/code
cd /p/home/jusers/ansart1/jureca/code
git clone https://github.com/Loki-Hackathons/cispa_final.git
cd cispa_final
git pull
```

## Teammate bootstrap (dougnon1, paoli1, abider1)

After owner completes setup:

```bash
jutil env activate -p training2625
cd /p/scratch/training2625/ansart1
wget https://huggingface.co/datasets/SprintML/hackathon/resolve/main/teammate.sh -O teammate.sh
# Edit: OWNER=ansart1, TEAM_FOLDER=loki
source teammate.sh
```

## Every new SSH session

```bash
jutil env activate -p training2625
source ~/.bashrc   # if needed
tmux attach -t hackathon || tmux new -s hackathon
```

## Working on a task

```bash
cd /p/scratch/training2625/ansart1/loki/<dataset-name>
source .venv/bin/activate
python main.py
# or: uv run main.py
```

## SLURM job (finals template)

Use `slurm/templates/1gpu_finals.sh` — account `training2625`, reservation `cispahack`.

```bash
cd cispa_final
mkdir -p logs output
sbatch slurm/templates/1gpu_finals.sh
squeue -u $USER
```

## Folder structure (after owner setup)

```
/p/scratch/training2625/ansart1/
    └── loki/
        ├── <dataset-1>/
        │   ├── .venv/
        │   ├── main.py
        │   └── requirements.txt
        ├── <dataset-2>/
        └── output/
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Permission denied (publickey)` | No JURECA system access yet — wait for project approval + SSH key on JuDoor |
| Cannot access scratch folder | Owner has not run `hackathon_setup.sh`, or wrong OWNER/TEAM_FOLDER in teammate.sh |
| `uv not found` | `curl -LsSf https://astral.sh/uv/install.sh \| sh && source ~/.local/bin/env` |
| Wrong project | Use **`training2625`**, not regional `training2557` |
