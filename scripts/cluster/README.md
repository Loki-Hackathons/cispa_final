# Cluster setup — CISPA Grand Finals (Team Loki)

Official organizer guide: [`docs/Hackathon_Setup Finale.md`](../../docs/Hackathon_Setup%20Finale.md)

**Project:** `training2625`

## Systems

| System | Host | Purpose |
|--------|------|---------|
| **JUDAC** | `judac.fz-juelich.de` | Data access, global filesystem — **no GPU** |
| **JURECA** | `jureca.fz-juelich.de` | GPU compute via SLURM (separate grant) |

Only **one person (owner)** runs `hackathon_setup.sh`. Others run `teammate.sh`.

## Prerequisites (JuDoor)

1. Project **`training2625`** approved ✅
2. Under **Systems** → **judac**: sign User Agreement + upload SSH public key
3. Wait ~15 min after adding SSH key
4. SSH to `judac.fz-juelich.de` → TOTP prompt (not `publickey denied`)
5. When JURECA is granted: repeat SSH key + agreement for **jureca** system

## Owner bootstrap (ansart1 — on JUDAC)

```bash
ssh -i ~/.ssh/id_ed25519 \
  -o Ciphers=aes256-ctr \
  -o MACs=hmac-sha2-256-etm@openssh.com \
  ansart1@judac.fz-juelich.de

jutil env activate -p training2625
tmux new -s hackathon

cd /p/scratch/training2625
mkdir -p ansart1 && cd ansart1
wget https://huggingface.co/datasets/SprintML/hackathon/resolve/main/hackathon_setup.sh -O hackathon_setup.sh
wget https://huggingface.co/datasets/SprintML/hackathon/resolve/main/teammate.sh -O teammate.sh

# Values from scripts/cluster/loki.env — or run configure_scripts.sh after cloning repo
source hackathon_setup.sh
```

## Clone team repo

```bash
mkdir -p /p/home/jusers/ansart1/judac/code
cd /p/home/jusers/ansart1/judac/code
git clone --recurse-submodules https://github.com/Loki-Hackathons/cispa_final.git
cd cispa_final
bash scripts/cluster/configure_scripts.sh   # run from directory containing the .sh files
bash scripts/task1/sync_watermark_repos.sh    # pin Task 1 detector submodules
```

## Teammate bootstrap

After owner completes setup — same SSH host (`judac.fz-juelich.de` until JURECA granted):

```bash
jutil env activate -p training2625
cd /p/scratch/training2625/ansart1
wget https://huggingface.co/datasets/SprintML/hackathon/resolve/main/teammate.sh -O teammate.sh
# OWNER=ansart1, TEAM_FOLDER=loki
source teammate.sh
```

## Every new SSH session

```bash
ssh ansart1@judac.fz-juelich.de   # or jureca when GPU access granted
jutil env activate -p training2625
tmux attach -t hackathon || tmux new -s hackathon
```

## SLURM (JURECA only)

```bash
sbatch slurm/templates/1gpu_finals.sh
squeue -A training2625
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Permission denied (publickey)` | SSH key not on JuDoor for **judac** system, or wait ~15 min |
| No GPU / sbatch fails | JURECA not granted yet — use JUDAC for data setup only |
| Cannot access scratch | Owner has not run `hackathon_setup.sh` |

## JSC contacts

- Project advisor: a.herten@fz-juelich.de
- User Services: user-services.jsc@fz-juelich.de
- SC support: sc@fz-juelich.de
