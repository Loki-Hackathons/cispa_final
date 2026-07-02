# Hackathon Setup (Finale)

# Hackathon Environment Guide

This guide explains how to set up and use the shared environment.
### Always change the project id before sharing. (eg. training2625)

You are given access to two scripts:

- `hackathon_setup.sh`  https://huggingface.co/datasets/SprintML/hackathon/resolve/main/hackathon_setup.sh
- `teammate.sh`
https://huggingface.co/datasets/SprintML/hackathon/resolve/main/teammate.sh

(Instruction on downloading these files is provided below)

# 1. Decide roles

Before starting:

- One person (Team Lead) = **Owner**
- Others = **Teammates**

Only one person from the team should run the OWNER setup.

# 2. Owner setup (run once)

## Create your folder and download the scripts

Owner must manually create a folder in scratch:

(This `<team_lead-username>` is where all your data will reside and only you and your team can access it)

```
cd /p/scratch/training2625
mkdir -p <team_lead-username>
cd <team_lead-username>
wget https://huggingface.co/datasets/SprintML/hackathon/resolve/main/hackathon_setup.sh -O hackathon_setup.sh
wget https://huggingface.co/datasets/SprintML/hackathon/resolve/main/teammate.sh -O teammate.sh
```

### Step 1: Edit the script

Open `hackathon_setup.sh` and fill:

```bash
OWNER="<your-username>"  # Your Juelich username

TEAMMATE_1="<teammate1>" # Your teammates' Juelich usernames. Leave unused as it is
TEAMMATE_2="<teammate2>"
TEAMMATE_3="<teammate3>"

YOUR_FOLDER="<team_lead-username>" # You created in previous step
TEAM_FOLDER="<team-name>" # Shared team folder inside YOUR_FOLDER
```

Do not leave these empty.

### Step 2: Run setup - ONLY OWNER

```bash
source hackathon_setup.sh
```

### What this does

- creates shared folder in `/p/scratch/training2625`
- restricts access to only team members
- downloads datasets
- creates per-task environments (`.venv`)
- sets shared cache (uv + HuggingFace)

# 3. Teammate setup (each TEAMMATE except OWNER)

<aside>
💡

Run this only after the owner completes setup.

</aside>

### Step 1: Edit the script

Open `teammate.sh` and fill:

```bash
OWNER="<owner-username>"
TEAM_FOLDER="<team-name>" # Same as what Owner used in hackathon_setup.sh script
```

---

### Step 2: Run

```bash
source teammate.sh
```

### What this does

- activates project environment
- sets shared cache paths
- installs `uv` if missing
- verifies access to shared folder

# 4. Working on tasks

Go to your task:

```bash
cd /p/scratch/training2625/<owner>/<team-folder>/<dataset-name>
```

Activate environment:

```bash
source .venv/bin/activate
```

Run code:

```bash
python main.py
```

Or directly:

```bash
uv run main.py
```

# 5. Every new session

Run:

```bash
jutil env activate -p training2625
```

If needed:

```bash
source ~/.bashrc
```

# 6. Folder structure

```
/p/scratch/training2625/<owner>/
    └── <team-folder>/
        ├── <dataset-1>/
        │   ├── .venv/
        │   ├── main.py
        │   └── requirements.txt
        ├── <dataset-2>/
        └── output/
```

# 7. Rules

- do not change permissions manually
- do not delete shared folders
- do not recreate `.venv` unless necessary
- avoid installing random packages in shared environment
- keep all work inside the team folder
- store logs in `output/`

# 8. Jobs

Create main.sh

```powershell
#!/bin/bash
#SBATCH --account=training2625
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu
#SBATCH --output=output/%j.out  
#SBATCH --error=output/%j.err  

# Go to folder where the code file is located
cd your-folder
# Activate the environment from the directory
source .venv/bin/activate
# Run the file: task_template.py
srun python code.py  
echo "done!"
```

Submit jobs using:

```bash
sbatch main.sh
```

# 9. Common issues

### Cannot access folder

- check correct `OWNER` and `TEAM_FOLDER`
- ensure owner ran setup

### uv not found

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
```

### environment not working

```bash
source .venv/bin/activate
```
