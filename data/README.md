# Local datasets

Large Hugging Face dataset clones live here on your laptop. **Never commit them** — they are gitignored.

## Task 1 — Text Watermark Localization

```powershell
cd cispa_final/data
git lfs install
git clone https://huggingface.co/datasets/SprintML/watermark_localization
```

Cluster path (JURECA): `/p/scratch/training2625/ansart1/loki/watermark_localization/`

Files after clone: `train.jsonl`, `validation.jsonl`, `test.jsonl`, `task_template.py`, `watermark_config.yaml`.

**Browse labeled docs locally (ground truth highlighted):**

```powershell
cd cispa_final
python scripts/task1/view_dataset.py
# → http://127.0.0.1:8765
```

## Other tasks (when needed)

```powershell
git clone https://huggingface.co/datasets/SprintML/MGI
git clone https://huggingface.co/datasets/SprintML/FL_Data_Reconstruction
```

Or pull from cluster:

```powershell
scp -r ansart1@jureca.fz-juelich.de:/p/scratch/training2625/ansart1/loki/watermark_localization .\watermark_localization
```
