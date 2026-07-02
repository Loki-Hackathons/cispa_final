"""Pre-download the entropy proxy model on a JURECA login node (has internet).

Compute nodes are offline; entropy_pass.py loads with local_files_only.
Set HF_HOME to the team scratch cache before running (see run_entropy.sh).
"""

from transformers import AutoModelForCausalLM

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
AutoModelForCausalLM.from_pretrained(NAME)
print("cached:", NAME)
