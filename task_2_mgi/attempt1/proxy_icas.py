"""Optional Stage-2 ICAS proxy for M<->N and fine-grained G->M vs G->N attacks.

Requires RAR-XL generator weights and ImageNet class labels inferred via ResNet-50.
Enable after v1 leaderboard confirms Stage-1 transfer works.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from config import DEFAULT_CLUSTER_SCRATCH, setup_oned_tokenizer_path


RAR_XL_CONFIG = """\
experiment:
  generator_checkpoint: ""

model:
  vq_model:
    codebook_size: 1024
    token_size: 256
    num_latent_tokens: 256
    finetune_decoder: False
    pretrained_tokenizer_weight: ""

  generator:
    hidden_size: 1280
    num_hidden_layers: 32
    num_attention_heads: 16
    intermediate_size: 5120
    dropout: 0.1
    attn_drop: 0.1
    class_label_dropout: 0.1
    image_seq_len: 256
    condition_num_classes: 1000
    use_checkpoint: False
"""


def _find_hf_rar_ckpt() -> Path | None:
    cache = DEFAULT_CLUSTER_SCRATCH / ".cache/hub/models--yucornetto--RAR/snapshots"
    if not cache.is_dir():
        return None
    for snap in sorted(cache.iterdir()):
        ckpt = snap / "rar_xl.bin"
        if ckpt.is_file():
            return ckpt
    return None


def load_rar_generator(
    generator_ckpt: Path | None = None,
    tokenizer_ckpt: Path | None = None,
    device: str = "cuda",
):
    setup_oned_tokenizer_path()
    import demo_util  # noqa: WPS433

    gen_path = generator_ckpt or _find_hf_rar_ckpt()
    if gen_path is None or not gen_path.is_file():
        raise FileNotFoundError("RAR-XL checkpoint not found")

    conf = OmegaConf.create(RAR_XL_CONFIG)
    conf.experiment.generator_checkpoint = str(gen_path)
    if tokenizer_ckpt is not None:
        conf.model.vq_model.pretrained_tokenizer_weight = str(tokenizer_ckpt)

    generator = demo_util.get_rar_generator(conf)
    generator.to(device)
    generator.eval()
    for p in generator.parameters():
        p.requires_grad_(False)
    return generator, conf


def load_class_predictor(device: str = "cuda"):
    from torchvision import models

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def predict_imagenet_class(
    images_bchw: torch.Tensor,
    classifier,
) -> torch.Tensor:
    """Return predicted ImageNet class indices (B,)."""
    device = images_bchw.device
    # preprocess expects PIL or tensor; manual normalize for speed
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = F.interpolate(images_bchw, size=(224, 224), mode="bilinear", align_corners=False)
    x = (x - mean) / std
    logits = classifier(x)
    return logits.argmax(dim=1)


def compute_icas_score(
    generator,
    tokenizer,
    images_bchw: torch.Tensor,
    class_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Simplified ICAS-style score: conditional log-prob minus unconditional.

    Higher score => more likely member (trained on similar conditional).
    This is a proxy — jury detector may differ.
    """
    device = images_bchw.device
    b = images_bchw.shape[0]

    # Encode to discrete tokens via tokenizer (no grad through encode in v1 ICAS eval)
    with torch.no_grad():
        codes = tokenizer.encode(images_bchw)

    cond = class_labels.long().to(device)
    cond_logits = generator.forward(codes, condition=cond, cond_drop_prob=0.0)
    uncond_logits = generator.forward(codes, condition=cond, cond_drop_prob=1.0)

    # Token NLL under conditional vs unconditional
    log_p_cond = F.log_softmax(cond_logits, dim=-1)
    log_p_uncond = F.log_softmax(uncond_logits, dim=-1)
    target = codes.view(b, -1)
    nll_cond = -log_p_cond.gather(-1, target.unsqueeze(-1)).squeeze(-1).mean(dim=1)
    nll_uncond = -log_p_uncond.gather(-1, target.unsqueeze(-1)).squeeze(-1).mean(dim=1)
    return nll_uncond - nll_cond


def build_icas_attack_targets(direction: str) -> str:
    """Map submission direction to ICAS attack goal."""
    if direction in ("M_N",):
        return "lower_icas"  # push toward non-member
    if direction in ("N_M",):
        return "raise_icas"  # push toward member
    if direction in ("G_M",):
        return "raise_icas"
    if direction in ("G_N",):
        return "lower_icas"
    raise ValueError(f"No ICAS target for {direction}")
