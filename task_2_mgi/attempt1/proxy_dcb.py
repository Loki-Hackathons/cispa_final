"""Proxy DCB Stage-1 detector: L_Q, L_R, L_A on MaskGIT VQ-GAN."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from config import AttackConfig, PathsConfig, setup_oned_tokenizer_path


def load_vqgan(ckpt_path: str | Path, device: str | torch.device = "cuda"):
    """Load PretrainedTokenizer (MaskGIT VQ-GAN) with frozen weights."""
    setup_oned_tokenizer_path()
    from modeling.titok import PretrainedTokenizer  # noqa: WPS433

    model = PretrainedTokenizer(str(ckpt_path))
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x.unsqueeze(0)
    return x


def encode_features(vqgan, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return continuous features f and quantized z_q (differentiable via STE)."""
    x = _ensure_bchw(x)
    f = vqgan.encoder(x)
    z_q, _, _ = vqgan.quantize(f)
    return f, z_q


def reconstruct(vqgan, x: torch.Tensor) -> torch.Tensor:
    x = _ensure_bchw(x)
    _, z_q = encode_features(vqgan, x)
    return vqgan.decoder(z_q)


def compute_LQ(vqgan, x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Quantization error L_Q = MSE(z_q, f). Returns per-image scalar if reduction=none."""
    f, z_q = encode_features(vqgan, x)
    per_pixel = (z_q - f).pow(2)
    per_image = per_pixel.flatten(1).mean(dim=1)
    if reduction == "mean":
        return per_image.mean()
    return per_image


def compute_LR(vqgan, x: torch.Tensor, eps: float = 1e-6, reduction: str = "mean") -> torch.Tensor:
    """Double reconstruction ratio L_R."""
    x = _ensure_bchw(x)
    x_hat = reconstruct(vqgan, x)
    x_hat2 = reconstruct(vqgan, x_hat)
    num = F.mse_loss(x, x_hat, reduction="none").flatten(1).mean(dim=1)
    den = F.mse_loss(x_hat, x_hat2, reduction="none").flatten(1).mean(dim=1).clamp(min=eps)
    per_image = num / den
    if reduction == "mean":
        return per_image.mean()
    return per_image


def compute_LA(
    vqgan,
    x: torch.Tensor,
    alpha: float,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    lq = compute_LQ(vqgan, x, reduction="none")
    lr = compute_LR(vqgan, x, eps=eps, reduction="none")
    la = lr + alpha * lq
    if reduction == "mean":
        return la.mean()
    return la


def uint8_to_tensor(images: np.ndarray, device: str | torch.device) -> torch.Tensor:
    """NHWC uint8 -> BCHW float [0, 1]."""
    t = torch.from_numpy(images).to(device=device, dtype=torch.float32) / 255.0
    if t.ndim == 3:
        t = t.unsqueeze(0)
    return t.permute(0, 3, 1, 2).contiguous()


def classify_stage1(la: torch.Tensor | np.ndarray | float, tau_g: float) -> str:
    """Stage-1 label: generated if L_A <= tau_G else non-generated."""
    value = float(la) if not isinstance(la, (torch.Tensor, np.ndarray)) else float(np.asarray(la).item())
    return "G" if value <= tau_g else "non-G"


def _optimal_binary_threshold(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """Maximize accuracy separating pos (G, lower) from neg (natural, higher)."""
    all_scores = np.concatenate([scores_pos, scores_neg])
    candidates = np.unique(all_scores)
    if len(candidates) == 0:
        return 0.0
    best_tau, best_acc = 0.0, -1.0
    for tau in candidates:
        tp = (scores_pos <= tau).mean()
        tn = (scores_neg > tau).mean()
        acc = 0.5 * (tp + tn)
        if acc > best_acc:
            best_acc = acc
            best_tau = float(tau)
    return best_tau


def calibrate_alpha(
    vqgan,
    images_natural: torch.Tensor,
    eps: float = 1e-6,
    batch_size: int = 16,
) -> float:
    """Median L_R / L_Q on natural (M+N) images."""
    ratios: list[float] = []
    n = images_natural.shape[0]
    for start in range(0, n, batch_size):
        batch = images_natural[start : start + batch_size]
        lq = compute_LQ(vqgan, batch, reduction="none").detach().cpu().numpy()
        lr = compute_LR(vqgan, batch, eps=eps, reduction="none").detach().cpu().numpy()
        for q, r in zip(lq, lr):
            if q > eps:
                ratios.append(float(r / q))
    if not ratios:
        return 1.0
    return float(np.median(ratios))


def calibrate_thresholds(
    vqgan,
    images_m: torch.Tensor,
    images_n: torch.Tensor,
    images_g: torch.Tensor,
    alpha: float | None = None,
    eps: float = 1e-6,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Calibrate alpha and tau_G on labeled reference images."""

    def _batch_la(images: torch.Tensor) -> np.ndarray:
        scores: list[np.ndarray] = []
        n = images.shape[0]
        for start in range(0, n, batch_size):
            batch = images[start : start + batch_size]
            la = compute_LA(vqgan, batch, alpha=alpha or 1.0, eps=eps, reduction="none")
            scores.append(la.detach().cpu().numpy())
        return np.concatenate(scores) if scores else np.array([])

    if alpha is None:
        alpha = calibrate_alpha(vqgan, torch.cat([images_m, images_n], dim=0), eps=eps, batch_size=batch_size)

    la_m = _batch_la(images_m)
    la_n = _batch_la(images_n)
    la_g = _batch_la(images_g)

    natural = np.concatenate([la_m, la_n])
    tau_mid = 0.5 * (float(la_g.max()) + float(natural.min()))
    tau_opt = _optimal_binary_threshold(la_g, natural)
    tau_g = tau_opt

    return {
        "alpha": alpha,
        "tau_G": tau_g,
        "tau_G_midpoint": tau_mid,
        "stats": {
            "L_A_M": {"mean": float(la_m.mean()), "std": float(la_m.std()), "min": float(la_m.min()), "max": float(la_m.max())},
            "L_A_N": {"mean": float(la_n.mean()), "std": float(la_n.std()), "min": float(la_n.min()), "max": float(la_n.max())},
            "L_A_G": {"mean": float(la_g.mean()), "std": float(la_g.std()), "min": float(la_g.min()), "max": float(la_g.max())},
        },
    }


def save_calibration(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2))


def load_calibration(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def smoke_check(stats: dict[str, Any]) -> bool:
    """Return True if G has lower L_A than M and N on average."""
    g = stats["stats"]["L_A_G"]["mean"]
    m = stats["stats"]["L_A_M"]["mean"]
    n = stats["stats"]["L_A_N"]["mean"]
    return g < m and g < n
