"""Carlini-Wagner L2 attack on DCB Stage-1 proxy (L_A)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from config import AttackConfig
from proxy_dcb import classify_stage1, compute_LA
from submission_io import jpeg_roundtrip_uint8


def input_diversity(
    x: torch.Tensor,
    resize_prob: float = 0.4,
    diversity_prob: float = 0.5,
    img_size: int = 256,
) -> torch.Tensor:
    """Standard input diversity for transferability."""
    if torch.rand(1).item() > diversity_prob:
        return x
    b, c, h, w = x.shape
    if torch.rand(1).item() < resize_prob:
        rnd = torch.randint(low=int(0.8 * img_size), high=img_size, size=(1,)).item()
        rescaled = F.interpolate(x, size=(rnd, rnd), mode="bilinear", align_corners=False)
        if rnd < img_size:
            pad = img_size - rnd
            pad_top = torch.randint(0, pad + 1, (1,)).item()
            pad_left = torch.randint(0, pad + 1, (1,)).item()
            x = F.pad(rescaled, (pad_left, pad - pad_left, pad_top, pad - pad_top))
        else:
            crop_top = torch.randint(0, rnd - img_size + 1, (1,)).item()
            crop_left = torch.randint(0, rnd - img_size + 1, (1,)).item()
            x = rescaled[:, :, crop_top : crop_top + img_size, crop_left : crop_left + img_size]
    return x


def _hinge_attack_loss(
    la: torch.Tensor,
    direction: str,
    tau_g: float,
    kappa: float,
) -> torch.Tensor:
    if direction == "to_G":
        return F.relu(la - (tau_g - kappa))
    if direction == "from_G":
        return F.relu((tau_g + kappa) - la)
    raise ValueError(f"Unknown direction: {direction}")


def _float_bchw_to_uint8(x: torch.Tensor) -> np.ndarray:
    adv = x.squeeze(0).permute(1, 2, 0).clamp(0, 1)
    return (adv * 255.0).round().clamp(0, 255).byte().cpu().numpy()


def _to_deployable_uint8(x_float_bchw: torch.Tensor, jpeg_quality: int | None) -> np.ndarray:
    adv_uint8 = _float_bchw_to_uint8(x_float_bchw)
    if jpeg_quality is not None:
        adv_uint8 = jpeg_roundtrip_uint8(adv_uint8, jpeg_quality)
    return adv_uint8


def _la_on_uint8(
    vqgan,
    img_uint8: np.ndarray,
    alpha: float,
    eps: float,
    device: torch.device,
) -> float:
    x = torch.from_numpy(img_uint8).to(device=device, dtype=torch.float32) / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)
    return compute_LA(vqgan, x, alpha=alpha, eps=eps, reduction="mean").item()


def _crosses_stage1(la: float, direction: str, tau_g: float) -> bool:
    if direction == "to_G":
        return la < tau_g
    if direction == "from_G":
        return la > tau_g
    raise ValueError(f"Unknown direction: {direction}")


def _deployable_mse(adv_uint8: np.ndarray, x_orig_uint8: np.ndarray) -> float:
    diff = adv_uint8.astype(np.float32) - x_orig_uint8.astype(np.float32)
    return float(np.mean(diff * diff) / (255.0 * 255.0))


def cw_attack_stage1(
    vqgan,
    x_orig_uint8: np.ndarray,
    direction: str,
    tau_g: float,
    alpha: float,
    cfg: AttackConfig,
) -> tuple[np.ndarray, bool, float, float, float]:
    """
    Run C&W attack for one image.

    Success is measured on deployable pixels (uint8 + optional JPEG), not float32.

    Returns: (adv_uint8, success, mse, la_deployable, la_float)
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    x_orig = torch.from_numpy(x_orig_uint8).to(device=device, dtype=torch.float32) / 255.0
    x_orig = x_orig.permute(2, 0, 1).unsqueeze(0)

    w = torch.arctanh((x_orig * 2.0 - 1.0).clamp(-1 + 1e-6, 1 - 1e-6))
    w = w.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=cfg.lr)

    best_adv_uint8: np.ndarray | None = None
    best_mse = float("inf")
    success = False
    la_deployable = float("nan")
    la_float = float("nan")

    for _ in range(cfg.max_steps):
        optimizer.zero_grad(set_to_none=True)
        x_adv = 0.5 * (torch.tanh(w) + 1.0)

        attack_loss = torch.zeros((), device=device)
        for _ in range(cfg.k_aug):
            x_aug = input_diversity(
                x_adv,
                resize_prob=cfg.resize_prob,
                diversity_prob=cfg.diversity_prob,
                img_size=x_adv.shape[-1],
            )
            la = compute_LA(vqgan, x_aug, alpha=alpha, eps=cfg.eps, reduction="mean")
            attack_loss = attack_loss + _hinge_attack_loss(la, direction, tau_g, cfg.kappa)
        attack_loss = attack_loss / cfg.k_aug

        distortion = F.mse_loss(x_adv, x_orig)
        loss = distortion + cfg.c * attack_loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x_test = 0.5 * (torch.tanh(w) + 1.0)
            la_float = compute_LA(
                vqgan, x_test, alpha=alpha, eps=cfg.eps, reduction="mean"
            ).item()

            adv_uint8 = _to_deployable_uint8(x_test, cfg.submission_jpeg_quality)
            la_dep = _la_on_uint8(vqgan, adv_uint8, alpha, cfg.eps, device)
            if _crosses_stage1(la_dep, direction, tau_g):
                mse = _deployable_mse(adv_uint8, x_orig_uint8)
                if mse < best_mse:
                    best_mse = mse
                    best_adv_uint8 = adv_uint8.copy()
                    la_deployable = la_dep
                    success = True
                if cfg.early_stop:
                    break

    if best_adv_uint8 is None:
        x_final = 0.5 * (torch.tanh(w) + 1.0)
        best_adv_uint8 = _to_deployable_uint8(x_final, cfg.submission_jpeg_quality)
        best_mse = _deployable_mse(best_adv_uint8, x_orig_uint8)
        la_deployable = _la_on_uint8(vqgan, best_adv_uint8, alpha, cfg.eps, device)
        if np.isnan(la_float):
            la_float = _la_on_uint8(
                vqgan,
                _float_bchw_to_uint8(x_final),
                alpha,
                cfg.eps,
                device,
            )

    return best_adv_uint8, success, best_mse, la_deployable, la_float


def _checkpoint_path(output_dir: Path, direction: str, index: int) -> Path:
    return output_dir / "checkpoints" / direction / f"{index:04d}.json"


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def attack_batch(
    vqgan,
    images_uint8: np.ndarray,
    direction: str,
    tau_g: float,
    alpha: float,
    cfg: AttackConfig,
    output_dir: Path,
    start_index: int = 0,
    resume: bool = True,
    on_progress: Any | None = None,
    checkpoint_key: str | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Attack a batch of images; optionally resume from checkpoints."""
    ck_key = checkpoint_key or direction
    n = images_uint8.shape[0]
    results = np.empty_like(images_uint8)
    logs: list[dict[str, Any]] = []

    for i in range(n):
        global_idx = start_index + i
        ckpt = _checkpoint_path(output_dir, ck_key, global_idx)
        if resume and ckpt.is_file():
            data = json.loads(ckpt.read_text())
            results[i] = np.array(data["image"], dtype=np.uint8)
            logs.append(data)
            if on_progress:
                on_progress(i + 1, n, direction)
            continue

        adv, success, mse, la_dep, la_float = cw_attack_stage1(
            vqgan, images_uint8[i], direction, tau_g, alpha, cfg
        )
        results[i] = adv
        entry = {
            "index": global_idx,
            "direction": direction,
            "success": success,
            "mse": mse,
            "la_final": la_dep,
            "la_float": la_float,
            "tau_G": tau_g,
            "submission_jpeg_quality": cfg.submission_jpeg_quality,
            "stage1_label": classify_stage1(la_dep, tau_g),
            "image": adv.tolist(),
        }
        logs.append(entry)
        _save_checkpoint(ckpt, entry)
        if on_progress:
            on_progress(i + 1, n, direction)

    return results, logs
