"""Analytic gradient extraction (the workhorse for MLP + CNN models).

Core identity (Boenisch et al. 2022, eq. 5-6): for a linear layer y = act(Wx+b),
    dL/dW_i = (dL/db_i) * x_i^T   =>   x_i = (dL/db_i)^{-1} * dL/dW_i
so each *row* of the weight-gradient, divided by the matching bias-gradient
scalar, reconstructs an input (perfectly when a single sample activated that
neuron; a weighted overlay otherwise).

This holds for any pointwise activation; ReLU just isolates single samples
best. We therefore always try it, then rank/dedup the rows downstream.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

import config
import utils


@dataclass
class ModelInfo:
    idx: int
    family: str            # mlp | cnn | vit
    activation: str        # relu | tanh | sigmoid | gelu
    feature_shape: tuple   # (C,H,W) input feature shape reported by the task
    grads: dict            # name -> gradient tensor
    tier: str              # T1 | T2 | T3
    note: str = ""


def introspect(i: int, grad: dict) -> ModelInfo:
    family = grad["family"]
    act = grad["activation"]
    fshape = tuple(grad["feature_shape"])
    grads = grad["gradients"]

    if family == "mlp":
        tier = "T1" if act == "relu" else "T3"  # relu isolates; others overlay
        note = "analytic on first Linear (net.0)"
    elif family == "cnn":
        tier = "T2" if act == "relu" else "T3"
        note = "analytic on fc1 (conv feature space) + upscale"
    else:  # vit
        tier = "T3"
        note = "no clean analytic path -> optimization / prior"
    return ModelInfo(i, family, act, fshape, grads, tier, note)


def _analytic_rows(gW: torch.Tensor, gb: torch.Tensor, eps: float = config.EPS):
    """Return rows x_i = gW_i / gb_i for neurons with |gb_i| > eps."""
    valid = gb.abs() > eps
    if valid.sum() == 0:
        return None, valid
    rows = gW[valid] / gb[valid].unsqueeze(1)
    return rows, valid


def extract_mlp(info: ModelInfo) -> torch.Tensor:
    """MLP: first layer sees the flattened image directly (12288 = 3*64*64)."""
    gW = info.grads["net.0.weight"]     # (n_neurons, 12288)
    gb = info.grads["net.0.bias"]       # (n_neurons,)
    rows, valid = _analytic_rows(gW, gb)
    if rows is None:
        return torch.empty(0, *config.IMG_SHAPE)

    in_features = gW.shape[1]
    if in_features == config.IMG_FLAT:
        imgs = utils.flat_to_image(rows, config.IMG_SHAPE)
    else:
        imgs = utils.flat_to_image(rows, info.feature_shape)
    return imgs


def extract_cnn(info: ModelInfo) -> torch.Tensor:
    """CNN: reconstruct fc1 input (flattened conv feature map), then upscale.

    conv.weight gives the conv output-channel count; fc1's in_features / that
    count gives the spatial size. Reconstructed feature maps are spatially
    aligned with the image for stride-1 'same' convs, so upscaling yields a
    recognizable guess (real fidelity comes later from optimization).
    """
    if "fc1.weight" not in info.grads or "fc1.bias" not in info.grads:
        return torch.empty(0, *config.IMG_SHAPE)

    gW = info.grads["fc1.weight"]       # (hidden, conv_out_flat)
    gb = info.grads["fc1.bias"]
    rows, valid = _analytic_rows(gW, gb)
    if rows is None:
        return torch.empty(0, *config.IMG_SHAPE)

    conv_out_flat = gW.shape[1]
    conv_channels = 1
    if "conv.weight" in info.grads:
        conv_channels = info.grads["conv.weight"].shape[0]  # out channels
    hc, wc = utils.infer_square(conv_out_flat, conv_channels)
    # Guard against rounding leaving a mismatch.
    if conv_channels * hc * wc != conv_out_flat:
        conv_channels = 1
        hc, wc = utils.infer_square(conv_out_flat, 1)
        if hc * wc != conv_out_flat:
            return torch.empty(0, *config.IMG_SHAPE)
    return utils.features_to_image(rows, conv_channels, hc, wc)


def extract_analytic(info: ModelInfo) -> torch.Tensor:
    """Route to the right analytic extractor. Returns (N,3,64,64) candidates."""
    try:
        if info.family == "mlp":
            return extract_mlp(info)
        if info.family == "cnn":
            return extract_cnn(info)
    except Exception as e:  # never let one model kill the run
        print(f"  [extract] model{info.idx}: analytic failed: {e}")
    return torch.empty(0, *config.IMG_SHAPE)
