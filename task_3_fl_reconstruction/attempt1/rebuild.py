"""Model reconstruction from state_dict + optimization fallback (Tier 3/4).

We can rebuild the MLP forward pass *exactly* from the parameter shapes, which
enables:
  - iDLG label inference (sign of last-layer gradient),
  - a gradient re-simulation validator (label-free-ish quality proxy),
  - Geiping-style gradient-matching reconstruction as a fallback.

CNN/ViT forward passes need hyper-parameters (stride, padding, pooling, patch
size) that are not fully determined by the state_dict; a best-effort CNN
builder is provided and clearly marked. ViT is left to the prior/optimization
path.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
import utils

_ACT = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "gelu": nn.GELU}


# --------------------------------------------------------------------------- #
# MLP (exact rebuild — matches net.0 / net.2 / net.5 naming)
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    def __init__(self, dims: list[int], activation: str):
        super().__init__()
        A = _ACT[activation]
        # positions: 0 Linear, 1 act, 2 Linear, 3 act, 4 Identity, 5 Linear
        self.net = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            A(),
            nn.Linear(dims[1], dims[2]),
            A(),
            nn.Identity(),
            nn.Linear(dims[2], dims[3]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


def build_mlp(state: dict, activation: str) -> MLP:
    d0 = state["net.0.weight"].shape       # (1024, 12288)
    d2 = state["net.2.weight"].shape       # (1024, 1024)
    d5 = state["net.5.weight"].shape       # (200, 1024)
    dims = [d0[1], d0[0], d2[0], d5[0]]
    model = MLP(dims, activation)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# iDLG-style label inference
# --------------------------------------------------------------------------- #
def infer_labels(grads: dict, k: int = config.BATCH) -> torch.Tensor:
    """Guess the k most likely present class labels from the last-layer bias.

    For softmax cross-entropy, dL/db_logit[c] = mean(p_c) - freq(c); classes
    truly present in the batch push their entry negative. We return the k
    classes with the most-negative gradient (with repetition if needed).
    """
    bias_key = None
    for name in ("head.bias", "net.5.bias"):
        if name in grads:
            bias_key = name
            break
    if bias_key is None:
        return torch.zeros(k, dtype=torch.long)
    g = grads[bias_key]
    order = torch.argsort(g)  # most negative first
    labels = order[: min(k, len(order))]
    if len(labels) < k:  # pad by cycling
        reps = (k + len(labels) - 1) // len(labels)
        labels = labels.repeat(reps)[:k]
    return labels.long()


# --------------------------------------------------------------------------- #
# Gradient re-simulation validator (MLP)
# --------------------------------------------------------------------------- #
def flat_grad(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
    model.zero_grad(set_to_none=True)
    out = model(x)
    loss = F.cross_entropy(out, y)
    grads = torch.autograd.grad(loss, [p for p in model.parameters()], create_graph=x.requires_grad)
    return list(grads)


def target_grad_list(model: nn.Module, grads: dict) -> list[torch.Tensor]:
    """Order the provided gradient dict to match model.parameters()."""
    ordered = []
    for name, _ in model.named_parameters():
        ordered.append(grads[name])
    return ordered


def cosine_grad_distance(gs: list[torch.Tensor], ts: list[torch.Tensor]) -> torch.Tensor:
    num = sum((g * t).sum() for g, t in zip(gs, ts))
    gn = torch.sqrt(sum((g * g).sum() for g in gs)).clamp_min(1e-12)
    tn = torch.sqrt(sum((t * t).sum() for t in ts)).clamp_min(1e-12)
    return 1.0 - num / (gn * tn)


# --------------------------------------------------------------------------- #
# Geiping-style gradient matching (Tier 3 fallback, GPU recommended)
# --------------------------------------------------------------------------- #
def gradient_match(
    model: nn.Module,
    grads: dict,
    labels: torch.Tensor,
    n_images: int = config.BATCH,
    steps: int = 4000,
    lr: float = 0.1,
    tv_weight: float = 1e-2,
    device: str = None,
    init: torch.Tensor = None,
    seed: int = config.SEED,
) -> torch.Tensor:
    """Reconstruct a batch of images by matching the full-model gradient.

    Minimizes  1 - cos(grad(x), G) + tv_weight * TV(x).  Warm-start with `init`
    (e.g. analytic candidates) for much faster convergence.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    targets = [t.to(device) for t in target_grad_list(model, grads)]
    labels = labels.to(device)[:n_images]

    torch.manual_seed(seed)
    if init is not None:
        x = init[:n_images].clone().to(device)
        if x.shape[0] < n_images:
            pad = torch.rand(n_images - x.shape[0], *config.IMG_SHAPE, device=device)
            x = torch.cat([x, pad], 0)
    else:
        x = torch.rand(n_images, *config.IMG_SHAPE, device=device)
    x.requires_grad_(True)

    opt = torch.optim.Adam([x], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        gs = flat_grad(model, x, labels)
        loss = cosine_grad_distance(gs, targets) + tv_weight * utils.total_variation(x).mean()
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            x.clamp_(0, 1)
        if step % 500 == 0:
            print(f"    [opt] step {step:4d}  loss {float(loss):.4f}")

    return utils.to_unit(x.detach().cpu()).float()
