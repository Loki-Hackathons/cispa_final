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
# CNN (best-effort rebuild from conv/fc1/head + feature_shape)
# --------------------------------------------------------------------------- #
class SimpleCNN(nn.Module):
    """conv -> act -> adaptive_avg_pool(feature_shape) -> fc1 -> act -> head.

    The exact pooling/padding of the original net is unknown from the
    state_dict alone. We use a 'same' conv (pad = k//2, stride 1) followed by
    an adaptive average pool to the reported feature map size. This is smooth
    and differentiable, which is what gradient matching needs; the first conv
    (which we know exactly) dominates the input-facing gradient, so imperfect
    pooling still pulls pixels toward the right low-level structure.
    """

    def __init__(self, state: dict, activation: str, feature_shape: tuple):
        super().__init__()
        A = _ACT[activation]
        cw = state["conv.weight"]              # (Cout, Cin, kh, kw)
        cout, cin, kh, kw = cw.shape
        self.conv = nn.Conv2d(cin, cout, (kh, kw), padding=(kh // 2, kw // 2))
        self.act1 = A()
        self.feat_hw = (int(feature_shape[1]), int(feature_shape[2]))
        fc1_w = state["fc1.weight"]            # (hidden, cout*Hf*Wf)
        self.fc1 = nn.Linear(fc1_w.shape[1], fc1_w.shape[0])
        self.act2 = A()
        head_w = state["head.weight"]          # (num_classes, hidden)
        self.head = nn.Linear(head_w.shape[1], head_w.shape[0])
        self.load_state_dict(state, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.conv(x))
        h = F.adaptive_avg_pool2d(h, self.feat_hw)
        h = h.flatten(1)
        h = self.act2(self.fc1(h))
        return self.head(h)


def build_cnn(state: dict, activation: str, feature_shape: tuple) -> SimpleCNN:
    model = SimpleCNN(state, activation, feature_shape)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# ViT (guarded timm-style rebuild; fast-fails if the arch does not match)
# --------------------------------------------------------------------------- #
# Submodule names below mirror timm's ViT exactly (patch_embed.proj, attn.qkv,
# attn.proj, mlp.fc1, mlp.fc2, ...) so the reconstructed model's parameter
# names line up 1:1 with the provided gradient dict. If anything differs,
# strict state_dict loading raises and the caller keeps the fallback.
class _PatchEmbed(nn.Module):
    def __init__(self, dim: int, patch: int):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, (patch, patch), stride=(patch, patch))


class _Attn(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)


class _ViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_hidden: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attn(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, mlp_hidden)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        h = self.norm1(x)
        qkv = self.attn.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x + self.attn.proj(out)
        h = self.norm2(x)
        x = x + self.mlp.fc2(F.gelu(self.mlp.fc1(h)))
        return x


class SimpleViT(nn.Module):
    """Minimal timm-style ViT reconstructed from a state_dict.

    Assumes standard timm naming (patch_embed.proj, cls_token, pos_embed,
    blocks.{i}.{norm1,attn.qkv,attn.proj,norm2,mlp.fc1,mlp.fc2}, norm, head).
    If any assumption is wrong, strict state_dict loading raises and the caller
    keeps the noise/analytic fallback instead of wasting a GPU slot.
    """

    def __init__(self, state: dict):
        super().__init__()
        pw = state["patch_embed.proj.weight"]      # (dim, 3, P, P)
        dim, _, ph, pw_ = pw.shape
        self.patch_embed = _PatchEmbed(dim, ph)
        self.pos_embed = nn.Parameter(state["pos_embed"].clone())
        n_pos = self.pos_embed.shape[1]
        n_patch = (64 // ph) * (64 // pw_)
        self.has_cls = "cls_token" in state and (n_pos == n_patch + 1)
        if self.has_cls:
            self.cls_token = nn.Parameter(state["cls_token"].clone())

        n_blocks = 0
        while f"blocks.{n_blocks}.norm1.weight" in state:
            n_blocks += 1
        num_heads = max(1, dim // 64)
        mlp_hidden = state["blocks.0.mlp.fc1.weight"].shape[0]
        self.blocks = nn.ModuleList(
            [_ViTBlock(dim, num_heads, mlp_hidden) for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(dim)
        head_w = state["head.weight"]
        self.head = nn.Linear(head_w.shape[1], head_w.shape[0])

        self.load_state_dict(state, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h = self.patch_embed.proj(x).flatten(2).transpose(1, 2)   # (B, N, dim)
        if self.has_cls:
            cls = self.cls_token.expand(B, -1, -1)
            h = torch.cat([cls, h], dim=1)
        h = h + self.pos_embed
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        pooled = h[:, 0] if self.has_cls else h.mean(dim=1)
        return self.head(pooled)


def build_vit(state: dict) -> SimpleViT:
    model = SimpleViT(state)
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
    """Order the provided gradient dict to match model.parameters().

    Raises a clear error (caught by callers) if the rebuilt model exposes a
    parameter that the gradient dict does not, which signals an architecture
    mismatch we should not silently optimize against.
    """
    ordered = []
    for name, _ in model.named_parameters():
        if name not in grads:
            raise KeyError(f"gradient dict missing parameter '{name}'")
        ordered.append(grads[name])
    return ordered


def cosine_grad_distance(gs: list[torch.Tensor], ts: list[torch.Tensor]) -> torch.Tensor:
    num = sum((g * t).sum() for g, t in zip(gs, ts))
    gn = torch.sqrt(sum((g * g).sum() for g in gs)).clamp_min(1e-12)
    tn = torch.sqrt(sum((t * t).sum() for t in ts)).clamp_min(1e-12)
    return 1.0 - num / (gn * tn)


def resim_distance(model: nn.Module, imgs: torch.Tensor, labels: torch.Tensor,
                   grads: dict, device: str = None) -> float:
    """Label-free-ish quality proxy: how well `imgs` reproduce the observed
    gradient under the (known or assumed) forward model. Lower is better.

    Uses the *actual leaked gradient* as the reference, so it never touches the
    leaderboard and cannot cause public-split overfitting. Returned as a plain
    float for cheap comparison / model selection. We only need first-order
    param gradients here (x is not optimized), so no create_graph.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(True)
    x = imgs.to(device).clone()             # no grad on x -> first-order only
    y = labels.to(device)[: x.shape[0]]
    targets = [t.to(device) for t in target_grad_list(model, grads)]
    gs = flat_grad(model, x, y)
    return float(cosine_grad_distance(gs, targets).detach().cpu())


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
    # Params MUST require grad: gradient matching simulates dL/dparams(x) and
    # differentiates the match w.r.t. x (second-order). Only x is optimized;
    # flat_grad zeroes model grads each step so nothing accumulates on params.
    for p in model.parameters():
        p.requires_grad_(True)

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

    # Track the best-matching iterate: gradient matching can oscillate/diverge,
    # so we return the lowest-distance snapshot rather than the final step.
    best_dist = float("inf")
    best_x = x.detach().clone()

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        gs = flat_grad(model, x, labels)
        match = cosine_grad_distance(gs, targets)
        loss = match + tv_weight * utils.total_variation(x).mean()
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            x.clamp_(0, 1)
            d = float(match.detach())
            if d < best_dist:
                best_dist = d
                best_x = x.detach().clone()
        if step % 500 == 0:
            print(f"    [opt] step {step:4d}  match {d:.4f}  best {best_dist:.4f}")

    print(f"    [opt] done: best gradient-match distance {best_dist:.4f}")
    return utils.to_unit(best_x.detach().cpu()).float()
