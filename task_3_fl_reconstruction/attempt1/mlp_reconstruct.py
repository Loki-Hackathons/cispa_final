"""Reconstruct MLP model images (models 1, 4, 5, 8) and patch them into a base
submission, mirroring how cnn_invert.py handles the CNN models.

Key facts that drive the strategy (different from CNN!):
  * net.0 sees the flattened RGB image directly, so row_i = gW0_i/gb0_i is an
    input reconstruction, exact when a single sample activated neuron i.
  * ReLU (models 5, 8): many neurons are activated by a single image -> clean
    rows exist; the lever is SELECTING the cleanest 128 distinct ones.
  * sigmoid/tanh (models 1, 4): every sample mixes into every neuron -> rows
    are blurry mixtures the selection cannot fix; the only real lever is
    gradient matching against the EXACTLY rebuilt MLP forward (--refine).

Two modes:
  analytic (default): improved candidate ranking + dedup selection.
  --refine          : Geiping gradient matching (exact forward) warm-started
                      from the analytic selection. Intended for 1 and 4.

Usage:
  # safe: better analytic selection for all four, patched into the best base
  python mlp_reconstruct.py --base submission_all_m3_5k.pt --out sub_mlp_sel.pt \
      --models 1 4 5 8

  # GPU: gradient-matching refine for the smooth-activation models only
  python mlp_reconstruct.py --base submission_all_m3_5k.pt --out sub_mlp_refine.pt \
      --models 1 4 --refine --steps 4000
"""
from __future__ import annotations

import argparse
import os

import torch

import config
import extract
import rebuild
import utils


def analytic_rows(info: extract.ModelInfo):
    """Return (rows, valid_mask) for net.0: row_i = gW0_i / gb0_i."""
    gW = info.grads["net.0.weight"]
    gb = info.grads["net.0.bias"]
    valid = gb.abs() > config.EPS
    if int(valid.sum()) == 0:
        return None, valid
    rows = gW[valid] / gb[valid].unsqueeze(1)
    return rows, valid


def own_activation(i: int, rows: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Pre-activation a_i(r_i) of each row's own neuron, using MODEL weights.

    For ReLU, a clean reconstruction of the single activating image satisfies
    a_i > 0 (the neuron was active during the client's forward pass), so this
    is a principled 'this row is a real activating image' signal. For smooth
    activations it is only weakly informative; callers gate on it for relu only.
    """
    state = utils.load_state(i)
    W0 = state["net.0.weight"][valid]
    b0 = state["net.0.bias"][valid]
    return (W0 * rows).sum(dim=1) + b0


def select_analytic(i: int, info: extract.ModelInfo) -> torch.Tensor:
    rows, valid = analytic_rows(info)
    if rows is None:
        return torch.rand(config.BATCH, *config.IMG_SHAPE).float()

    imgs = utils.flat_to_image(rows, config.IMG_SHAPE)     # (N,3,64,64) normed
    quality = utils.quality_score(imgs)

    if info.activation == "relu":
        # Prefer rows whose own neuron is actually active (real single-image
        # reconstructions) before falling back to the rest. We do NOT drop the
        # inactive ones outright, we just rank them last, so we still fill 128.
        act = own_activation(i, rows, valid)
        active = (act > 0).float()
        # Rank: active first (big offset), then by quality within each group.
        qn = (quality - quality.min()) / (quality.max() - quality.min() + 1e-8)
        score = active * 10.0 + qn
    else:
        score = quality

    return utils.dedup_select(imgs, score, k=config.BATCH)


def refine_gradient_match(i: int, info: extract.ModelInfo, init: torch.Tensor,
                          steps: int, lr: float, tv: float,
                          device: str) -> torch.Tensor:
    """Exact-forward gradient matching for smooth-activation MLPs.

    The MLP forward is rebuilt strictly from the state_dict (rebuild.build_mlp),
    so unlike CNN/ViT this is not a guessed forward. Warm-started from the
    analytic selection.
    """
    state = utils.load_state(i)
    model = rebuild.build_mlp(state, info.activation)
    labels = rebuild.infer_labels(info.grads)
    return rebuild.gradient_match(
        model, info.grads, labels,
        n_images=config.BATCH, steps=steps, lr=lr, tv_weight=tv,
        device=device, init=init,
    )


def reconstruct(i: int, refine: bool, steps: int, lr: float, tv: float,
                device: str) -> torch.Tensor:
    grad = utils.load_gradient(i)
    info = extract.introspect(i, grad)
    if info.family != "mlp":
        raise ValueError(f"model{i} is {info.family}, not mlp")

    imgs = select_analytic(i, info)
    tag = f"analytic({info.activation})"

    if refine:
        opt = refine_gradient_match(i, info, imgs, steps, lr, tv, device)
        imgs = opt
        tag = f"refine({info.activation},{steps})"

    imgs = utils.to_unit(imgs).float()
    print(f"model{i:2d} [{info.activation:7s}] -> {tag} "
          f"quality={float(utils.quality_score(imgs).mean()):.4f}")
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=int, nargs="*", default=[1, 4, 5, 8])
    ap.add_argument("--base", type=str, default="submission_all_m3_5k.pt")
    ap.add_argument("--out", type=str, default="submission_mlp.pt")
    ap.add_argument("--refine", action="store_true",
                    help="gradient-matching refine (exact MLP forward); use for 1,4")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--tv", type=float, default=1e-2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(config.SEED)

    if os.path.exists(args.base):
        submission = torch.load(args.base, weights_only=False)
        print(f"[mlp] loaded base {args.base}")
    else:
        submission = {f"model{k}": torch.rand(config.BATCH, *config.IMG_SHAPE)
                      for k in range(1, config.NUM_MODELS + 1)}
        print("[mlp] no base found; non-MLP models start from noise")

    print(f"[mlp] device={device} models={args.models} refine={args.refine}")
    for i in args.models:
        submission[f"model{i}"] = reconstruct(
            i, args.refine, args.steps, args.lr, args.tv, device)

    path = utils.save_submission(submission, args.out)
    print(f"[mlp] wrote {path} (validated OK)")


if __name__ == "__main__":
    main()
