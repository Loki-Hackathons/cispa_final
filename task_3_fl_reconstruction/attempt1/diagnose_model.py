"""Visual + numeric diagnostic for CNN fc1 analytic targets and inversion.

Modes:
  1. Full diagnostic (needs inverted submission):
       python diagnose_model.py --model 6 --inv inv6.pt
  2. Analytic-only (no GPU inversion needed — tests contamination hypothesis):
       python diagnose_model.py --model 6 --analytic-only
  3. Compare several models:
       python diagnose_model.py --compare 6 3 7 10 12

Outputs under output/diagnose/:
  - model{N}_target.png   analytic fc1 targets (RGB collapse)
  - model{N}_pred.png     conv(x) of inverted images (if --inv given)
  - model{N}_image.png    inverted RGB images (if --inv given)
  - model{N}_conf.png     confidence vs quality scatter (selected 128)
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import config
import fc1_analytic
import utils
from cnn_invert import conv_features


def collapse_to_rgb(feat: torch.Tensor) -> torch.Tensor:
    """(N, C, H, W) feature maps -> (N, 3, 64, 64) preview in [0,1]."""
    n, c, h, w = feat.shape
    rows = feat.reshape(n, c * h * w)
    return utils.features_to_image(rows, c, h, w)


def grid(imgs: torch.Tensor, cols: int = 8, labels: list[str] | None = None) -> Image.Image:
    imgs = imgs.clamp(0, 1).cpu()
    n, _, h, w = imgs.shape
    rows = (n + cols - 1) // cols
    label_h = 14 if labels else 0
    canvas = torch.zeros(3, rows * (h + label_h), cols * w)
    pil = Image.new("RGB", (cols * w, rows * (h + label_h)), (32, 32, 32))
    draw = ImageDraw.Draw(pil) if labels else None
    for i in range(n):
        r, c = divmod(i, cols)
        y0 = r * (h + label_h) + label_h
        canvas[:, y0:y0 + h, c * w:(c + 1) * w] = imgs[i]
        if draw and labels and i < len(labels):
            draw.text((c * w + 2, r * (h + label_h) + 1), labels[i], fill=(220, 220, 80))
    arr = (canvas.permute(1, 2, 0).numpy() * 255).astype("uint8")
    base = Image.fromarray(arr)
    if labels:
        base.paste(pil, (0, 0), pil)
    return base


def conf_scatter(cands: fc1_analytic.Fc1Candidates, sel_idx: torch.Tensor,
                 path: str) -> None:
    """Save a simple confidence-vs-quality scatter for selected rows."""
    w, h = 480, 320
    img = Image.new("RGB", (w, h), (24, 24, 28))
    draw = ImageDraw.Draw(img)
    q = cands.quality.float()
    c = cands.confidence.float()
    qn = (q - q.min()) / (q.max() - q.min() + 1e-8)
    sel = set(sel_idx.tolist())

    def px(v, lo, hi, span):
        return int((float(v) - lo) / max(hi - lo, 1e-8) * span)

    for i in range(q.numel()):
        x = 40 + px(qn[i], 0, 1, w - 60)
        y = h - 30 - px(c[i], 0, 1, h - 50)
        color = (80, 200, 120) if i in sel else (90, 90, 100)
        r = 4 if i in sel else 2
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    draw.text((10, 8), "quality (norm) ->", fill=(180, 180, 180))
    draw.text((8, h // 2), "conf", fill=(180, 180, 180))
    draw.text((w - 120, h - 22), "green=selected", fill=(120, 200, 140))
    img.save(path)


def diagnose_one(model: int, inv_path: str | None, n: int, outdir: str,
                 analytic_only: bool) -> dict:
    os.makedirs(outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cands = fc1_analytic.extract_fc1_candidates(model)
    rep = fc1_analytic.model_confidence_report(cands)
    sel_idx = rep["selected_indices"]
    target = cands.feats[sel_idx]
    n = min(n, target.shape[0])

    tgt_rgb = collapse_to_rgb(target[:n])
    labels = [f"c={float(cands.confidence[sel_idx[i]]):.2f}" for i in range(n)]
    grid(tgt_rgb, labels=labels).save(f"{outdir}/model{model}_target.png")

    comp = cands.components
    print(
        f"\nmodel{model} ({cands.activation}, {cands.feature_shape}) "
        f"n={rep['n_candidates']}"
    )
    print(
        f"  confidence  all: mean={rep['conf_mean_all']:.3f} "
        f"median={rep['conf_median_all']:.3f} p90={rep['conf_p90_all']:.3f} "
        f"high(>0.5)={rep['conf_high_frac']:.1%}"
    )
    print(
        f"  selected128 conf: mean={rep['conf_mean_selected']:.3f} "
        f"min={rep['conf_min_selected']:.3f}  quality_mean={rep['quality_mean_selected']:.3f}"
    )
    for name in ("uniqueness", "sparsity", "gb_sweet", "norm_ok"):
        v = comp[name][sel_idx]
        print(f"  sel {name:11s}: mean={float(v.mean()):.3f} min={float(v.min()):.3f}")

    conf_scatter(cands, sel_idx, f"{outdir}/model{model}_conf.png")

    result = {"model": model, **rep, "feature_mse": None}

    if analytic_only or inv_path is None:
        print(f"  [analytic-only] wrote {outdir}/model{model}_target.png + _conf.png")
        return result

    grad = utils.load_gradient(model)
    state = utils.load_state(model)
    act = grad["activation"]
    hw = target.shape[-2:]

    sub = torch.load(inv_path, weights_only=False)
    x = sub[f"model{model}"].to(device)

    with torch.no_grad():
        pred = conv_features(x, state, act, hw)

    pred_rgb = collapse_to_rgb(pred[:n].cpu())
    x_rgb = x[:n].cpu()
    grid(pred_rgb).save(f"{outdir}/model{model}_pred.png")
    grid(x_rgb).save(f"{outdir}/model{model}_image.png")

    mse = float(F.mse_loss(pred.cpu(), target).item())
    per_slot = ((pred.cpu() - target) ** 2).mean(dim=(1, 2, 3))
    worst = int(per_slot.argmax())
    result["feature_mse"] = mse
    print(
        f"  feature_mse={mse:.6f}  worst_slot={worst} "
        f"mse={float(per_slot[worst]):.4f} conf={float(cands.confidence[sel_idx[worst]]):.3f}"
    )
    print(f"  wrote {outdir}/model{model}_{{target,pred,image,conf}}.png")
    return result


def compare_models(models: list[int], inv_path: str | None, outdir: str) -> None:
    print("\n=== FC1 isolation confidence comparison ===")
    fc1_analytic.print_confidence_table(models)
    rows = []
    for m in models:
        rows.append(diagnose_one(m, inv_path, n=8, outdir=outdir, analytic_only=inv_path is None))
    if any(r.get("feature_mse") is not None for r in rows):
        print("\n=== Feature MSE (lower = better inversion) ===")
        for r in rows:
            if r["feature_mse"] is not None:
                print(f"  model{r['model']:2d}: mse={r['feature_mse']:.6f} "
                      f"conf_sel={r['conf_mean_selected']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=int, default=None)
    ap.add_argument("--compare", type=int, nargs="*", default=None,
                    help="compare several models (e.g. 6 3 7 10)")
    ap.add_argument("--inv", type=str, default=None,
                    help="submission .pt with inverted CNN models")
    ap.add_argument("--analytic-only", action="store_true",
                    help="skip inversion comparison; only fc1 targets + confidence")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--outdir", type=str, default="output/diagnose")
    args = ap.parse_args()

    if args.compare:
        compare_models(args.compare, args.inv, args.outdir)
        return
    if args.model is None:
        ap.error("provide --model N or --compare N1 N2 ...")

    diagnose_one(
        args.model,
        inv_path=args.inv,
        n=args.n,
        outdir=args.outdir,
        analytic_only=args.analytic_only or args.inv is None,
    )


if __name__ == "__main__":
    main()
