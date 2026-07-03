#!/usr/bin/env python3
"""Unified per-cell MGI attack.

Each of the 6 directions targets an exact detector cell (Stage 1, Stage 2):

  Stage 1 (autoencoder L_A):
    "generate"  -> push L_A below tau_G  (image looks model-generated)
    "natural"   -> push/keep L_A above tau_G (image looks non-generated)
    "off"       -> no Stage-1 term
  Stage 2 (RAR ICAS = nll_uncond - nll_cond), only meaningful for non-generated:
    "raise"     -> push ICAS above a member target (looks like a member)
    "lower"     -> push ICAS below tau_MN (looks like a non-member)
    "off"       -> no Stage-2 term

Direction -> (source class, Stage 1, Stage 2):
    M_N: (M, natural , lower)
    M_G: (M, generate, off  )
    N_M: (N, natural , raise)
    N_G: (N, generate, off  )
    G_M: (G, natural , raise)   # generated ICAS is already high; keep it up
    G_N: (G, natural , lower)

The differentiable ICAS path (soft-token straight-through + RAR forward
replication) is imported from attack_membership.py and is validated by its
--selftest. Always run --selftest before trusting a real run, then score the
resulting block with evaluate_submission.py.

Run on a GPU node:
  export ONED_TOKENIZER_ROOT=/p/scratch/training2625/dougnon1/Loki/1d-tokenizer
  export PYTHONPATH=$ONED_TOKENIZER_ROOT:$PWD/task_2_mgi/attempt1
  python task_2_mgi/attempt1/attack_combined.py --directions M_N,M_G,N_M,N_G,G_M,G_N --selftest --limit 8
  python task_2_mgi/attempt1/attack_combined.py --directions M_G,N_G --steps 200
  python task_2_mgi/attempt1/attack_combined.py --directions M_N,N_M,G_M,G_N --steps 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import CLASS_SIZE, PathsConfig, setup_oned_tokenizer_path  # noqa: E402
from proxy_dcb import compute_LA, load_calibration, load_vqgan, uint8_to_tensor  # noqa: E402
from submission_io import jpeg_roundtrip_uint8, load_reference_images  # noqa: E402

CLASS_RANGES = {"M": (0, CLASS_SIZE), "N": (CLASS_SIZE, 2 * CLASS_SIZE),
                "G": (2 * CLASS_SIZE, 3 * CLASS_SIZE)}

# name -> (source_class, stage1, stage2)
DIRECTION_SPEC = {
    "M_N": ("M", "natural", "lower"),
    "M_G": ("M", "generate", "off"),
    "N_M": ("N", "natural", "raise"),
    "N_G": ("N", "generate", "off"),
    "G_M": ("G", "natural", "raise"),
    "G_N": ("G", "natural", "lower"),
}


def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.clamp(0.0, 1.0).permute(0, 2, 3, 1).contiguous()
    return (x.cpu().numpy() * 255.0).round().astype(np.uint8)


def _optimal_member_threshold(icas_m: np.ndarray, icas_n: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([icas_m, icas_n]))
    best_tau, best_acc = float(np.median(candidates)), -1.0
    for tau in candidates:
        acc = 0.5 * ((icas_m >= tau).mean() + (icas_n < tau).mean())
        if acc > best_acc:
            best_acc, best_tau = acc, float(tau)
    return best_tau


def _stage1_loss(la: torch.Tensor, mode: str, tau_g: float, kappa: float) -> torch.Tensor:
    if mode == "generate":
        return F.relu(la - (tau_g - kappa))
    if mode == "natural":
        return F.relu((tau_g + kappa) - la)
    return torch.zeros((), device=la.device)


def _stage2_loss(icas: torch.Tensor, mode: str, target: float) -> torch.Tensor:
    if mode == "raise":
        return F.relu(target - icas)
    if mode == "lower":
        return F.relu(icas - target)
    return torch.zeros((), device=icas.device)


def _cell_ok(la: float, icas: float | None, stage1: str, stage2: str,
             tau_g: float, tau_mn: float, member_target: float) -> bool:
    if stage1 == "generate" and not (la < tau_g):
        return False
    if stage1 == "natural" and not (la > tau_g):
        return False
    if stage2 == "off" or icas is None:
        return True
    if stage2 == "raise":
        return icas >= member_target
    if stage2 == "lower":
        return icas < tau_mn
    return True


def attack_direction(name, refs, tokenizer, generator, classifier, vqgan,
                     device, cfg, thresholds):
    """Attack one 300-image block; return (adv_uint8, logs)."""
    from proxy_icas import compute_membership_stats, predict_imagenet_class
    from attack_membership import icas_from_emb, soft_tokens

    src_cls, stage1, stage2 = DIRECTION_SPEC[name]
    tau_g, alpha, tau_mn, member_target = thresholds
    n = len(refs)
    out = refs.copy()
    logs: list[dict] = []

    for start in range(0, n, cfg.batch_size):
        chunk = refs[start:start + cfg.batch_size]
        x_orig = uint8_to_tensor(chunk, device)
        b = x_orig.shape[0]
        need_icas = stage2 != "off"
        labels = None
        if need_icas:
            with torch.no_grad():
                labels = predict_imagenet_class(x_orig, classifier)

        w = torch.arctanh((x_orig * 2.0 - 1.0).clamp(-1 + 1e-6, 1 - 1e-6))
        w = w.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([w], lr=cfg.lr)

        best_mse = np.full(b, np.inf, dtype=np.float64)
        best_img = tensor_to_uint8(x_orig)
        best_ok = np.zeros(b, dtype=bool)

        for step in range(cfg.steps):
            opt.zero_grad(set_to_none=True)
            x_adv = 0.5 * (torch.tanh(w) + 1.0)

            la = compute_LA(vqgan, x_adv, alpha=alpha, eps=cfg.eps, reduction="none")
            loss_s1 = _stage1_loss(la, stage1, tau_g, cfg.kappa).mean()
            loss_s2 = torch.zeros((), device=device)
            if need_icas:
                emb, codes_flat = soft_tokens(tokenizer, generator, x_adv, cfg.temp)
                icas = icas_from_emb(generator, emb, codes_flat, labels)
                loss_s2 = _stage2_loss(icas, stage2, member_target if stage2 == "raise"
                                       else (tau_mn - cfg.margin)).mean()

            mse = ((x_adv - x_orig) ** 2).mean()
            loss = cfg.w_s1 * loss_s1 + cfg.w_s2 * loss_s2 + cfg.c_mse * mse
            loss.backward()
            opt.step()

            with torch.no_grad():
                x_test = 0.5 * (torch.tanh(w) + 1.0)
                adv_uint8 = tensor_to_uint8(x_test)
                if cfg.jpeg_quality is not None:
                    adv_uint8 = np.stack([
                        jpeg_roundtrip_uint8(a, cfg.jpeg_quality) for a in adv_uint8])
                x_dep = uint8_to_tensor(adv_uint8, device)
                la_dep = compute_LA(vqgan, x_dep, alpha=alpha, eps=cfg.eps,
                                    reduction="none").cpu().numpy()
                icas_dep = None
                if need_icas:
                    _, _, ic = compute_membership_stats(generator, tokenizer, x_dep, labels)
                    icas_dep = ic.cpu().numpy()
                diff = adv_uint8.astype(np.float32) - chunk.astype(np.float32)
                mse_dep = np.mean((diff / 255.0) ** 2, axis=(1, 2, 3))

                for i in range(b):
                    ok = _cell_ok(float(la_dep[i]),
                                  float(icas_dep[i]) if icas_dep is not None else None,
                                  stage1, stage2, tau_g, tau_mn, member_target)
                    # prefer any success (min MSE among successes); else min MSE overall
                    better = (ok and (not best_ok[i] or mse_dep[i] < best_mse[i])) or \
                             (not best_ok[i] and not ok and mse_dep[i] < best_mse[i])
                    if better:
                        best_mse[i] = mse_dep[i]
                        best_img[i] = adv_uint8[i]
                        best_ok[i] = ok

            if step % cfg.log_every == 0 or step == cfg.steps - 1:
                print(f"    [{name}] batch {start:3d} step {step:4d}  "
                      f"ok={best_ok.mean():.2f}  mse={best_mse[best_ok].mean() if best_ok.any() else float('nan'):.5f}",
                      flush=True)

        out[start:start + b] = best_img
        for i in range(b):
            logs.append({"index": int(start + i), "direction": name,
                         "success": bool(best_ok[i]), "mse": float(best_mse[i])})

    return out, logs


class Cfg:
    pass


def main() -> int:
    p = argparse.ArgumentParser(description="Unified per-cell MGI attack")
    p.add_argument("--directions", type=str, default=",".join(DIRECTION_SPEC))
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--kappa", type=float, default=0.0,
                   help="Stage-1 margin past tau_G (push franc = larger)")
    p.add_argument("--margin", type=float, default=0.05,
                   help="Stage-2 margin below tau_MN for 'lower'")
    p.add_argument("--member-margin", type=float, default=0.30,
                   help="ICAS target above member max for 'raise'")
    p.add_argument("--w-s1", type=float, default=10.0)
    p.add_argument("--w-s2", type=float, default=10.0)
    p.add_argument("--c-mse", type=float, default=1.0)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--jpeg-quality", type=int, default=80,
                   help="Validate deployable success after JPEG q (None = uint8 only)")
    p.add_argument("--no-jpeg", action="store_true")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--eps", type=float, default=1e-6)
    args = p.parse_args()

    paths = PathsConfig()
    setup_oned_tokenizer_path(paths.oned_tokenizer_root)
    if not paths.tokenizer_ckpt.is_file():
        print(f"ERROR: tokenizer not found: {paths.tokenizer_ckpt}", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = args.out_dir or (paths.output_dir / "blocks")
    out_dir.mkdir(parents=True, exist_ok=True)
    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
    for d in directions:
        if d not in DIRECTION_SPEC:
            print(f"ERROR: unknown direction {d!r}", file=sys.stderr)
            return 1

    originals = load_reference_images(paths.data_dir)
    tokenizer = load_vqgan(paths.tokenizer_ckpt, device=device)

    need_icas = any(DIRECTION_SPEC[d][2] != "off" for d in directions)
    generator = classifier = None
    tau_mn, member_target = 0.0, 0.0
    if need_icas or args.selftest:
        from proxy_icas import (compute_membership_stats, load_class_predictor,
                                load_rar_generator, predict_imagenet_class)
        generator, _ = load_rar_generator(tokenizer_ckpt=paths.tokenizer_ckpt, device=device)
        classifier = load_class_predictor(device=device)

    if args.selftest:
        from attack_membership import selftest
        refs = originals[CLASS_RANGES["N"][0]:CLASS_RANGES["N"][0] + min(8, args.limit or 8)]
        ok = selftest(tokenizer, generator, classifier, refs, device, args.temp)
        if not ok:
            print("SELFTEST FAILED — do not trust the attack.", file=sys.stderr)
            return 2

    # calibration (tau_G, alpha)
    if paths.calibration_path and Path(paths.calibration_path).is_file():
        cal = load_calibration(paths.calibration_path)
        tau_g, alpha = cal["tau_G"], cal["alpha"]
    else:
        from proxy_dcb import calibrate_thresholds
        cal = calibrate_thresholds(
            tokenizer,
            uint8_to_tensor(originals[0:CLASS_SIZE], device),
            uint8_to_tensor(originals[CLASS_SIZE:2 * CLASS_SIZE], device),
            uint8_to_tensor(originals[2 * CLASS_SIZE:3 * CLASS_SIZE], device),
            eps=args.eps, batch_size=8)
        tau_g, alpha = cal["tau_G"], cal["alpha"]
    print(f"Stage-1: tau_G={tau_g:.6f}  alpha={alpha:.4f}")

    if need_icas:
        from proxy_icas import compute_membership_stats, predict_imagenet_class

        def _ref_icas(sl):
            vals = []
            for s in range(0, len(sl), 8):
                x = uint8_to_tensor(sl[s:s + 8], device)
                with torch.no_grad():
                    lab = predict_imagenet_class(x, classifier)
                    _, _, ic = compute_membership_stats(generator, tokenizer, x, lab)
                vals.append(ic.cpu().numpy())
            return np.concatenate(vals)

        icas_m = _ref_icas(originals[0:CLASS_SIZE])
        icas_n = _ref_icas(originals[CLASS_SIZE:2 * CLASS_SIZE])
        tau_mn = _optimal_member_threshold(icas_m, icas_n)
        member_target = float(icas_m.max()) + args.member_margin
        print(f"Stage-2: tau_MN={tau_mn:+.4f}  member_target={member_target:+.4f} "
              f"(M max={icas_m.max():+.4f})")

    cfg = Cfg()
    cfg.steps, cfg.lr, cfg.kappa, cfg.margin = args.steps, args.lr, args.kappa, args.margin
    cfg.w_s1, cfg.w_s2, cfg.c_mse = args.w_s1, args.w_s2, args.c_mse
    cfg.temp, cfg.batch_size, cfg.eps, cfg.log_every = args.temp, args.batch_size, args.eps, args.log_every
    cfg.jpeg_quality = None if args.no_jpeg else args.jpeg_quality

    thresholds = (tau_g, alpha, tau_mn, member_target)
    for name in directions:
        src_cls = DIRECTION_SPEC[name][0]
        a, bnd = CLASS_RANGES[src_cls]
        refs = originals[a:bnd]
        if args.limit is not None:
            refs = refs[:args.limit]
        print(f"\n=== {name}  source={src_cls}  "
              f"stage1={DIRECTION_SPEC[name][1]}  stage2={DIRECTION_SPEC[name][2]}  "
              f"({len(refs)} images) ===")
        adv, logs = attack_direction(name, refs, tokenizer, generator, classifier,
                                     tokenizer, device, cfg, thresholds)
        np.save(out_dir / f"{name}_combined.npy", adv)
        ok_rate = float(np.mean([l["success"] for l in logs]))
        mse = float(np.mean([l["mse"] for l in logs]))
        (out_dir / f"{name}_combined.json").write_text(json.dumps(
            {"direction": name, "success_rate": ok_rate, "mean_mse": mse,
             "logs": logs}, indent=2))
        print(f"  saved {out_dir / f'{name}_combined.npy'}  "
              f"proxy_ok={ok_rate:.1%}  mean_mse={mse:.5f}")

    print("\nAssemble + score:")
    print("  python task_2_mgi/attempt1/build_submission_v2.py \\")
    for name in directions:
        print(f"      --dir {name}=block:{out_dir / f'{name}_combined.npy'} \\")
    print("  python task_2_mgi/attempt1/evaluate_submission.py output/submission.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
