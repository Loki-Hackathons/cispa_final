#!/usr/bin/env python3
"""Approach B for the '*->M' directions: gradient attack that RAISES an image's
RAR membership signal (icas = conditional-likelihood discrepancy) past the
member distribution, while keeping the image close to the reference (low MSE)
and optionally natural (autoencoder L_A in the member band, not 'generated').

Why this and not Approach A
---------------------------
The free diagnostic (score_membership.py --diagnose) showed icas cleanly
separates members (M) from non-members (N): TPR .957 / FPR .043. But real
members top out at icas ~+0.205, and submitting even the single strongest
member does NOT flip ->M (the detector's member threshold sits ABOVE every real
member). So no natural image works; we must MANUFACTURE an image whose icas is
pushed well beyond +0.205, without making it look generated (G has the highest
icas of all, so a pure likelihood push risks landing in G).

How it stays correct
---------------------
The tokenizer (MaskGIT VQ-GAN, titok.PretrainedTokenizer) maps pixels -> a
continuous encoder map -> nearest-codebook indices (non-differentiable argmin).
RAR consumes those indices. To get gradients pixel<-icas we:
  1. encoder(x) -> hidden (B,256,16,16)
  2. distances to the codebook (tokenizer.quantize.embedding.weight, (1024,256))
  3. soft assignment p = softmax(-d/temp); hard idx = argmin(d)
  4. RAR image-token embedding = p @ RAR.embeddings.weight[:1024]  (differentiable)
     with a straight-through estimator so the forward value uses the HARD tokens
     (exact icas the detector would see) but gradients flow through the soft map.
  5. replicate RAR.forward_fn (raster order) with those injected embeddings for
     both the class condition and the null condition -> icas.

--selftest verifies (a) our token ordering matches tokenizer.encode and (b) our
straight-through icas matches proxy_icas.compute_membership_stats on real
images, BEFORE spending GPU time on all 300. If those don't match, the codebook
distance / token order is wrong and the attack would be optimizing garbage.

Run on a GPU node:
  export ONED_TOKENIZER_ROOT=/p/scratch/training2625/dougnon1/Loki/1d-tokenizer
  export PYTHONPATH=$ONED_TOKENIZER_ROOT:$PWD/task_2_mgi/attempt1
  # sanity check first (fast):
  python task_2_mgi/attempt1/attack_membership.py --direction N_M --selftest --limit 8
  # then the real run:
  python task_2_mgi/attempt1/attack_membership.py --direction N_M --steps 250 --batch-size 4
  python task_2_mgi/attempt1/attack_membership.py --direction G_M --steps 250 --batch-size 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import CLASS_SIZE, PathsConfig, setup_oned_tokenizer_path  # noqa: E402
from proxy_dcb import compute_LA, load_vqgan, uint8_to_tensor  # noqa: E402
from proxy_icas import (  # noqa: E402
    compute_membership_stats,
    load_class_predictor,
    load_rar_generator,
    predict_imagenet_class,
)
from submission_io import load_reference_images  # noqa: E402

CLASS_RANGES = {"M": (0, CLASS_SIZE), "N": (CLASS_SIZE, 2 * CLASS_SIZE),
                "G": (2 * CLASS_SIZE, 3 * CLASS_SIZE)}
# Direction -> source class whose 300 images we perturb (goal: classified M).
DIR_SOURCE = {"N_M": "N", "G_M": "G"}


def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.clamp(0.0, 1.0).permute(0, 2, 3, 1).contiguous()
    return (x.cpu().numpy() * 255.0).round().astype(np.uint8)


# --------------------------------------------------------------------------- #
# Differentiable soft-token path + RAR forward replication                    #
# --------------------------------------------------------------------------- #
def soft_tokens(tokenizer, generator, x: torch.Tensor, temperature: float):
    """pixels -> (image-token embeddings for RAR, hard token indices).

    Returns:
      emb        : (B, T, E) RAR image-token embeddings (straight-through:
                   forward = hard-token embedding, backward = soft).
      codes_flat : (B, T) long, the HARD token indices (== tokenizer.encode).
    """
    hidden = tokenizer.encoder(x)                      # (B, C, H, W)
    b, c, h, w = hidden.shape
    hf = hidden.permute(0, 2, 3, 1).reshape(-1, c)     # (B*H*W, C), row-major (h,w)

    codebook = tokenizer.quantize.embedding.weight     # (K, C)
    d = (hf * hf).sum(1, keepdim=True) \
        + (codebook * codebook).sum(1) \
        - 2.0 * hf @ codebook.t()                      # (B*H*W, K)

    idx = d.argmin(dim=1)                               # (B*H*W,)
    p = F.softmax(-d / temperature, dim=1)              # (B*H*W, K)

    rar_emb = generator.embeddings.weight[:codebook.shape[0]]  # (K, E)
    soft = p @ rar_emb                                  # (B*H*W, E)
    hard = rar_emb[idx]                                 # (B*H*W, E)
    ste = hard.detach() + (soft - soft.detach())        # value=hard, grad=soft

    e = rar_emb.shape[1]
    return ste.view(b, -1, e), idx.view(b, -1)


def rar_logits_from_emb(generator, img_emb: torch.Tensor,
                        condition_ids: torch.Tensor) -> torch.Tensor:
    """Replicate RAR.forward_fn (raster order, non-sampling, no kv-cache) but
    inject precomputed image-token embeddings. Returns logits for the T image
    tokens: (B, T, codebook_size)."""
    gen = generator
    b = img_emb.shape[0]
    seq_len = gen.image_seq_len
    prefix = 2

    cond_emb = gen.embeddings(condition_ids.view(b, 1))     # (B,1,E)
    condition_token = cond_emb[:, 0]                        # (B,E)
    embeddings = torch.cat([cond_emb, img_emb], dim=1)      # (B, 1+T, E)

    cls_tokens = gen.cls_token.expand(b, -1, -1)            # (B,1,E)
    x = torch.cat((cls_tokens, embeddings), dim=1)          # (B, 2+T, E)

    pos_embed = gen.pos_embed.repeat(b, 1, 1)
    pos_prefix = pos_embed[:, :prefix]
    pos_postfix = pos_embed[:, prefix:prefix + seq_len]     # raster: identity
    x = x + torch.cat([pos_prefix, pos_postfix], dim=1)[:, :x.shape[1]]

    tap = gen.target_aware_pos_embed.repeat(b, 1, 1)
    tap_postfix = tap[:, prefix:prefix + seq_len]
    tap_full = torch.cat(
        [torch.zeros_like(x[:, :prefix - 1]), tap_postfix, torch.zeros_like(x[:, -1:])],
        dim=1)
    x = x + tap_full[:, :x.shape[1]]

    attn_mask = gen.attn_mask[:x.shape[1], :x.shape[1]]
    condition_token = condition_token.unsqueeze(1) + gen.timesteps_embeddings[:, :x.shape[1]]

    for blk in gen.blocks:
        x = blk(x, attn_mask=attn_mask, c=condition_token)

    x = x[:, prefix - 1:]
    condition_token = condition_token[:, prefix - 1:]
    x = gen.adaln_before_head(x, condition_token)
    x = gen.lm_head(x)                                      # (B, 1+T, K)
    return x[:, :seq_len, :]


def icas_from_emb(generator, img_emb: torch.Tensor, codes_flat: torch.Tensor,
                  class_labels: torch.Tensor) -> torch.Tensor:
    """Differentiable icas (nll_uncond - nll_cond) from injected embeddings."""
    b = img_emb.shape[0]
    cond_ids = generator.preprocess_condition(
        class_labels.view(b, 1).clone(), cond_drop_prob=0.0)
    none_ids = generator.get_none_condition(cond_ids)

    emb2 = torch.cat([img_emb, img_emb], dim=0)
    cond2 = torch.cat([cond_ids, none_ids], dim=0)
    logits2 = rar_logits_from_emb(generator, emb2, cond2)
    cond_logits, uncond_logits = logits2[:b], logits2[b:]

    tgt = codes_flat.unsqueeze(-1)
    nll_c = -F.log_softmax(cond_logits, -1).gather(-1, tgt).squeeze(-1).mean(1)
    nll_u = -F.log_softmax(uncond_logits, -1).gather(-1, tgt).squeeze(-1).mean(1)
    return nll_u - nll_c


# --------------------------------------------------------------------------- #
# Self-test: our replication must match the ground-truth implementations       #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def selftest(tokenizer, generator, classifier, images_uint8, device, temperature):
    x = uint8_to_tensor(images_uint8, device)
    labels = predict_imagenet_class(x, classifier)

    emb, codes_flat = soft_tokens(tokenizer, generator, x, temperature)
    ref_codes = tokenizer.encode(x).view(x.shape[0], -1).long().to(device)
    token_match = (codes_flat == ref_codes).float().mean().item()

    ours = icas_from_emb(generator, emb, codes_flat, labels)
    _, _, ref = compute_membership_stats(generator, tokenizer, x, labels)

    print("\n================ SELF-TEST ================")
    print(f"  token-order match vs tokenizer.encode : {token_match * 100:.2f}%")
    print(f"  our icas   : {ours.detach().cpu().numpy().round(4)}")
    print(f"  ref  icas  : {ref.detach().cpu().numpy().round(4)}")
    max_abs = (ours - ref).abs().max().item()
    print(f"  max |our - ref| icas : {max_abs:.5f}")
    ok = token_match > 0.999 and max_abs < 1e-3
    print(f"  RESULT: {'PASS - replication correct' if ok else 'FAIL - do NOT trust attack'}")
    return ok


# --------------------------------------------------------------------------- #
# Attack                                                                       #
# --------------------------------------------------------------------------- #
def attack_batch(tokenizer, generator, classifier, vqgan, x_orig_uint8,
                 device, cfg) -> tuple[np.ndarray, np.ndarray]:
    x_orig = uint8_to_tensor(x_orig_uint8, device)
    with torch.no_grad():
        labels = predict_imagenet_class(x_orig, classifier)

    w = torch.arctanh((x_orig * 2.0 - 1.0).clamp(-1 + 1e-6, 1 - 1e-6))
    w = w.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=cfg.lr)

    b = x_orig.shape[0]
    best_icas = np.full(b, -1e9, dtype=np.float32)
    best_img = x_orig_uint8.copy()

    for step in range(cfg.steps):
        opt.zero_grad(set_to_none=True)
        x_adv = 0.5 * (torch.tanh(w) + 1.0)

        emb, codes_flat = soft_tokens(tokenizer, generator, x_adv, cfg.temp)
        icas = icas_from_emb(generator, emb, codes_flat, labels)

        # Hinge: push icas up to target, then stop (spend budget on MSE instead).
        icas_loss = F.relu(cfg.target_icas - icas).mean()
        mse = ((x_adv - x_orig) ** 2).mean()
        loss = icas_loss + cfg.c_mse * mse

        if cfg.la_weight > 0.0:
            la = compute_LA(vqgan, x_adv, alpha=cfg.la_alpha, reduction="mean")
            # keep L_A high (natural); penalise dropping toward 'generated'
            loss = loss + cfg.la_weight * F.relu(cfg.la_tau - la)

        loss.backward()
        opt.step()

        with torch.no_grad():
            x_test = 0.5 * (torch.tanh(w) + 1.0)
            adv_uint8 = tensor_to_uint8(x_test)
            x_deploy = uint8_to_tensor(adv_uint8, device)
            _, _, icas_dep = compute_membership_stats(
                generator, tokenizer, x_deploy, labels)
            icas_np = icas_dep.detach().cpu().numpy()
            for i in range(b):
                if icas_np[i] > best_icas[i]:
                    best_icas[i] = icas_np[i]
                    best_img[i] = adv_uint8[i]

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            print(f"    step {step:4d}  icas(soft)={icas.mean().item():+.4f}  "
                  f"icas(deploy)={icas_np.mean():+.4f}  mse={mse.item():.5f}",
                  flush=True)

    return best_img, best_icas


class Cfg:
    pass


def main() -> int:
    p = argparse.ArgumentParser(description="Gradient attack raising RAR membership (icas) for ->M")
    p.add_argument("--direction", choices=["N_M", "G_M"], required=True)
    p.add_argument("--target-icas", type=float, default=0.5,
                   help="Push icas up to this (member max ~0.205, G max ~0.98)")
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--c-mse", type=float, default=0.1, help="MSE penalty weight")
    p.add_argument("--temp", type=float, default=1.0, help="Soft-quant softmax temperature")
    p.add_argument("--la-weight", type=float, default=0.0,
                   help="Weight to keep autoencoder L_A natural (non-generated). 0=off")
    p.add_argument("--la-tau", type=float, default=0.0, help="Natural L_A floor to maintain")
    p.add_argument("--la-alpha", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="Only first N images (debug)")
    p.add_argument("--selftest", action="store_true",
                   help="Verify replication vs ground truth, then exit")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    paths = PathsConfig()
    setup_oned_tokenizer_path(paths.oned_tokenizer_root)
    if not paths.tokenizer_ckpt.is_file():
        print(f"ERROR: tokenizer checkpoint not found: {paths.tokenizer_ckpt}", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = args.out_dir or (paths.output_dir / "blocks")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  direction={args.direction}  target_icas={args.target_icas}")

    originals = load_reference_images(paths.data_dir)
    src_cls = DIR_SOURCE[args.direction]
    a, bnd = CLASS_RANGES[src_cls]
    refs = originals[a:bnd]
    if args.limit is not None:
        refs = refs[:args.limit]

    tokenizer = load_vqgan(paths.tokenizer_ckpt, device=device)
    generator, _ = load_rar_generator(tokenizer_ckpt=paths.tokenizer_ckpt, device=device)
    classifier = load_class_predictor(device=device)

    if args.selftest:
        ok = selftest(tokenizer, generator, classifier,
                      refs[:min(8, len(refs))], device, args.temp)
        return 0 if ok else 2

    cfg = Cfg()
    cfg.target_icas = args.target_icas
    cfg.steps = args.steps
    cfg.lr = args.lr
    cfg.c_mse = args.c_mse
    cfg.temp = args.temp
    cfg.la_weight = args.la_weight
    cfg.la_tau = args.la_tau
    cfg.la_alpha = args.la_alpha
    cfg.log_every = args.log_every

    out = np.empty_like(refs)
    all_icas = np.empty(len(refs), dtype=np.float32)
    for start in range(0, len(refs), args.batch_size):
        chunk = refs[start:start + args.batch_size]
        print(f"\n[{args.direction}] batch {start}..{start + len(chunk)} / {len(refs)}")
        adv, icas = attack_batch(tokenizer, generator, classifier, tokenizer,
                                 chunk, device, cfg)
        out[start:start + len(chunk)] = adv
        all_icas[start:start + len(chunk)] = icas

    diff = out.astype(np.float32) - refs.astype(np.float32)
    mse = float(np.mean((diff / 255.0) ** 2))
    name = f"{args.direction}_attack.npy"
    np.save(out_dir / name, out)
    print(f"\nSaved {out_dir / name}")
    print(f"  final icas: mean={all_icas.mean():+.4f} min={all_icas.min():+.4f} "
          f"max={all_icas.max():+.4f}  (member max was ~0.205)")
    print(f"  fraction above +0.30: {(all_icas > 0.30).mean():.2f}   "
          f"above +0.50: {(all_icas > 0.50).mean():.2f}")
    print(f"  mse_norm={mse:.5f}  (1-mse)={1.0 - mse:.5f}")
    print("\nPlug into a submission:")
    print(f"  --dir {args.direction}=block:{out_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
