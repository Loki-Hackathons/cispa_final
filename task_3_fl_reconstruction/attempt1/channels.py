"""Exact CNN channel inversion via detected 'transmit' conv filters.

Boenisch et al., Appendix B: to extend the analytic attack to CNNs, the malicious
server initializes the convolution so it *transmits* the input unaltered to the
first fully-connected layer.  A transmit filter is (near-)zero everywhere except
one tap (the centre for size-preserving conv), which copies one input channel
with some scale ``s``.  The remaining ``noise`` filters are set so ReLU zeroes
them out.  So the fc1 *input* feature map has, among its ``C_out`` channels,
exactly ``C_in`` (=3) channels that are a scaled copy of the RGB input; the rest
are junk.

The previous pipeline collapsed all ``C_out`` channels to RGB by *averaging
chunks*, which mixes the real image channels with the noise channels and washes
out the signal.  Here we instead:

  1. read the KNOWN conv weights/bias and detect which output channel transmits
     which input channel (delta-like filter -> (in_channel, scale)),
  2. invert the activation exactly (ReLU identity / atanh / logit) and the scale,
     recovering the RGB input at feature resolution,
  3. fall back to the old channel-averaging only when no clear transmit
     structure is present (defended / non-trap models).

Everything here uses only the provided conv weights + the recovered fc1-input
feature maps.  No optimisation, no labels, no leaderboard.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

import config
import utils


@dataclass
class ConvTransmit:
    """Detected transmit structure of one conv layer."""

    in_channels: int
    out_channels: int
    # For each input channel c (0..in_channels-1): the out-channel that best
    # transmits it, its scale, and a [0,1] 'delta-ness' score. -1 out-channel
    # means "no transmit channel found for c".
    out_for_in: list[int]
    scale_for_in: list[float]
    delta_for_in: list[float]
    strength: float              # mean delta score of the chosen channels

    def is_transmit(self, min_delta: float = 0.55) -> bool:
        """True if enough input channels have a clearly delta-like transmitter."""
        good = [d for d in self.delta_for_in if d >= min_delta]
        return len(good) >= max(1, self.in_channels - 1)


def analyze_conv(conv_w: torch.Tensor, conv_b: torch.Tensor | None = None) -> ConvTransmit:
    """Detect transmit filters from conv weights (C_out, C_in, kh, kw).

    A transmit filter concentrates its energy in a single (in_channel, y, x) tap.
    We score each output channel by ``max|w| / sum|w|`` (1.0 == perfect delta)
    and, for each input channel, keep the output channel with the strongest,
    most delta-like tap on it.
    """
    conv_w = conv_w.detach().float()
    cout, cin, kh, kw = conv_w.shape
    flat = conv_w.reshape(cout, cin, kh * kw)
    absf = flat.abs()

    # Per out-channel: which input channel + tap holds the peak, and delta score.
    energy = absf.reshape(cout, -1)                      # (cout, cin*k*k)
    peak_val, peak_idx = energy.max(dim=1)               # (cout,)
    total = energy.sum(dim=1).clamp_min(config.EPS)
    delta = (peak_val / total).clamp(0, 1)               # 1.0 == pure delta
    peak_in = (peak_idx // (kh * kw)).long()             # which input channel
    # Signed scale at the peak tap (may be negative).
    scale = torch.gather(flat.reshape(cout, -1), 1,
                         peak_idx.unsqueeze(1)).squeeze(1)

    out_for_in, scale_for_in, delta_for_in = [], [], []
    used: set[int] = set()
    for c in range(cin):
        # Out-channels whose peak lands on input channel c, ranked by delta score.
        mask = peak_in == c
        if not bool(mask.any()):
            out_for_in.append(-1)
            scale_for_in.append(1.0)
            delta_for_in.append(0.0)
            continue
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        # Prefer a not-yet-used channel with the highest delta score.
        order = idx[torch.argsort(delta[idx], descending=True)]
        pick = None
        for o in order.tolist():
            if o not in used:
                pick = o
                break
        if pick is None:
            pick = int(order[0])
        used.add(pick)
        out_for_in.append(int(pick))
        scale_for_in.append(float(scale[pick]))
        delta_for_in.append(float(delta[pick]))

    chosen = [d for d in delta_for_in if d > 0]
    strength = float(sum(chosen) / len(chosen)) if chosen else 0.0
    return ConvTransmit(cin, cout, out_for_in, scale_for_in, delta_for_in, strength)


def _inv_activation(y: torch.Tensor, name: str) -> torch.Tensor:
    """Invert a pointwise activation to recover its pre-activation.

    ReLU is treated as identity on the transmit channels: trap filters are set so
    the pre-activation stays positive, hence ReLU(pre) == pre wherever the image
    has signal; zeros just map back to zero pre-activation.
    """
    if name == "relu":
        return y
    if name == "tanh":
        return torch.atanh(y.clamp(-1 + 1e-4, 1 - 1e-4))
    if name == "sigmoid":
        y = y.clamp(1e-4, 1 - 1e-4)
        return torch.log(y) - torch.log1p(-y)
    if name == "gelu":
        # GELU is monotone for pre >~ -0.75; treat as identity (good enough for
        # the positive transmit regime).
        return y
    return y


def transmit_features_to_rgb(
    feats: torch.Tensor,
    conv_w: torch.Tensor,
    conv_b: torch.Tensor | None,
    activation: str,
    min_delta: float = 0.55,
) -> tuple[torch.Tensor, bool]:
    """Recover RGB from recovered fc1-input feature maps using transmit filters.

    ``feats`` : (N, C_out, H, W) recovered feature maps (the fc1 input for each
    candidate image).  Returns ``(rgb, used_transmit)`` where ``rgb`` is
    (N, 3, 64, 64) in [0,1].  Falls back to channel-averaging + per-image
    min/max when no transmit structure is detected, so it never regresses.
    """
    tm = analyze_conv(conv_w, conv_b)
    n, cout, h, w = feats.shape
    conv_b = conv_b.detach().float() if conv_b is not None else torch.zeros(cout)

    if not tm.is_transmit(min_delta) or tm.in_channels != 3:
        # Defended / non-trap model: keep the old robust behaviour.
        rows = feats.reshape(n, cout * h * w)
        return utils.features_to_image(rows, cout, h, w), False

    chans = []
    for c in range(3):
        o = tm.out_for_in[c]
        s = tm.scale_for_in[c]
        if o < 0 or abs(s) < 1e-6:
            chans.append(torch.zeros(n, h, w))
            continue
        y = feats[:, o]                                  # (N,H,W) == act(s*x_c + b_o)
        pre = _inv_activation(y, activation)             # s*x_c + b_o
        x_c = (pre - conv_b[o]) / s                       # recover input channel
        chans.append(x_c)
    rgb = torch.stack(chans, dim=1)                       # (N,3,H,W)

    if (h, w) != (64, 64):
        rgb = F.interpolate(rgb, size=(64, 64), mode="bilinear", align_corners=False)

    # Transmit inversion yields true-scale pixels; clamp instead of min/max
    # stretch so the luminance matches the ground truth (better SSIM).
    return rgb.clamp(0, 1).float(), True


def report(model_ids: list[int]) -> None:
    """Print detected transmit structure per CNN model (diagnostic)."""
    print(f"{'model':>5} | {'act':7} | {'shape':>10} | {'strength':>8} | "
          f"{'transmit?':>9} | per-in (out:scale:delta)")
    print("-" * 92)
    for i in model_ids:
        grad = utils.load_gradient(i)
        if grad["family"] != "cnn":
            continue
        state = utils.load_state(i)
        cw = state["conv.weight"]
        cb = state.get("conv.bias")
        tm = analyze_conv(cw, cb)
        fs = tuple(grad["feature_shape"])
        per = " ".join(
            f"{tm.out_for_in[c]}:{tm.scale_for_in[c]:+.2f}:{tm.delta_for_in[c]:.2f}"
            for c in range(min(3, tm.in_channels))
        )
        print(f"{i:>5} | {grad['activation']:7} | {str(fs):>10} | "
              f"{tm.strength:8.3f} | {str(tm.is_transmit()):>9} | {per}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=int, nargs="*",
                    default=[2, 3, 6, 7, 10, 12])
    args = ap.parse_args()
    report(args.models)
