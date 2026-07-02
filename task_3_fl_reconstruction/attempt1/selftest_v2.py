"""End-to-end self-test for the v2 pipeline on SYNTHETIC trap-weight models.

We fabricate MLP and CNN models with trap-like weights, run a real forward +
cross-entropy backward over a batch of 128 [0,1] images to produce gradients in
the exact task format, then check that reconstruct_v2 recovers the images.

This validates the recovery math (analytic identity, row clustering, conv
transmit inversion) without needing the cluster data or a GPU.  Run:

    python selftest_v2.py
"""
from __future__ import annotations

import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


def _isolating_linear(inp: torch.Tensor, out_f: int) -> nn.Linear:
    """Build a linear layer whose ReLU neurons each activate on exactly ONE
    batch item — the trap-weight ideal (Boenisch eq. 9 satisfied for a single x).

    We do this constructively: random weights, then per-neuron bias set just
    below its single largest pre-activation over the batch, so only the argmax
    image yields preact>0.  This is exactly the single-image regime the analytic
    identity reconstructs perfectly, so it validates the recovery math.
    """
    in_f = inp.shape[1]
    lin = nn.Linear(in_f, out_f)
    with torch.no_grad():
        w = torch.randn(out_f, in_f) / (in_f ** 0.5)
        lin.weight.copy_(w)
        pre = inp @ w.t()                                # (B, out_f)
        top2 = pre.topk(2, dim=0).values                 # (2, out_f)
        # bias so neuron i activates only for its top image (margin below 2nd).
        b = -0.5 * (top2[0] + top2[1])
        lin.bias.copy_(b)
    return lin


def _make_batch(n=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Low-frequency structured images (more image-like than white noise).
    base = torch.rand(n, 3, 8, 8, generator=g)
    return F.interpolate(base, size=(64, 64), mode="bilinear",
                         align_corners=False).clamp(0, 1)


def _nearest_ssim(recon: torch.Tensor, truth: torch.Tensor) -> float:
    import utils
    m = utils.ssim_matrix(recon, truth)          # (Nr, Nt)
    return float(m.max(dim=1).values.mean())     # avg best-match (upper-ish bound)


def make_mlp(root: str):
    torch.manual_seed(1)
    x = _make_batch(seed=7)
    l0 = _isolating_linear(x.flatten(1), 1024)           # single-image neurons
    net = nn.Sequential(
        l0, nn.ReLU(),
        nn.Linear(1024, 1024), nn.ReLU(), nn.Identity(),
        nn.Linear(1024, 200),
    )
    y = torch.randint(0, 200, (x.shape[0],))
    out = net(x.flatten(1))
    loss = F.cross_entropy(out, y)
    grads = torch.autograd.grad(loss, [p for p in net.parameters()])
    names = [n for n, _ in net.named_parameters()]
    gdict = {f"net.{n}": g.detach() for n, g in zip(names, grads)}

    os.makedirs(os.path.join(root, "models"), exist_ok=True)
    os.makedirs(os.path.join(root, "gradients"), exist_ok=True)
    state = {f"net.{n}": p.detach() for n, p in net.named_parameters()}
    torch.save(state, os.path.join(root, "models", "model1.pt"))
    torch.save({"gradients": gdict, "family": "mlp", "activation": "relu",
                "feature_shape": (3, 64, 64), "batch_size": 128},
               os.path.join(root, "gradients", "model1.pt"))
    return x


class TrapCNN(nn.Module):
    def __init__(self, x: torch.Tensor):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        with torch.no_grad():
            # 3 transmit filters (delta at centre) copy the RGB channels.
            self.conv.weight.zero_()
            for c in range(3):
                self.conv.weight[c, c, 1, 1] = 1.0
            # remaining channels: random small (noise), negative bias to zero them
            self.conv.weight[3:] = 0.01 * torch.randn(5, 3, 3, 3)
            self.conv.bias.zero_()
            self.conv.bias[3:] = -5.0
            feat = F.relu(self.conv(x)).flatten(1)       # fc1 input over batch
        self.fc1 = _isolating_linear(feat, 512)          # single-image neurons
        self.head = nn.Linear(512, 200)

    def forward(self, x):
        h = F.relu(self.conv(x))
        h = h.flatten(1)
        h = F.relu(self.fc1(h))
        return self.head(h)


def make_cnn(root: str):
    torch.manual_seed(2)
    x = _make_batch(seed=11)
    model = TrapCNN(x)
    y = torch.randint(0, 200, (x.shape[0],))
    out = model(x)
    loss = F.cross_entropy(out, y)
    grads = torch.autograd.grad(loss, list(model.parameters()))
    names = [n for n, _ in model.named_parameters()]
    gdict = {n: g.detach() for n, g in zip(names, grads)}
    torch.save({n: p.detach() for n, p in model.named_parameters()},
               os.path.join(root, "models", "model2.pt"))
    torch.save({"gradients": gdict, "family": "cnn", "activation": "relu",
                "feature_shape": (8, 64, 64), "batch_size": 128},
               os.path.join(root, "gradients", "model2.pt"))
    return x


def main():
    root = tempfile.mkdtemp(prefix="task3_selftest_")
    os.environ["TASK3_DATA_ROOT"] = root
    # config was imported with the default root; patch its module globals.
    config.DATA_ROOT = root
    config.MODELS_DIR = os.path.join(root, "models")
    config.GRADIENTS_DIR = os.path.join(root, "gradients")

    x_mlp = make_mlp(root)
    x_cnn = make_cnn(root)

    import channels
    import reconstruct_v2 as v2

    print("=== channel transmit detection (CNN model2) ===")
    channels.report([2])

    print("\n=== MLP model1 recovery ===")
    rec_mlp, m1 = v2.recover_model(1)
    s_mlp = _nearest_ssim(rec_mlp, x_mlp)
    print(f"  method={m1}  nearest-match SSIM(recon->truth)={s_mlp:.3f}")

    print("\n=== CNN model2 recovery ===")
    rec_cnn, m2 = v2.recover_model(2)
    s_cnn = _nearest_ssim(rec_cnn, x_cnn)
    print(f"  method={m2}  nearest-match SSIM(recon->truth)={s_cnn:.3f}")

    ok = s_mlp > 0.6 and s_cnn > 0.6
    print(f"\nSELFTEST {'PASS' if ok else 'CHECK'} "
          f"(MLP {s_mlp:.3f}, CNN {s_cnn:.3f}; both should be high for trap weights)")


if __name__ == "__main__":
    main()
