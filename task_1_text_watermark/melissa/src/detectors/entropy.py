"""Per-token entropy from a proxy LM (TextSeal §3.2 entropy weighting).

Low-entropy positions carry almost no watermark signal (the top token dominates
regardless of the PRF), so entropy tells the calibrator how much any detector score
should count. We estimate ``H_i`` with a single forward pass of a small proxy model
teacher-forced on the document's ``token_ids``.

If transformers/torch or a GPU are unavailable, we return a cheap key-free proxy
(local token-id novelty) so the pipeline still runs; the calibrator treats it as one
more feature.
"""

from __future__ import annotations

from typing import Sequence

from .. import config
from ..config import ENTROPY_PROXY_MODEL

_MODEL_CACHE: dict = {}
# Once the proxy LM is unavailable (or explicitly disabled) we remember it so we do
# NOT retry the (failing) Hugging Face load on every single document.
_PROXY_DISABLED = False
_DISABLED_LOGGED = False


def _disable_proxy(reason: str) -> None:
    """Permanently fall back to the novelty proxy and log the reason once."""
    global _PROXY_DISABLED, _DISABLED_LOGGED
    _PROXY_DISABLED = True
    if not _DISABLED_LOGGED:
        print(f"[entropy] proxy LM disabled, using novelty proxy only ({reason})")
        _DISABLED_LOGGED = True


def _novelty_proxy(token_ids: Sequence[int], window: int = 50) -> list[float]:
    """Cheap entropy stand-in: local fraction of not-recently-seen tokens."""
    n = len(token_ids)
    out = [0.0] * n
    for i in range(n):
        lo = max(0, i - window)
        recent = token_ids[lo:i]
        out[i] = 0.0 if not recent else float(token_ids[i] not in set(recent))
    return out


def entropy_scores(token_ids: Sequence[int], use_proxy_lm: bool = True) -> list[float]:
    """Per-token predictive entropy H_i (nats), or a novelty proxy as fallback."""
    # Explicitly disabled by caller, by config/env, or by a previous failed attempt.
    if not use_proxy_lm:
        return _novelty_proxy(token_ids)
    if config.DISABLE_PROXY_LM:
        _disable_proxy("WML_DISABLE_PROXY_LM=1")
        return _novelty_proxy(token_ids)
    if _PROXY_DISABLED:
        return _novelty_proxy(token_ids)
    try:
        import torch
        from transformers import AutoModelForCausalLM

        if not torch.cuda.is_available():
            _disable_proxy("no CUDA")
            return _novelty_proxy(token_ids)

        model = _MODEL_CACHE.get("model")
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                ENTROPY_PROXY_MODEL, torch_dtype=torch.float16
            ).to("cuda").eval()
            _MODEL_CACHE["model"] = model

        ids = torch.tensor([list(token_ids)], device="cuda")
        with torch.no_grad():
            logits = model(ids).logits[0]  # (n, vocab)
            logp = torch.log_softmax(logits.float(), dim=-1)
            p = logp.exp()
            ent = -(p * logp).sum(dim=-1)  # (n,) entropy of dist predicting each position
        # Shift: entropy of the distribution that *produced* token i is at position i-1.
        ent = ent.roll(1)
        ent[0] = 0.0
        return ent.tolist()
    except Exception as exc:  # noqa: BLE001
        # Disable permanently so we never retry the failing HF load per-document.
        _disable_proxy(f"proxy LM unavailable: {exc}")
        return _novelty_proxy(token_ids)
