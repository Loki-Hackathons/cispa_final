"""Log-odds ensemble of several SMM configs (roadmap.md, final step).

Scores from score_document are sigmoid(log_odds / 40), so averaging in
logit space is exactly averaging the watermark log-odds of the component
models — the principled fusion for a ranking metric. A monotone transform
of any single model leaves TPR@FPR unchanged, so the ensemble can only help
through genuine disagreement between components.

Two modes:

CV (decision): 5-fold protocol identical to cv_smm.eval_config, all
components fitted on the same folds, per-doc log-odds averaged before
pooling. Accept the ensemble only if it beats every component on the same
seed (and re-check on --seed 1).

    python ensemble_smm.py --cv b50_ps2_elo_entbin5_iso b50_ps2_elo_entbin5_iso_kgwxens [--seed 0]

Combine (submission): average existing submission jsonl files.

    python ensemble_smm.py --combine submission_a.jsonl submission_b.jsonl --out submission_ens.jsonl
"""

from __future__ import annotations

import argparse
import json

import numpy as np

EPS = 1e-9


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_logit(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, EPS, 1.0 - EPS)
    return np.log(s / (1.0 - s))


def cv_scores_for_config(name, docs, kgw, cache, entropy, exact_cache,
                         p_target, lpg, seed) -> dict[str, np.ndarray]:
    """Per-document held-out scores under the cv_smm fold protocol."""
    from cv_smm import CONFIGS, N_FOLDS, _needs_entropy, _needs_exact, _needs_lpg
    from fit_smm import fit_params
    from smm_scorer import score_document

    kwargs = CONFIGS[name]
    needs_entropy = _needs_entropy(kwargs)
    needs_exact = _needs_exact(kwargs)
    needs_lpg = _needs_lpg(kwargs)
    if kwargs is not None:
        kwargs = {k: v for k, v in kwargs.items() if k != "use_synthetic"}
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(docs))
    folds = [order[i::N_FOLDS] for i in range(N_FOLDS)]
    out = {}
    for fold in folds:
        held = set(fold.tolist())
        fit_docs = [docs[i] for i in range(len(docs)) if i not in held]
        params = fit_params(fit_docs, kgw=kgw, signal_cache=cache,
                            entropy_cache=entropy if needs_entropy else None,
                            **kwargs)
        for i in fold:
            rec = docs[i]
            did = str(rec["document_id"])
            out[did] = score_document(
                rec["token_ids"], params=params, signals=cache[did],
                entropy=entropy[did] if needs_entropy else None,
                exact_signals=exact_cache[did] if needs_exact else None,
                p_target=p_target[did] if needs_exact else None,
                lpg=lpg[did] if needs_lpg else None)
    return out


def run_cv(names, seed):
    from cv_smm import (build_cache, build_exact_cache, load_entropy,
                        load_labeled, load_lpg, load_p_target, tpr_at_fpr)

    docs, kgw = load_labeled()
    print(f"{len(docs)} labeled docs; building signal cache...", flush=True)
    cache = build_cache(docs, kgw)
    entropy = load_entropy()
    p_target = load_p_target()
    lpg = load_lpg()
    exact_cache = build_exact_cache(docs) if p_target is not None else None

    per_config = {}
    labels = {str(d["document_id"]): np.array(d["labels"]) for d in docs}
    for name in names:
        per_config[name] = cv_scores_for_config(
            name, docs, kgw, cache, entropy, exact_cache, p_target, lpg, seed)
        s = [per_config[name][d] for d in labels]
        y = [labels[d] for d in labels]
        print(f"{name:36s} CV TPR@0.1%FPR = {tpr_at_fpr(s, y):.4f}", flush=True)

    ens_scores, ys = [], []
    for did in labels:
        logits = np.mean([to_logit(per_config[n][did]) for n in names], axis=0)
        ens_scores.append(1.0 / (1.0 + np.exp(-logits)))
        ys.append(labels[did])
    print(f"{'ENSEMBLE(' + str(len(names)) + ')':36s} CV TPR@0.1%FPR = "
          f"{tpr_at_fpr(ens_scores, ys):.4f}   (seed {seed})", flush=True)


def run_combine(paths, out_path, weights=None):
    subs = [ {str(r['document_id']): np.asarray(r['scores'], dtype=np.float64)
              for r in read_jsonl(p)} for p in paths ]
    dids = list(subs[0])
    for s in subs[1:]:
        if set(s) != set(dids):
            raise SystemExit("submission files cover different documents")
    if weights is None:
        w = np.full(len(subs), 1.0 / len(subs))
    else:
        if len(weights) != len(subs):
            raise SystemExit("--weights must match the number of files")
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
    with open(out_path, "w", encoding="utf-8") as f:
        for did in dids:
            logits = sum(wi * to_logit(s[did]) for wi, s in zip(w, subs))
            sc = 1.0 / (1.0 + np.exp(-logits))
            f.write(json.dumps({"document_id": did,
                                "scores": [float(x) for x in sc]}) + "\n")
    print(f"Wrote {len(dids)} docs to {out_path} "
          f"(ensemble of {len(paths)} submissions, weights={list(np.round(w, 3))})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", nargs="+", metavar="CONFIG")
    ap.add_argument("--combine", nargs="+", metavar="SUBMISSION_JSONL")
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="per-file weights for --combine (renormalized)")
    ap.add_argument("--out", default="submission_ensemble.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if bool(args.cv) == bool(args.combine):
        raise SystemExit("use exactly one of --cv / --combine")
    if args.cv:
        from cv_smm import CONFIGS
        for n in args.cv:
            if n not in CONFIGS:
                raise SystemExit(f"unknown config {n}")
        run_cv(args.cv, args.seed)
    else:
        run_combine(args.combine, args.out, weights=args.weights)


if __name__ == "__main__":
    main()
