"""Audit of the current CV-winner pipeline: where is the FPR budget spent,
where is TPR lost, and is the model well-specified per scheme/length.

All local, no GPU/cluster needed — uses the existing 5-fold CV harness.

Usage: python audit_results.py
"""

from __future__ import annotations

import numpy as np

from cv_smm import CONFIGS, N_FOLDS, SEED, build_cache, load_entropy, load_labeled
from detectors import H0_MOMENTS
from fit_smm import ASSIGN_Z, MIN_SPAN_FIT, fit_params, iter_spans
from smm_scorer import doc_signals, score_document

CONFIG = "b50_ps2_elo_entbin5"


def assign_scheme(signals, s, e):
    best_name, best_z = None, ASSIGN_Z
    for name, (sig, valid) in signals.items():
        mu0, var0 = H0_MOMENTS.get(name, (0.5, 0.25 * 0.75))
        v = valid[s:e]
        if v.sum() < 5:
            continue
        x = (sig[s:e][v] - mu0) / np.sqrt(var0)
        z = x.sum() / np.sqrt(len(x))
        if z >= best_z:
            best_name, best_z = name, z
    return best_name, best_z


def main():
    docs, kgw = load_labeled()
    cache = build_cache(docs, kgw)
    entropy = load_entropy()
    kwargs = CONFIGS[CONFIG]

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(docs))
    folds = [order[i::N_FOLDS] for i in range(N_FOLDS)]

    all_scores, all_labels, all_docid, all_pos = [], [], [], []
    span_records = []  # (doc_id, s, e, scheme, best_z, mean_score, valid_frac, edge)

    for fold in folds:
        held = set(fold.tolist())
        fit_docs = [docs[i] for i in range(len(docs)) if i not in held]
        params = fit_params(fit_docs, kgw=kgw, signal_cache=cache,
                            entropy_cache=entropy, **kwargs)
        for i in fold:
            rec = docs[i]
            did = str(rec["document_id"])
            signals = cache[did]
            ent = entropy[did] if entropy is not None and did in entropy else None
            sc = score_document(rec["token_ids"], params=params, signals=signals,
                                entropy=ent)
            lab = np.array(rec["labels"])
            n = len(lab)
            all_scores.append(sc)
            all_labels.append(lab)
            all_docid.extend([did] * n)
            all_pos.extend(range(n))

            for k, s, e in iter_spans(lab):
                if k != 1:
                    continue
                scheme, z = assign_scheme(signals, s, e)
                sig, valid = signals.get(scheme, (None, None)) if scheme else (None, None)
                valid_frac = float(valid[s:e].mean()) if valid is not None else float("nan")
                span_records.append(dict(
                    doc_id=did, s=s, e=e, length=e - s, scheme=scheme, z=z,
                    mean_score=float(sc[s:e].mean()), min_score=float(sc[s:e].min()),
                    valid_frac=valid_frac, edge=(s == 0 or e == n)))

    s_all = np.concatenate(all_scores)
    y_all = np.concatenate(all_labels)
    docid_arr = np.array(all_docid)
    pos_arr = np.array(all_pos)

    clean = np.sort(s_all[y_all == 0])[::-1]
    k = max(int(len(clean) * 0.001), 1)
    tau = clean[k - 1]
    tpr = float((s_all[y_all == 1] >= tau).mean())
    print(f"=== Pooled metric ({CONFIG}) ===")
    print(f"tau@0.1%FPR = {tau:.4f}  |  TPR = {tpr:.4f}  |  "
          f"n_clean={len(clean)}, n_wm={int((y_all==1).sum())}")

    # 1) False positives: top clean tokens by score
    print("\n=== Top 20 false positives (label=0, highest score) ===")
    fp_mask = y_all == 0
    idx = np.argsort(-s_all[fp_mask])[:20]
    fp_docid = docid_arr[fp_mask][idx]
    fp_pos = pos_arr[fp_mask][idx]
    fp_score = s_all[fp_mask][idx]
    doc_by_id = {str(d["document_id"]): d for d in docs}
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    except Exception as e:
        tok = None
        print(f"(tokenizer unavailable: {e})")
    for did, pos, sc in zip(fp_docid, fp_pos, fp_score):
        rec = doc_by_id[did]
        ids = rec["token_ids"]
        ctx = ids[max(0, pos - 6):pos + 3]
        txt = tok.decode(ctx) if tok else str(ctx)
        print(f"  doc={did} pos={pos} score={sc:.4f}  ctx=...{txt!r}...")

    # 2) Missed detections: label=1 spans with lowest mean posterior
    print("\n=== Bottom 15 watermarked spans by mean score ===")
    span_records.sort(key=lambda r: r["mean_score"])
    for r in span_records[:15]:
        print(f"  doc={r['doc_id']} [{r['s']}:{r['e']}] len={r['length']} "
              f"scheme={r['scheme']} z={r['z']:.1f} mean_score={r['mean_score']:.4f} "
              f"valid_frac={r['valid_frac']:.2f} edge={r['edge']}")

    # 3) Per-scheme breakdown
    print("\n=== Per-scheme span stats ===")
    schemes = sorted(set(r["scheme"] for r in span_records), key=lambda x: (x is None, x))
    for sch in schemes:
        rs = [r for r in span_records if r["scheme"] == sch]
        if not rs:
            continue
        mean_scores = np.array([r["mean_score"] for r in rs])
        lengths = np.array([r["length"] for r in rs])
        valid_fracs = np.array([r["valid_frac"] for r in rs])
        detected = (mean_scores >= tau).mean()
        print(f"  {str(sch):10s} n_spans={len(rs):3d}  mean_len={lengths.mean():6.1f}  "
              f"mean(valid_frac)={np.nanmean(valid_fracs):.2f}  "
              f"mean(mean_score)={mean_scores.mean():.4f}  "
              f"frac_spans_detected(mean>=tau)={detected:.2f}")

    # 4) Length x scheme cross-tab (only assigned spans)
    print("\n=== Length distribution per scheme (assigned spans only) ===")
    for sch in schemes:
        if sch is None:
            continue
        lens = [r["length"] for r in span_records if r["scheme"] == sch]
        vals, counts = np.unique(lens, return_counts=True)
        print(f"  {sch}: " + ", ".join(f"{int(v)}x{int(c)}" for v, c in zip(vals, counts)))

    # 5) Edge vs middle spans
    print("\n=== Edge vs middle spans ===")
    for edge in (True, False):
        rs = [r for r in span_records if r["edge"] == edge]
        if not rs:
            continue
        mean_scores = np.array([r["mean_score"] for r in rs])
        print(f"  edge={edge}: n={len(rs)}  mean(mean_score)={mean_scores.mean():.4f}  "
              f"frac_detected={ (mean_scores >= tau).mean():.2f}")

    # 6) Unassigned spans (no scheme reached ASSIGN_Z) — invisible to the fit
    n_unassigned = sum(1 for r in span_records if r["scheme"] is None)
    n_short = sum(1 for r in span_records if r["length"] < MIN_SPAN_FIT)
    print(f"\nUnassigned spans (z < {ASSIGN_Z} on all detectors): "
          f"{n_unassigned}/{len(span_records)} "
          f"(of which {n_short} are short/<{MIN_SPAN_FIT}tok, excluded from fit anyway)")


if __name__ == "__main__":
    main()
