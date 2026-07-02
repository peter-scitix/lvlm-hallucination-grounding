#!/usr/bin/env python3
"""
Analyze detection separability from the logit-lens probe output.

For each signal column (peakA/B, tkA/B, looseA/B at each layer) compute ROC-AUC for predicting
hallucinated=1. We report the DIRECTED AUROC = max(auc, 1-auc) with the implied direction, since
e.g. high peak grounding -> FAITHFUL (so raw AUROC < 0.5 means a strong detector with flipped sign).

Training-free check: we also report a simple UNSUPERVISED fixed combination (z-score sum of the two
best signals, oriented so higher = more hallucinated) — NO learned classifier, unlike the supervised
competitor 2604.04863.
"""
import argparse, json
import numpy as np
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", default="detect/probe_full.jsonl")
ap.add_argument("--layers", default="4,8,12,14,16,20,24,31")
a = ap.parse_args()
layers = [int(x) for x in a.layers.split(",")]

rows = [json.loads(l) for l in open(a.inp)]
y = np.array([r["hallucinated"] for r in rows])
n_h, n_f = int(y.sum()), int((1 - y).sum())
print(f"rows={len(rows)}  hallucinated={n_h}  faithful={n_f}  base_rate={n_h/len(rows):.3f}\n")

def directed_auc(score, y):
    m = np.isfinite(score)
    if m.sum() < 10 or len(np.unique(y[m])) < 2:
        return None, None
    auc = roc_auc_score(y[m], score[m])
    # directed: detector may use +score or -score
    return (auc, "+") if auc >= 0.5 else (1 - auc, "-")

def col(name):
    return np.array([r.get(name, np.nan) for r in rows], dtype=float)

print(f"{'signal':<10} " + " ".join(f"L{L:>2}" for L in layers) + "   best")
sig_best = {}
for sig in ["peakA", "peakB", "tkA", "tkB", "looseA", "looseB"]:
    line, best = [], (0.0, None, None)
    for L in layers:
        auc, d = directed_auc(col(f"{sig}_L{L}"), y)
        line.append(f"{auc:.3f}" if auc else "  -  ")
        if auc and auc > best[0]:
            best = (auc, L, d)
    sig_best[sig] = best
    star = f"  AUC={best[0]:.3f}@L{best[1]}({best[2]})" if best[1] else ""
    print(f"{sig:<10} " + " ".join(f"{x:>5}" for x in line) + star)

# ---- unsupervised fixed combination of the two strongest signals (one peak-type, one loose-type) ----
def z(v):
    v = v.astype(float); m = np.nanmean(v); s = np.nanstd(v) + 1e-9
    return (v - m) / s

peak_sigs = {k: v for k, v in sig_best.items() if k.startswith(("peak", "tk")) and v[1]}
loose_sigs = {k: v for k, v in sig_best.items() if k.startswith("loose") and v[1]}
if peak_sigs and loose_sigs:
    pk = max(peak_sigs, key=lambda k: peak_sigs[k][0]); pa, pL, pd = peak_sigs[pk]
    lk = max(loose_sigs, key=lambda k: loose_sigs[k][0]); la, lL, ld = loose_sigs[lk]
    # orient so higher = more hallucinated (flip if directed sign was '-')
    sp = z(col(f"{pk}_L{pL}")) * (-1 if pd == "-" else 1)
    sl = z(col(f"{lk}_L{lL}")) * (-1 if ld == "-" else 1)
    combo = sp + sl
    cauc = roc_auc_score(y[np.isfinite(combo)], combo[np.isfinite(combo)])
    print(f"\n[best peak-type] {pk}@L{pL}: AUC={pa:.3f} (dir {pd})")
    print(f"[best loose ]    {lk}@L{lL}: AUC={la:.3f} (dir {ld})")
    print(f"[unsupervised combo z({pk})+z({lk})] AUC={max(cauc,1-cauc):.3f}   (training-free, no learned classifier)")
