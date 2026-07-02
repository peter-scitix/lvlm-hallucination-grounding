#!/usr/bin/env python3
"""
Conformal Risk Control calibration for grounding-gated object suppression (no GPU; uses saved probe data).
TRACK B: compare candidate nonconformity scores (calibrated tkB / margin / combo) by AUROC.
TRACK A: cal/test split over images; for the chosen score, calibrate threshold gamma per target alpha via
         Conformal Risk Control (empirical hallucination-rate among KEPT objects + Hoeffding slack), and
         report cal & TEST CHAIR_i(kept) + recall(kept) at each gamma to check the guarantee E[CHAIR_i]<=alpha.
Outputs detect/conformal_gamma.json {alpha: gamma} for the decode-time gate.
"""
import json, numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score

v1={(r["image_id"],r["object"]):r for r in (json.loads(l) for l in open("detect/probe_full.jsonl"))}
v2={(r["image_id"],r["object"]):r for r in (json.loads(l) for l in open("detect/probe_v2.jsonl"))}
keys=[k for k in v1 if k in v2]
img=np.array([k[0] for k in keys]); obj=[k[1] for k in keys]
y=np.array([v1[k]["hallucinated"] for k in keys])          # 1 = hallucinated
tkB=np.array([v1[k]["tkB_L31"] for k in keys])
mar=np.array([v2[k]["margin_tk_L31"] for k in keys])
def demean(v):
    m=defaultdict(list)
    for o,x in zip(obj,v): m[o].append(x)
    mu={o:np.mean(x) for o,x in m.items()}; return np.array([x-mu[o] for o,x in zip(obj,v)]), mu
gc_tkB,mu_tkB=demean(tkB); gc_mar,mu_mar=demean(mar)
def z(v): return (v-v.mean())/(v.std()+1e-9)
combo=z(gc_tkB)+z(gc_mar)
def auc(score): a=roc_auc_score(y,-score); return max(a,1-a)   # low grounding -> halluc
print("=== TRACK B: nonconformity score AUROC (hallucinated detection) ===")
for name,s in [("gc_tkB",gc_tkB),("gc_margin",gc_mar),("z(tkB)+z(margin)",combo)]:
    print(f"  {name:18} AUROC={auc(s):.3f}")
score=gc_tkB   # gground.py's decode score (gc on tkB). keep consistent for the gate.

# ---- TRACK A: Conformal Risk Control on gc_tkB ----
# gate: KEEP object if score>=gamma (well grounded); SUPPRESS if score<gamma.
# CHAIR_i(kept) = fraction hallucinated among kept objects (this is what the emitted-object hallucination rate approximates).
uimg=np.unique(img); rng=np.arange(len(uimg))
cal_imgs=set(uimg[rng%2==0]); test_imgs=set(uimg[rng%2==1])   # 250/250 image split
cal=np.array([i in cal_imgs for i in img]); test=~cal
def chair_i_kept(mask, gamma):
    keep=mask & (score>=gamma)
    return (y[keep].mean() if keep.sum()>0 else 0.0), int(keep.sum()), int((keep&(y==0)).sum())
def recall_kept(mask, gamma):
    # fraction of real(kept) objects retained vs all real objects mentioned in mask
    real=mask & (y==0); keptreal=real & (score>=gamma)
    return keptreal.sum()/max(real.sum(),1)
grid=np.quantile(score, np.linspace(0.01,0.6,120))
delta=0.1
gamma_for={}
print("\n=== TRACK A: Conformal Risk Control (cal 250 imgs / test 250 imgs) ===")
print(f"{'alpha':>6} {'gamma':>8} {'cal_Ci':>7} {'test_Ci':>8} {'test_recallKept':>15} {'guarantee':>10}")
for alpha in [0.05,0.06,0.07,0.08,0.10,0.12]:
    chosen=None
    for gamma in sorted(grid):                                # smallest gamma (max recall) with cal risk+slack<=alpha
        ci,nk,_=chair_i_kept(cal,gamma)
        slack=np.sqrt(np.log(1/delta)/(2*max(nk,1)))          # Hoeffding upper conf
        if ci+slack<=alpha: chosen=gamma; break
    if chosen is None: chosen=grid.max()
    gamma_for[alpha]=float(chosen)
    tci,_,_=chair_i_kept(test,chosen); cci,_,_=chair_i_kept(cal,chosen); rk=recall_kept(test,chosen)
    ok="OK" if tci<=alpha+0.02 else "VIOL"
    print(f"{alpha:>6.2f} {chosen:>8.4f} {cci*100:>6.1f}% {tci*100:>7.1f}% {rk*100:>14.1f}% {ok:>10}")
json.dump({str(k):v for k,v in gamma_for.items()}, open("detect/conformal_gamma.json","w"),indent=1)
print("\nsaved detect/conformal_gamma.json  (alpha -> gamma on gc_tkB, for gground.py --stau=gamma)")
