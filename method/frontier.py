#!/usr/bin/env python3
"""Recompute CHAIR_s/i/recall for every saved method/*.jsonl (join pred with CHAIR gt by image_id) and
print a Pareto table sorted by recall, marking points that DOMINATE the plain-PAI frontier."""
import json, glob, os
from datasets import load_dataset
ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
GT={i:list(ds[i]["gt_object"]) for i in range(len(ds))}
def metrics(rows):
    res=[{"answer":GT[r["image_id"]],"pred":r.get("pred",[])} for r in rows if "image_id" in r and "pred" in r]
    if not res: return None
    n=len(res); halluc_s=0; allm=0; hallm=0; mt=0; tot=0
    for r in res:
        gt=set(r["answer"]); pred=r["pred"]
        allm+=len(pred); hallm+=sum(1 for w in pred if w not in gt)
        if any(w not in gt for w in pred): halluc_s+=1
        mt+=len(gt&set(pred)); tot+=len(gt)
    return dict(n=n, CHAIR_s=100*halluc_s/n, CHAIR_i=100*hallm/max(allm,1), recall=100*mt/max(tot,1),
               objs=allm/n)
rows=[]
for f in sorted(glob.glob("/volume/exploration/EvolvingLMMs/method/*.jsonl")):
    try:
        data=[json.loads(l) for l in open(f)]
        m=metrics(data)
        if m and m["n"]>=80: rows.append((os.path.basename(f), m))
    except Exception: pass
# reference PAI frontier points (n=120, from logs)
print(f"{'method':<22}{'n':>4}{'CHAIR_s':>9}{'CHAIR_i':>9}{'recall':>8}{'objs':>7}")
print("-"*60)
print(f"{'BASELINE':<22}{120:>4}{53.30:>9.1f}{15.60:>9.1f}{80.20:>8.1f}{7.5:>7.1f}")
print(f"{'PAI a0.3':<22}{120:>4}{49.20:>9.1f}{13.10:>9.1f}{82.10:>8.1f}{7.8:>7.1f}")
print(f"{'PAI a0.5':<22}{120:>4}{30.00:>9.1f}{11.70:>9.1f}{71.60:>8.1f}{9.1:>7.1f}")
print("-"*60)
for name,m in sorted(rows,key=lambda x:-x[1]["recall"]):
    print(f"{name:<22}{m['n']:>4}{m['CHAIR_s']:>9.1f}{m['CHAIR_i']:>9.1f}{m['recall']:>8.1f}{m['objs']:>7.1f}")
