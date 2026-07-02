#!/usr/bin/env python3
"""可部署方法: 训练无关 post-hoc 文本切除 + base-rate-aware 保护。
generate(baseline caption, 已在F_base.jsonl) -> 对每个 mentioned object 算 grounding gc(已在excise_gc.json)
-> flag: gc<tau 且 object 不在 PROTECT(cal-split低幻觉率类) -> 从 caption 文本里删掉该 object 的名词短语
-> 输出修正caption + 重算CHAIR。这是 excise frontier 的真实文本realization(对CHAIR等价list-filter)。
在 test split(odd) 评, PROTECT 来自 cal(even)。同时打印样例caption检查是否破碎。"""
import json, re, importlib.util, argparse
p="lmms-eval/lmms_eval/tasks/coco_cap_chair/utils.py"
s=importlib.util.spec_from_file_location("cu",p);cu=importlib.util.module_from_spec(s);s.loader.exec_module(cu)
from datasets import load_dataset
ap=argparse.ArgumentParser()
ap.add_argument("--tau",type=float,default=-0.06)
ap.add_argument("--protect",default="person,tennis racket,surfboard,sports ball,elephant,umbrella")
ap.add_argument("--split",default="test"); ap.add_argument("--show",type=int,default=5)
a=ap.parse_args()
PROTECT=set(x.strip() for x in a.protect.split(",") if x.strip())
ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
gt={i:list(ds[i]["gt_object"]) for i in range(len(ds))}
gc=json.load(open("method/excise_gc.json"))
base=[json.loads(l) for l in open("method/F_base.jsonl")]
rows=[r for r in base if (a.split=="all" or (a.split=="test" and r["image_id"]%2==1) or (a.split=="cal" and r["image_id"]%2==0))]
# canonical -> 同义surface(用于在文本里定位删除)
canon2surf={}
for w,canon in cu.INVERSE_SYNONYM_DICT.items(): canon2surf.setdefault(canon,set()).add(w)
def excise_text(cap, flagged_canons):
    """删掉 flagged canonical object 的名词短语: 匹配 (a|an|the|数词)? (adj)* surface(s)? , 尽量整短语删。"""
    out=cap
    surfs=set()
    for c in flagged_canons: surfs|=canon2surf.get(c,{c})
    # 长surface优先(dining table 先于 table)
    for sf in sorted(surfs,key=lambda x:-len(x)):
        # 删 "a/an/the/some/two.. (形容词)* sf(s)": 保守删冠词+可选形容词+名词
        pat=r'\b(?:a|an|the|some|several|two|three|four|many|various)?\s*(?:\w+\s+){0,2}'+re.escape(sf)+r's?\b'
        out=re.sub(pat, ' ', out, flags=re.IGNORECASE)
    out=re.sub(r'\s+',' ',out)
    out=re.sub(r'\s+([.,;])',r'\1',out)
    out=re.sub(r'([.,;])\s*([.,;])',r'\1',out)
    return out.strip()
def agg(preds):
    R=[{"answer":gt[r["image_id"]],"pred":pr} for r,pr in zip(rows,preds)]
    return (cu.coco_cap_chair_aggregate_results_chair_s(R),cu.coco_cap_chair_aggregate_results_chair_i(R),
            cu.coco_cap_chair_aggregate_results_recall(R),sum(len(x) for x in preds)/len(preds))
# baseline
b=agg([r["pred"] for r in rows]); print(f"[{a.split} n={len(rows)}] baseline: {b[0]:.2f}/{b[1]:.2f}/{b[2]:.2f} objs={b[3]:.2f}")
# list-filter(上界) vs 真实文本切除
lf_preds=[]; tx_preds=[]; samples=[]
for r in rows:
    d=gc.get(str(r["image_id"]),{})
    flagged=[o for o in set(r["pred"]) if o not in PROTECT and d.get(o,1.0)<a.tau]
    # list-filter pred
    lf_preds.append([o for o in r["pred"] if o not in flagged])
    # 真实文本切除 -> 重parse
    newcap=excise_text(r["caption"], set(flagged)) if flagged else r["caption"]
    _,node,_,_=cu.caption_to_words(newcap); tx_preds.append(node)
    if flagged and len(samples)<a.show: samples.append((r["image_id"],flagged,r["caption"],newcap))
lf=agg(lf_preds); tx=agg(tx_preds)
print(f"[{a.split}] list-filter(上界)   tau={a.tau} protect={len(PROTECT)}类: {lf[0]:.2f}/{lf[1]:.2f}/{lf[2]:.2f} objs={lf[3]:.2f}")
print(f"[{a.split}] 真实文本切除        tau={a.tau}: {tx[0]:.2f}/{tx[1]:.2f}/{tx[2]:.2f} objs={tx[3]:.2f}")
print(f"\n样例(检查caption是否破碎):")
for iid,fl,old,new in samples:
    print(f"--- img{iid} 删除{fl}")
    print(f"  原: {old[:150]}")
    print(f"  新: {new[:150]}")
