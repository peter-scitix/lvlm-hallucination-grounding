#!/usr/bin/env python3
"""架构无关消除 killer figure: InstructBLIP(Q-Former) 上的 self-verify + ban 消除。
logit-lens/EAZY 在 Q-Former(32 query token, 非patch)上无定义/失效; 但 self-verify(行为信号)可用。
方法: 生成caption -> 对每个mentioned object 问 is-there-X -> 对判为幻觉的, 重生成时ban其输出token(+可选不ban对照)。
证: 在 Q-Former 上我们能消除幻觉, 而内部信号方法跑不了。 报 baseline vs sv+ban 的 CHAIR。"""
import argparse, importlib.util, json, os, sys
from collections import defaultdict
if "--gpu" in sys.argv: os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[sys.argv.index("--gpu")+1]
import torch
import numpy as np
from datasets import load_dataset
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
def CU():
    p="/volume/exploration/EvolvingLMMs/lmms-eval/lmms_eval/tasks/coco_cap_chair/utils.py"
    s=importlib.util.spec_from_file_location("cu",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ap=argparse.ArgumentParser()
ap.add_argument("--model",default="Salesforce/instructblip-vicuna-7b")
ap.add_argument("--n",type=int,default=250); ap.add_argument("--max_new_tokens",type=int,default=256)
ap.add_argument("--gpu",default="1"); ap.add_argument("--out",default="method/ib_svban.jsonl")
a=ap.parse_args(); cu=CU()
torch.set_grad_enabled(False)
proc=InstructBlipProcessor.from_pretrained(a.model)
model=InstructBlipForConditionalGeneration.from_pretrained(a.model,torch_dtype=torch.float16,device_map="cuda:0").eval()
DEV="cuda:0"; tok=proc.tokenizer
YES=tok.encode("Yes",add_special_tokens=False)[-1]; NO=tok.encode("No",add_special_tokens=False)[-1]
yl_=tok.encode("yes",add_special_tokens=False)[-1]; nl_=tok.encode("no",add_special_tokens=False)[-1]
IRREG={"man":"men","woman":"women","person":"people","child":"children"}
canon2ft=defaultdict(set)
for w,canon in cu.INVERSE_SYNONYM_DICT.items():
    for s in [w,w+"s",w.rstrip("s")," "+w," "+w+"s",w.capitalize()]+([" "+IRREG[w]] if w in IRREG else []):
        ids=tok.encode(s,add_special_tokens=False)
        if ids: canon2ft[canon].add(ids[-1])
_svc={}
@torch.no_grad()
def caption(image,banned=None):
    inp=proc(images=image,text="Describe the image in detail.",return_tensors="pt").to(DEV,torch.float16)
    kw={"bad_words_ids":[[t] for t in banned]} if banned else {}
    out=model.generate(**inp,max_new_tokens=a.max_new_tokens,do_sample=False,num_beams=1,**kw)
    return proc.batch_decode(out,skip_special_tokens=True)[0].strip()
@torch.no_grad()
def sv_no(image,o):
    if o in _svc: return _svc[o]
    inp=proc(images=image,text=f"Is there a {o} in the image? Answer yes or no.",return_tensors="pt").to(DEV,torch.float16)
    lg=model(**inp).logits[0,-1,:].float()
    v=float(torch.logsumexp(lg[[NO,nl_]],0)-torch.logsumexp(lg[[YES,yl_]],0)); _svc[o]=v; return v
ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
idxs=[i for i in range(len(ds)) if i%2==1][:a.n]
res={"baseline":[],"sv_ban":[]}; nab=0; fout=open(a.out,"w")
print(f"[ibsv] n={len(idxs)} model={a.model}",flush=True)
for c,i in enumerate(idxs):
    img=ds[i]["image"].convert("RGB"); _svc.clear()
    cap=caption(img); _,node,_,_=cu.caption_to_words(cap); objs=set(node)
    bad=[o for o in objs if sv_no(img,o)>0]
    banned=list({t for o in bad for t in canon2ft.get(o,set())})
    cap2=caption(img,banned) if banned else cap
    if banned: nab+=1
    res["baseline"].append({"answer":list(ds[i]["gt_object"]),"pred":node})
    res["sv_ban"].append({"answer":list(ds[i]["gt_object"]),"pred":cu.caption_to_words(cap2)[1]})
    fout.write(json.dumps({"i":i,"bad":bad})+"\n")
    if (c+1)%40==0: print(f"[ibsv] {c+1}/{len(idxs)} ablated={nab}",flush=True)
fout.close()
def agg(r): return (cu.coco_cap_chair_aggregate_results_chair_s(r),cu.coco_cap_chair_aggregate_results_chair_i(r),cu.coco_cap_chair_aggregate_results_recall(r),sum(len(x['pred']) for x in r)/len(r))
b=agg(res["baseline"]); v=agg(res["sv_ban"])
print(f"\n[ibsv] InstructBLIP (Q-Former; logit-lens/EAZY 在此无定义)")
print(f"[ibsv] baseline  {b[0]:.2f}/{b[1]:.2f}/{b[2]:.2f} objs={b[3]:.2f}")
print(f"[ibsv] sv+ban    {v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f} objs={v[3]:.2f}  ΔCHAIR_s={v[0]-b[0]:+.2f} (ablated={nab})")
