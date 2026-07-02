#!/usr/bin/env python3
"""A0 pivotal: 模型能否抓自己生成时犯的幻觉? 对 F_base caption 里每个 mentioned object 聚焦问
"Is there a {o} in the image?" 取模型 yes/no logit; 与 probe_full 的 hallucinated 标签比。
若 self-check AUROC >> grounding 的 0.82 => 用模型自我验证做移除决策可突破检测精度天花板(GOAL方法A)。
也测 grounding(召回)→self-verify(精度) 两阶段的联合精度。"""
import argparse, json, os, sys
if "--gpu" in sys.argv: os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[sys.argv.index("--gpu")+1]
import torch, torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
SYS=("A chat between a curious user and an artificial intelligence assistant. "
     "The assistant gives helpful, detailed, and polite answers to the user's questions.")
ap=argparse.ArgumentParser()
ap.add_argument("--gpu",default="1"); ap.add_argument("--probe",default="detect/probe_full.jsonl")
ap.add_argument("--gccache",default="method/excise_gc.json")
ap.add_argument("--out",default="detect/selfverify.json"); ap.add_argument("--recompute",action="store_true")
a=ap.parse_args()
torch.set_grad_enabled(False)
proc=AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf");tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",torch_dtype=torch.float16,device_map="cuda:0",attn_implementation="eager").eval()
img_id=model.config.image_token_index;DEV="cuda:0"
YES=tok.encode("Yes",add_special_tokens=False)[0]; NO=tok.encode("No",add_special_tokens=False)[0]
yl_=tok.encode("yes",add_special_tokens=False)[0]; nl_=tok.encode("no",add_special_tokens=False)[0]
ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
probe=[json.loads(l) for l in open(a.probe)]

@torch.no_grad()
def selfcheck(image,obj):
    prompt=f"{SYS} USER: <image>\nIs there a {obj} in the image?\nAnswer the question using a single word or phrase. ASSISTANT:"
    inp=proc(images=image,text=prompt,return_tensors="pt").to(DEV,torch.float16)
    lg=model(**inp).logits[0,-1,:].float()
    yl=float(torch.logsumexp(lg[[YES,yl_]],0)); nl=float(torch.logsumexp(lg[[NO,nl_]],0))
    return yl,nl

if a.recompute or not os.path.exists(a.out):
    # group by image to reuse loaded image
    byimg={}
    for p in probe: byimg.setdefault(int(p["image_id"]),[]).append(p["object"])
    out={}
    for c,(i,objs) in enumerate(byimg.items()):
        img=ds[i]["image"].convert("RGB")
        d={}
        for o in set(objs):
            yl,nl=selfcheck(img,o); d[o]={"yl":yl,"nl":nl}
        out[str(i)]=d
        if (c+1)%50==0: print(f"selfverify {c+1}/{len(byimg)}",flush=True)
    json.dump(out,open(a.out,"w")); print("saved",a.out,flush=True)
else:
    out=json.load(open(a.out)); print("loaded",flush=True)

# ==== analyze ====
from sklearn.metrics import roc_auc_score
gc=json.load(open(a.gccache)) if os.path.exists(a.gccache) else {}
rows=[]
for p in probe:
    i=str(p["image_id"]); o=p["object"]; d=out.get(i,{}).get(o)
    if d is None: continue
    g=gc.get(i,{}).get(o)
    rows.append({"y":int(p["hallucinated"]),"yl":d["yl"],"nl":d["nl"],"gc":g})
y=np.array([r["y"] for r in rows])
sv=np.array([r["nl"]-r["yl"] for r in rows])   # high => model says NO => hallucination
def au(s): a_=roc_auc_score(y,s); return max(a_,1-a_)
print(f"\n[selfverify] n={len(rows)} hallucinated={int(y.sum())} base_rate={y.mean():.3f}")
print(f"  self-check (nl-yl)  AUROC={au(sv):.4f}   <- 模型自我验证抓自己的幻觉")
# 硬 yes/no: model says no
pred_no=(sv>0).astype(int)
tp=((pred_no==1)&(y==1)).sum(); fp=((pred_no==1)&(y==0)).sum(); fn=((pred_no==1)&(y==0)).sum()
prec=tp/max((pred_no==1).sum(),1); rec=tp/max(y.sum(),1)
print(f"  model-says-no as detector: precision={prec:.3f} recall={rec:.3f} (flagged {int((pred_no==1).sum())}, 其中真幻觉{int(tp)})")
# grounding 对照 + 组合
have_g=[r for r in rows if r["gc"] is not None]
if have_g:
    yg=np.array([r["y"] for r in have_g]); g=np.array([r["gc"] for r in have_g]); s2=np.array([r["nl"]-r["yl"] for r in have_g])
    def au2(s,yy): a_=roc_auc_score(yy,s); return max(a_,1-a_)
    print(f"  [同子集 n={len(have_g)}] grounding(-gc) AUROC={au2(-g,yg):.4f}  self-check AUROC={au2(s2,yg):.4f}")
    def z(v): return (v-v.mean())/(v.std()+1e-9)
    combo=z(s2)-z(g)   # 都是 高=>幻觉
    print(f"  组合 z(self-check)+z(-grounding) AUROC={au2(combo,yg):.4f}")
    # 两阶段: grounding召回top-K候选, 在候选内用self-check精度
    for q in [0.5,0.4,0.3]:
        thr=np.quantile(g,q)  # 低gc候选
        cand=g<=thr
        if cand.sum()>0 and len(set(yg[cand]))>1:
            print(f"  两阶段: grounding召回最低{int(q*100)}%({int(cand.sum())}个,含真幻觉{int(yg[cand].sum())}) 内 self-check AUROC={au2(s2[cand],yg[cand]):.4f}")
