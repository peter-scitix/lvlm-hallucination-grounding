#!/usr/bin/env python3
"""POPE提点尝试: 多问法自一致性(TACO式, 无参数聚合, 不会过拟合) + 可选grounding作额外voter。
对每个POPE问题(object X), 用K种问法各问一次, 取yes/no logit margin; 聚合:
  vote(多数), mean_margin(自信度加权), + 加grounding voter。对比单次baseline。
survey说这条在LLaVA-1.5-7B上有效(TACO adv+5.3); 且方向对(模型错误主要是假阴)。"""
import argparse, json, os, sys, re
if "--gpu" in sys.argv: os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[sys.argv.index("--gpu")+1]
import torch, torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
SYS=("A chat between a curious user and an artificial intelligence assistant. "
     "The assistant gives helpful, detailed, and polite answers to the user's questions.")
PHRASINGS=[
 "Is there a {x} in the image?",
 "Does the image contain a {x}?",
 "Can you see a {x} in the image?",
 "Is a {x} present in this image?",
 "Look at the image carefully. Is there a {x}?",
 "Does a {x} appear anywhere in the image?",
]
ap=argparse.ArgumentParser()
ap.add_argument("--splits",default="adversarial,popular,random"); ap.add_argument("--n",type=int,default=500)
ap.add_argument("--K",type=int,default=6); ap.add_argument("--gpu",default="1")
a=ap.parse_args()
torch.set_grad_enabled(False)
proc=AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf");tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",torch_dtype=torch.float16,device_map="cuda:0",attn_implementation="eager").eval()
img_id=model.config.image_token_index;DEV="cuda:0";W=model.lm_head.weight;norm=model.model.language_model.norm
YES=tok.encode("Yes",add_special_tokens=False)[0]; NO=tok.encode("No",add_special_tokens=False)[0]
yl_=tok.encode("yes",add_special_tokens=False)[0]; nl_=tok.encode("no",add_special_tokens=False)[0]
cal=json.load(open("detect/calibration_tkb.json"));calmean=sum(cal.values())/len(cal)
def emb(o):
    ids=tok.encode(o,add_special_tokens=False); return F.normalize(W[ids].float().mean(0),dim=-1).to(DEV) if ids else None
def extract(q):
    m=re.search(r"[Ii]s there (?:a |an |the )?(.+?) in the image",q); return m.group(1).strip().lower() if m else None
@torch.no_grad()
def margin(image_inp,question):
    prompt=f"{SYS} USER: <image>\n{question}\nAnswer the question using a single word or phrase. ASSISTANT:"
    inp=proc(images=image_inp,text=prompt,return_tensors="pt").to(DEV,torch.float16)
    lg=model(**inp).logits[0,-1,:].float()
    return float(torch.logsumexp(lg[[YES,yl_]],0)-torch.logsumexp(lg[[NO,nl_]],0))  # >0 => yes
@torch.no_grad()
def grounding_present(image,o):
    e=emb(o)
    if e is None: return 0.0
    vl=proc(images=image,text=f"{SYS} USER: <image>\nDescribe. ASSISTANT:",return_tensors="pt").to(DEV,torch.float16)
    vis=(vl.input_ids[0]==img_id).nonzero(as_tuple=True)[0]
    ot=model(**vl,output_hidden_states=True,use_cache=False)
    hn=F.normalize(norm(ot.hidden_states[31][0,vis,:]).float(),dim=-1)
    return (hn@e).topk(min(10,hn.shape[0])).values.mean().item()-cal.get(o,calmean)
ALL=load_dataset("lmms-lab/POPE",split="test",token=False)
for split in a.splits.split(","):
    ds=ALL.filter(lambda x:x["category"]==split); n=min(a.n,len(ds))
    y=[]; single=[]; margins=[]; gcs=[]
    for i in range(n):
        ex=ds[i]; o=extract(ex["question"]);
        if o is None: continue
        img=ex["image"].convert("RGB"); gt=1 if ex["answer"].strip().lower()=="yes" else 0
        ms=[margin(img,p.format(x=o)) for p in PHRASINGS[:a.K]]
        y.append(gt); single.append(ms[0]); margins.append(ms)
        if (i+1)%200==0: print(f"[sc] {split} {i+1}/{n}",flush=True)
    y=np.array(y); single=np.array(single); M=np.array(margins)
    def acc(pred): return (pred==y).mean()
    base=acc((single>0).astype(int))
    vote=acc((( M>0).mean(1)>0.5).astype(int))            # 多数投票
    meanm=acc((M.mean(1)>0).astype(int))                  # 平均margin(自信度加权)
    # self-confidence: 每问法用|margin|加权
    conf=acc(((M*np.abs(M)).sum(1)>0).astype(int))
    print(f"[sc] {split:12} n={len(y)} single(baseline)={base:.4f}  vote={vote:.4f}  mean_margin={meanm:.4f}  conf_weighted={conf:.4f}  => {'提升✓' if max(vote,meanm,conf)>base+0.005 else '无提升'}")
