#!/usr/bin/env python3
"""第三架构 (InstructBLIP-Vicuna-7B, Q-Former) 验证 self-verify(生成-验证gap) 检测是否普适。
InstructBLIP 视觉token是32个Q-Former query(非patch), logit-lens grounding 不直接适用 → 只测 self-verify。
流程: 生成caption -> 每个mentioned COCO object 问"Is there X" 读yes/no margin -> 幻觉标签(∉GT) -> self-verify AUROC。
若 0.85+ => self-verify 跨3架构(LLaVA/Qwen/InstructBLIP)普适。"""
import argparse, importlib.util, json, os, sys
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
ap.add_argument("--n",type=int,default=400); ap.add_argument("--max_new_tokens",type=int,default=256)
ap.add_argument("--gpu",default="1"); ap.add_argument("--out",default="method/instructblip_detect.json")
a=ap.parse_args(); cu=CU()
ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
gt={i:set(ds[i]["gt_object"]) for i in range(len(ds))}
torch.set_grad_enabled(False)
proc=InstructBlipProcessor.from_pretrained(a.model)
model=InstructBlipForConditionalGeneration.from_pretrained(a.model,torch_dtype=torch.float16,device_map="cuda:0").eval()
DEV="cuda:0"; tok=proc.tokenizer
YES=tok.encode("Yes",add_special_tokens=False)[-1]; NO=tok.encode("No",add_special_tokens=False)[-1]
yl_=tok.encode("yes",add_special_tokens=False)[-1]; nl_=tok.encode("no",add_special_tokens=False)[-1]
@torch.no_grad()
def caption(image):
    inp=proc(images=image,text="Describe the image in detail.",return_tensors="pt").to(DEV,torch.float16)
    out=model.generate(**inp,max_new_tokens=a.max_new_tokens,do_sample=False,num_beams=1)
    return proc.batch_decode(out,skip_special_tokens=True)[0].strip()
@torch.no_grad()
def selfverify_no(image,o):
    inp=proc(images=image,text=f"Is there a {o} in the image? Answer yes or no.",return_tensors="pt").to(DEV,torch.float16)
    lg=model(**inp).logits[0,-1,:].float()
    return float(torch.logsumexp(lg[[NO,nl_]],0)-torch.logsumexp(lg[[YES,yl_]],0))
recs=[]
print(f"[ib] n={min(a.n,len(ds))} model={a.model} YES={YES} NO={NO}",flush=True)
for i in range(min(a.n,len(ds))):
    img=ds[i]["image"].convert("RGB")
    cap=caption(img); _,node,_,_=cu.caption_to_words(cap)
    for o in set(node):
        sv=selfverify_no(img,o); hall=int(o not in gt[i])
        recs.append({"i":i,"o":o,"hall":hall,"sv":sv})
    if (i+1)%50==0: print(f"[ib] {i+1}/{min(a.n,len(ds))} recs={len(recs)}",flush=True)
json.dump({"model":a.model,"recs":recs},open(a.out,"w"))
def auroc(y,s):
    y=np.array(y);s=np.array(s);pos=s[y==1];neg=s[y==0]
    if len(pos)==0 or len(neg)==0: return float("nan")
    return float((sum((pos[:,None]>neg[None,:]).sum(1))+0.5*sum((pos[:,None]==neg[None,:]).sum(1)))/(len(pos)*len(neg)))
te=[r for r in recs if r["i"]%2==1]; y=[r["hall"] for r in te]
print(f"\n[ib] test recs={len(te)} hall率={np.mean(y):.3f}")
print(f"[ib] self-verify AUROC = {auroc(y,[r['sv'] for r in te]):.3f}")
