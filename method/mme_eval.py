#!/usr/bin/env python3
"""MME baseline (llava-hf). yes/no VQA, 复用 mme 官方 scorer 报 Perception/Cognition 总分。
MME 是判别任务(我们的生成控制不适用), 此处建立 baseline 以覆盖该 benchmark。"""
import argparse, importlib.util, json, os, sys
if "--gpu" in sys.argv: os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[sys.argv.index("--gpu")+1]
import torch
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
SYS=("A chat between a curious user and an artificial intelligence assistant. "
     "The assistant gives helpful, detailed, and polite answers to the user's questions.")
def lm(p,n):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
MM=lm("/volume/exploration/EvolvingLMMs/lmms-eval/lmms_eval/tasks/mme/utils.py","mm")
ap=argparse.ArgumentParser()
ap.add_argument("--n",type=int,default=0,help="0=all"); ap.add_argument("--gpu",default="1")
ap.add_argument("--model",default="llava-hf/llava-1.5-7b-hf"); ap.add_argument("--out",default="method/mme_out.json")
a=ap.parse_args()
torch.set_grad_enabled(False)
proc=AutoProcessor.from_pretrained(a.model);tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained(a.model,torch_dtype=torch.float16,device_map="cuda:0",attn_implementation="eager").eval()
DEV="cuda:0"
ds=load_dataset("lmms-lab/MME",split="test")
N=len(ds) if a.n==0 else min(a.n,len(ds))
per=[]; cog=[]
print(f"[mme] n={N} model={a.model}",flush=True)
@torch.no_grad()
def answer(image,q):
    pr=f"{SYS} USER: <image>\n{q} ASSISTANT:"
    inp=proc(images=image,text=pr,return_tensors="pt").to(DEV,torch.float16)
    out=model.generate(**inp,max_new_tokens=16,do_sample=False,num_beams=1)
    return tok.decode(out[0,inp.input_ids.shape[1]:],skip_special_tokens=True).strip()
for c in range(N):
    doc=ds[c]; q=doc["question"].strip()
    ans=answer(doc["image"].convert("RGB"),q)
    r=MM.mme_process_results(doc,[ans])
    if "mme_perception_score" in r: per.append(r["mme_perception_score"])
    else: cog.append(r["mme_cognition_score"])
    if (c+1)%200==0: print(f"[mme] {c+1}/{N}",flush=True)
P=MM.mme_aggregate_results(per) if per else 0.0
C=MM.mme_aggregate_results(cog) if cog else 0.0
json.dump({"model":a.model,"perception":P,"cognition":C,"n":N},open(a.out,"w"))
print(f"\n[mme] Perception={P:.2f}  Cognition={C:.2f}  Total={P+C:.2f}  (n={N})")
