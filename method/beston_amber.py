#!/usr/bin/env python3
"""AMBER-generative 上的 best-of-N + 检测重排 (第二个生成benchmark, 验证CHAIR上的生成控制是否迁移)。
采样 N 个 caption -> 用 grounding(gc) + self-verify 重排挑幻觉最少的 -> 用 AMBER官方 scorer 打分。
报 greedy baseline vs 各重排 vs ORACLE(按AMBER真实hallucination挑)。AMBER指标: CHAIR(↓)/Cover(↑)/Hal(↓)。"""
import argparse, importlib.util, json, os, sys
import torch, torch.nn.functional as F
if "--gpu" in sys.argv: os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[sys.argv.index("--gpu")+1]
os.environ.setdefault("AMBER_BASE_DIR","/volume/exploration/EvolvingLMMs/data/amber")
import numpy as np
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
SYS=("A chat between a curious user and an artificial intelligence assistant. "
     "The assistant gives helpful, detailed, and polite answers to the user's questions.")
P_IMG=f"{SYS} USER: <image>\nDescribe this image. ASSISTANT:"   # AMBER 生成prompt
def load_mod(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
cu=load_mod("/volume/exploration/EvolvingLMMs/lmms-eval/lmms_eval/tasks/coco_cap_chair/utils.py","cu")
AM=load_mod("/volume/exploration/EvolvingLMMs/lmms-eval/lmms_eval/tasks/amber_g/utils.py","am")
ap=argparse.ArgumentParser()
ap.add_argument("--N",type=int,default=6); ap.add_argument("--temp",type=float,default=0.8)
ap.add_argument("--tau",type=float,default=-0.03)
ap.add_argument("--n",type=int,default=250); ap.add_argument("--max_new_tokens",type=int,default=512)
ap.add_argument("--gpu",default="1"); ap.add_argument("--out",default="method/beston_amber_out.jsonl")
ap.add_argument("--model",default="llava-hf/llava-1.5-7b-hf"); ap.add_argument("--cal",default="detect/calibration_tkb.json")
a=ap.parse_args()
torch.set_grad_enabled(False)
proc=AutoProcessor.from_pretrained(a.model);tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained(a.model,torch_dtype=torch.float16,device_map="cuda:0",attn_implementation="eager").eval()
img_id=model.config.image_token_index;DEV="cuda:0";W=model.lm_head.weight;norm=model.model.language_model.norm
Llast=model.config.text_config.num_hidden_layers
cal=json.load(open(a.cal));calmean=sum(cal.values())/len(cal)
YES=tok.encode("Yes",add_special_tokens=False)[0]; NO=tok.encode("No",add_special_tokens=False)[0]
yl_=tok.encode("yes",add_special_tokens=False)[0]; nl_=tok.encode("no",add_special_tokens=False)[0]
_svc={}
@torch.no_grad()
def selfverify_no(image,o):
    if o in _svc: return _svc[o]
    pr=f"{SYS} USER: <image>\nIs there a {o} in the image?\nAnswer the question using a single word or phrase. ASSISTANT:"
    inp=proc(images=image,text=pr,return_tensors="pt").to(DEV,torch.float16)
    lg=model(**inp).logits[0,-1,:].float()
    v=float(torch.logsumexp(lg[[NO,nl_]],0)-torch.logsumexp(lg[[YES,yl_]],0)); _svc[o]=v; return v
_e={}
def emb(o):
    if o not in _e:
        ids=tok.encode(o,add_special_tokens=False);_e[o]=F.normalize(W[ids].float().mean(0),dim=-1).to(DEV) if ids else None
    return _e[o]
@torch.no_grad()
def prep(image):
    vl=proc(images=image,text=P_IMG,return_tensors="pt").to(DEV,torch.float16)
    vis=(vl.input_ids[0]==img_id).nonzero(as_tuple=True)[0]
    o=model(**vl,output_hidden_states=True,use_cache=False)
    hn=F.normalize(norm(o.hidden_states[Llast][0,vis,:]).float(),dim=-1)
    return vl,hn
def gc_of(hn,o):
    e=emb(o)
    return ((hn@e).topk(min(10,hn.shape[0])).values.mean().item()-cal.get(o,calmean)) if e is not None else 1.0
@torch.no_grad()
def sample_caps(vl,greedy):
    caps=[greedy]
    out=model.generate(**vl,max_new_tokens=a.max_new_tokens,do_sample=True,temperature=a.temp,num_return_sequences=a.N,top_p=0.95)
    for j in range(out.shape[0]): caps.append(tok.decode(out[j,vl.input_ids.shape[1]:],skip_special_tokens=True).strip())
    return caps
@torch.no_grad()
def greedy_cap(vl):
    out=model.generate(**vl,max_new_tokens=a.max_new_tokens,do_sample=False,num_beams=1)
    return tok.decode(out[0,vl.input_ids.shape[1]:],skip_special_tokens=True).strip()

# AMBER scorer: 对一个 (doc, caption) 返回 per-metric dict; 聚合报 CHAIR/Cover/Hal
AM.load_metadata()
def amber_score_one(doc,cap):
    pr=AM.amber_g_process_result(doc,[cap])
    return pr
def aggregate(prs):
    ch=AM.amber_g_aggregate_chair([p["amber_chair"] for p in prs])
    co=AM.amber_g_aggregate_cover([p["amber_cover"] for p in prs])
    ha=AM.amber_g_aggregate_hal([p["amber_hal"] for p in prs])
    return ch,co,ha
def chair_num_of(pr):  # 该caption的AMBER幻觉分(用于oracle: 挑最小)
    m=pr["amber_chair"]; return m["chair_score"]

ds=load_dataset("Kyunnilee/amber_g",split="train",trust_remote_code=True)
N=min(a.n,len(ds))
picks={"greedy":[],"minhall":[],"recall_aware":[],"minhall_SV":[],"combo_gc+SV":[],"ORACLE":[]}
fout=open(a.out,"w")
print(f"[amber] N={a.N} n={N} model={a.model}",flush=True)
for c in range(N):
    doc=ds[c]; img=doc["image"].convert("RGB")
    vl,hn=prep(img); g=greedy_cap(vl); caps=sample_caps(vl,g)
    _svc.clear()
    scored=[]
    for cap in caps:
        _,node,_,_=cu.caption_to_words(cap); objs=set(node)
        nh=sum(gc_of(hn,o)<a.tau for o in objs)
        nh_sv=sum(selfverify_no(img,o)>0 for o in objs)
        no=len(objs)
        pr=amber_score_one(doc,cap)
        scored.append({"cap":cap,"nh":nh,"nh_sv":nh_sv,"no":no,"pr":pr,"amber_hall":chair_num_of(pr)})
    picks["greedy"].append(scored[0]["pr"])
    picks["minhall"].append(min(scored,key=lambda s:(s["nh"],-s["no"]))["pr"])
    picks["recall_aware"].append(min(scored,key=lambda s:(s["nh"]-0.5*s["no"]))["pr"])
    picks["minhall_SV"].append(min(scored,key=lambda s:(s["nh_sv"],-s["no"]))["pr"])
    picks["combo_gc+SV"].append(min(scored,key=lambda s:(s["nh"]+s["nh_sv"]-0.5*s["no"]))["pr"])
    picks["ORACLE"].append(min(scored,key=lambda s:(s["amber_hall"],-s["no"]))["pr"])
    fout.write(json.dumps({"i":c,"nh":[s["nh"] for s in scored],"amber_hall":[s["amber_hall"] for s in scored]})+"\n")
    if (c+1)%40==0: print(f"[amber] {c+1}/{N}",flush=True)
fout.close()
print()
for k,prs in picks.items():
    ch,co,ha=aggregate(prs); print(f"[amber] {k:14} CHAIR={ch:.2f} Cover={co:.2f} Hal={ha:.2f}")
