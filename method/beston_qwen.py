#!/usr/bin/env python3
"""Qwen2.5-VL best-of-N 幻觉控制: 采样N个caption, 用 self-verify(0.94, Qwen上唯一强检测器)重排挑幻觉最少的。
验证"控制跨架构": 既然self-verify检测迁移到Qwen, 用它做best-of-N应也能降Qwen生成幻觉。
grounding在Qwen弱(0.59)故仅作参考, 主力重排=self-verify。支持 --bench chair|amber。"""
import argparse, importlib.util, json, os, sys
import torch, torch.nn.functional as F
if "--gpu" in sys.argv: os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[sys.argv.index("--gpu")+1]
os.environ.setdefault("AMBER_BASE_DIR","/volume/exploration/EvolvingLMMs/data/amber")
import numpy as np
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
def lm(p,n):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
cu=lm("/volume/exploration/EvolvingLMMs/lmms-eval/lmms_eval/tasks/coco_cap_chair/utils.py","cu")
ap=argparse.ArgumentParser()
ap.add_argument("--bench",default="chair",choices=["chair","amber"])
ap.add_argument("--N",type=int,default=6); ap.add_argument("--temp",type=float,default=0.8)
ap.add_argument("--n",type=int,default=250); ap.add_argument("--max_new_tokens",type=int,default=512)
ap.add_argument("--gpu",default="1"); ap.add_argument("--out",default="method/beston_qwen_out.jsonl")
ap.add_argument("--model",default="Qwen/Qwen2.5-VL-7B-Instruct")
a=ap.parse_args()
torch.set_grad_enabled(False)
proc=AutoProcessor.from_pretrained(a.model);tok=proc.tokenizer
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map="cuda:0",attn_implementation="eager").eval()
DEV="cuda:0"
YES=tok.encode("Yes",add_special_tokens=False)[0]; NO=tok.encode("No",add_special_tokens=False)[0]
yl_=tok.encode("yes",add_special_tokens=False)[0]; nl_=tok.encode("no",add_special_tokens=False)[0]
def build_inputs(image,text):
    msgs=[{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":text}]}]
    chat=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    imgs,vids=process_vision_info(msgs)
    return proc(text=[chat],images=imgs,videos=vids,return_tensors="pt",padding=True).to(DEV)
_svc={}
@torch.no_grad()
def selfverify_no(image,o):
    if o in _svc: return _svc[o]
    inp=build_inputs(image,f"Is there a {o} in the image? Answer the question using a single word or phrase.")
    lg=model(**inp).logits[0,-1,:].float()
    v=float(torch.logsumexp(lg[[NO,nl_]],0)-torch.logsumexp(lg[[YES,yl_]],0)); _svc[o]=v; return v
@torch.no_grad()
def gen(image,greedy=True,N=1):
    inp=build_inputs(image,"Describe this image.")
    if greedy:
        out=model.generate(**inp,max_new_tokens=a.max_new_tokens,do_sample=False,num_beams=1)
        return [tok.decode(out[0,inp.input_ids.shape[1]:],skip_special_tokens=True).strip()]
    out=model.generate(**inp,max_new_tokens=a.max_new_tokens,do_sample=True,temperature=a.temp,top_p=0.95,num_return_sequences=N)
    return [tok.decode(out[j,inp.input_ids.shape[1]:],skip_special_tokens=True).strip() for j in range(out.shape[0])]

if a.bench=="chair":
    ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
    idxs=[i for i in range(len(ds)) if i%2==1][:a.n]
    def get_img(i): return ds[i]["image"].convert("RGB")
    gt={i:set(ds[i]["gt_object"]) for i in idxs}
    def agg(sel):
        R=[{"answer":list(gt[i]),"pred":s} for i,s in zip(idxs,sel)]
        return (cu.coco_cap_chair_aggregate_results_chair_s(R),cu.coco_cap_chair_aggregate_results_chair_i(R),
                cu.coco_cap_chair_aggregate_results_recall(R))
    def true_hall(i,objs): return sum(1 for o in objs if o not in gt[i])
    HDR="CHAIR_s CHAIR_i recall"
else:
    AM=lm("/volume/exploration/EvolvingLMMs/lmms-eval/lmms_eval/tasks/amber_g/utils.py","am"); AM.load_metadata()
    ds=load_dataset("Kyunnilee/amber_g",split="train"); idxs=list(range(min(a.n,len(ds))))
    def get_img(i): return ds[i]["image"].convert("RGB")
    _docs={i:ds[i] for i in idxs}
    def prscore(i,cap): return AM.amber_g_process_result(_docs[i],[cap])
    def agg(sel):  # sel = list of process_result dicts
        return (AM.amber_g_aggregate_chair([p["amber_chair"] for p in sel]),
                AM.amber_g_aggregate_cover([p["amber_cover"] for p in sel]),
                AM.amber_g_aggregate_hal([p["amber_hal"] for p in sel]))
    HDR="CHAIR Cover Hal"

picks={"greedy":[],"minhall_SV":[],"recall_aware_SV":[],"ORACLE":[]}
fout=open(a.out,"w")
print(f"[qwen-beston] bench={a.bench} N={a.N} n={len(idxs)} model={a.model}",flush=True)
for c,i in enumerate(idxs):
    img=get_img(i); g=gen(img,greedy=True)[0]; caps=[g]+gen(img,greedy=False,N=a.N)
    _svc.clear()
    scored=[]
    for cap in caps:
        _,node,_,_=cu.caption_to_words(cap); objs=set(node)
        nh_sv=sum(selfverify_no(img,o)>0 for o in objs); no=len(objs)
        item={"cap":cap,"objs":list(objs),"nh_sv":nh_sv,"no":no}
        if a.bench=="amber": item["pr"]=prscore(i,cap); item["ah"]=item["pr"]["amber_chair"]["chair_score"]
        else: item["th"]=true_hall(i,objs)
        scored.append(item)
    def pack(item): return item["pr"] if a.bench=="amber" else item["objs"]
    picks["greedy"].append(pack(scored[0]))
    picks["minhall_SV"].append(pack(min(scored,key=lambda s:(s["nh_sv"],-s["no"]))))
    picks["recall_aware_SV"].append(pack(min(scored,key=lambda s:(s["nh_sv"]-0.5*s["no"]))))
    orc_key=(lambda s:(s["ah"],-s["no"])) if a.bench=="amber" else (lambda s:(s["th"],-s["no"]))
    picks["ORACLE"].append(pack(min(scored,key=orc_key)))
    fout.write(json.dumps({"i":i,"nh_sv":[s["nh_sv"] for s in scored]})+"\n"); fout.flush()
    if (c+1)%40==0: print(f"[qwen-beston] {c+1}/{len(idxs)}",flush=True)
fout.close()
print(f"\n[qwen-beston] bench={a.bench}  ({HDR})")
for k,sel in picks.items():
    m=agg(sel); print(f"[qwen-beston] {k:16} {m[0]:.2f} {m[1]:.2f} {m[2]:.2f}")
