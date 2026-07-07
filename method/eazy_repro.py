#!/usr/bin/env python3
"""忠实复现 EAZY 的定位, 与我们的 grounding 定位头对头(是不是改进)。
EAZY 定位: caption 生成时, 幻觉 object 的 token 在 **layer 14** 对图像token的 attention, 取 top-k patch, 清零。
我们定位: logit-lens grounding (hn@objE[o]) top-k patch, 清零。
控制: 同一批 flagged object(self-verify)、同一 zero 干预、同 rk —— 只变"定位方式":
  baseline / eazy_loc(attn-L14) / grounding_loc / random_loc
若 grounding_loc 的 CHAIR 明显 < eazy_loc => 我们的定位确实是对 EAZY 的改进。
(注: EAZY 原实现清的是像素patch并重过vision encoder; 这里两种定位都清 projector 输出, 以隔离'定位'单一变量。)"""
import argparse, importlib.util, json, os, sys
from collections import defaultdict
if "--gpu" in sys.argv: os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[sys.argv.index("--gpu")+1]
import torch, torch.nn.functional as F
import numpy as np, random
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
SYS=("A chat between a curious user and an artificial intelligence assistant. "
     "The assistant gives helpful, detailed, and polite answers to the user's questions.")
P_IMG=f"{SYS} USER: <image>\nPlease describe this image in detail. ASSISTANT:"
def CU():
    p="/volume/exploration/EvolvingLMMs/lmms-eval/lmms_eval/tasks/coco_cap_chair/utils.py"
    s=importlib.util.spec_from_file_location("cu",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ap=argparse.ArgumentParser()
ap.add_argument("--rk",type=int,default=20); ap.add_argument("--eazy_layer",type=int,default=14)
ap.add_argument("--n",type=int,default=250); ap.add_argument("--max_new_tokens",type=int,default=512)
ap.add_argument("--gpu",default="1"); ap.add_argument("--out",default="method/eazy_repro_out.jsonl")
ap.add_argument("--model",default="llava-hf/llava-1.5-7b-hf"); ap.add_argument("--cal",default="detect/calibration_tkb.json")
a=ap.parse_args(); cu=CU()
torch.set_grad_enabled(False)
proc=AutoProcessor.from_pretrained(a.model);tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained(a.model,torch_dtype=torch.float16,device_map="cuda:0",attn_implementation="eager").eval()
img_id=model.config.image_token_index;DEV="cuda:0";W=model.lm_head.weight;norm=model.model.language_model.norm
Llast=model.config.text_config.num_hidden_layers
cal=json.load(open(a.cal));cand=list(cal.keys());calmean=sum(cal.values())/len(cal)
objE={o:F.normalize(W[tok.encode(o,add_special_tokens=False)].float().mean(0),dim=-1).to(DEV) for o in cand}
YES=tok.encode("Yes",add_special_tokens=False)[0]; NO=tok.encode("No",add_special_tokens=False)[0]
yl_=tok.encode("yes",add_special_tokens=False)[0]; nl_=tok.encode("no",add_special_tokens=False)[0]
IRREG={"man":"men","woman":"women","person":"people","child":"children"}
canon2ft=defaultdict(set)
for w,canon in cu.INVERSE_SYNONYM_DICT.items():
    for s in [w,w+"s",w.rstrip("s")," "+w," "+w+"s"]+([" "+IRREG[w]] if w in IRREG else []):
        ids=tok.encode(s,add_special_tokens=False)
        if ids: canon2ft[canon].add(ids[0])
S={"sel":None}
def hook(mod,inp,out):
    if S["sel"] is None: return out
    flat=out.view(-1,out.shape[-1]) if out.dim()==3 else out
    for i in S["sel"]:
        if i<flat.shape[0]: flat[i]=0
    return out
model.multi_modal_projector.register_forward_hook(hook)
_svc={}
@torch.no_grad()
def sv_no(image,o):
    if o in _svc: return _svc[o]
    pr=f"{SYS} USER: <image>\nIs there a {o} in the image?\nAnswer the question using a single word or phrase. ASSISTANT:"
    inp=proc(images=image,text=pr,return_tensors="pt").to(DEV,torch.float16); S["sel"]=None
    lg=model(**inp).logits[0,-1,:].float()
    v=float(torch.logsumexp(lg[[NO,nl_]],0)-torch.logsumexp(lg[[YES,yl_]],0)); _svc[o]=v; return v
@torch.no_grad()
def gen(vl):   # 不在此 reset S["sel"]; 由调用方控制(baseline前置None, 干预前置sel)
    out=model.generate(**vl,max_new_tokens=a.max_new_tokens,do_sample=False,num_beams=1)
    return out,tok.decode(out[0,vl.input_ids.shape[1]:],skip_special_tokens=True).strip()
@torch.no_grad()
def prep_ground(vl):
    S["sel"]=None
    vis=(vl.input_ids[0]==img_id).nonzero(as_tuple=True)[0]
    o=model(**vl,output_hidden_states=True,use_cache=False)
    hn=F.normalize(norm(o.hidden_states[Llast][0,vis,:]).float(),dim=-1)
    return hn,vis
def gc_of(hn,o): return (hn@objE[o]).topk(min(10,hn.shape[0])).values.mean().item()-cal.get(o,calmean)
def ground_loc(hn,o): return (hn@objE[o]).topk(min(a.rk,hn.shape[0])).indices.tolist()
@torch.no_grad()
def eazy_loc(seq,vis,bad):
    """EAZY: 全序列前向取 attentions; 对每个bad object的token位置, layer eazy_layer 对图像token的attn top-rk。"""
    o=model(input_ids=seq,pixel_values=PIX,output_attentions=True,use_cache=False)
    att=o.attentions[a.eazy_layer][0]  # (heads,T,T)
    gen_ids=seq[0,PLEN:]; ngen=gen_ids.shape[0]; T=seq.shape[1]; base=T-ngen
    vis_set=vis.tolist(); sel=set()
    for ob in bad:
        fts=canon2ft.get(ob,set())
        for j in range(ngen):
            if int(gen_ids[j]) in fts:
                pos=base+j
                a2img=att[:,pos,:][:,vis].mean(0)   # (P,) 该token对图像token的注意力(head平均)
                for idx in a2img.topk(min(a.rk,a2img.shape[0])).indices.tolist(): sel.add(idx)
                break
    return sorted(sel)
ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
idxs=[i for i in range(len(ds)) if i%2==1][:a.n]
COND=["baseline","eazy_loc","grounding_loc","random_loc"]; res={k:[] for k in COND}; nab=0
fout=open(a.out,"w"); print(f"[eazy] n={len(idxs)} rk={a.rk} eazy_layer={a.eazy_layer}",flush=True)
for c,i in enumerate(idxs):
    img=ds[i]["image"].convert("RGB")
    vl=proc(images=img,text=P_IMG,return_tensors="pt").to(DEV,torch.float16)
    global PIX,PLEN; PIX=vl.pixel_values; PLEN=vl.input_ids.shape[1]
    S["sel"]=None; seq,cap=gen(vl); _svc.clear()
    hn,vis=prep_ground(vl)
    _,node,_,_=cu.caption_to_words(cap); objs=[o for o in set(node) if o in objE]
    bad=[o for o in objs if sv_no(img,o)>0]
    caps={"baseline":cap}
    if bad:
        nab+=1
        gl=set()
        for o in bad: gl|=set(ground_loc(hn,o))
        el=eazy_loc(seq,vis,bad)
        rng=random.Random(i); rl=rng.sample(range(len(vis)),min(len(gl),len(vis)))
        for name,sel in [("eazy_loc",el),("grounding_loc",sorted(gl)),("random_loc",rl)]:
            S["sel"]=sel; _,nc=gen(vl); S["sel"]=None; caps[name]=nc
    else:
        for name in ["eazy_loc","grounding_loc","random_loc"]: caps[name]=cap
    for k in COND:
        _,nv,_,_=cu.caption_to_words(caps[k]); res[k].append({"answer":list(ds[i]["gt_object"]),"pred":nv})
    fout.write(json.dumps({"i":i,"bad":bad})+"\n")
    if (c+1)%40==0: print(f"[eazy] {c+1}/{len(idxs)} ablated={nab}",flush=True)
fout.close()
def agg(r): return (cu.coco_cap_chair_aggregate_results_chair_s(r),cu.coco_cap_chair_aggregate_results_chair_i(r),
                    cu.coco_cap_chair_aggregate_results_recall(r),sum(len(x['pred']) for x in r)/len(r))
b=agg(res["baseline"])
print(f"\n[eazy] {'cond':14} CHAIR_s CHAIR_i recall objs  ΔCHAIR_s  (ablated={nab}, 同self-verify flag+同zero, 只变定位)")
for k in COND:
    m=agg(res[k]); print(f"[eazy] {k:14} {m[0]:6.2f} {m[1]:6.2f} {m[2]:6.2f} {m[3]:5.2f}  {m[0]-b[0]:+.2f}")
