#!/usr/bin/env python3
"""干预维度: soft-zero vs hard-ban vs zero+ban (同 self-verify flagging)。
zero: 清幻觉object支撑视觉token(软, 模型可能还说). ban: 生成时禁掉幻觉object的输出token(硬保证移除).
看 zero+ban 是否 > zero(推向 oracle −17.6), 及 ban 单独的 recall 代价。"""
import argparse, importlib.util, json, os, sys
from collections import defaultdict
if "--gpu" in sys.argv: os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[sys.argv.index("--gpu")+1]
import torch, torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
SYS=("A chat between a curious user and an artificial intelligence assistant. "
     "The assistant gives helpful, detailed, and polite answers to the user's questions.")
P_IMG=f"{SYS} USER: <image>\nPlease describe this image in detail. ASSISTANT:"
def CU():
    p="/volume/exploration/EvolvingLMMs/lmms-eval/lmms_eval/tasks/coco_cap_chair/utils.py"
    s=importlib.util.spec_from_file_location("cu",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ap=argparse.ArgumentParser()
ap.add_argument("--rk",type=int,default=20); ap.add_argument("--n",type=int,default=250); ap.add_argument("--max_new_tokens",type=int,default=512)
ap.add_argument("--gpu",default="1"); ap.add_argument("--out",default="method/ban_out.jsonl")
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
IRREG={"man":"men","woman":"women","person":"people","child":"children","foot":"feet","tooth":"teeth"}
canon2ft=defaultdict(set)
for w,canon in cu.INVERSE_SYNONYM_DICT.items():
    for s in [w,w+"s",w.rstrip("s"),w.capitalize()]+([IRREG[w],IRREG[w].capitalize()] if w in IRREG else []):
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
def prep(image):
    S["sel"]=None
    vl=proc(images=image,text=P_IMG,return_tensors="pt").to(DEV,torch.float16)
    vis=(vl.input_ids[0]==img_id).nonzero(as_tuple=True)[0]
    o=model(**vl,output_hidden_states=True,use_cache=False)
    hn=F.normalize(norm(o.hidden_states[Llast][0,vis,:]).float(),dim=-1)
    return vl,hn
@torch.no_grad()
def gen(vl,banned=None):
    kw={"bad_words_ids":[[t] for t in banned]} if banned else {}
    out=model.generate(**vl,max_new_tokens=a.max_new_tokens,do_sample=False,num_beams=1,**kw)
    return tok.decode(out[0,vl.input_ids.shape[1]:],skip_special_tokens=True).strip()
def support(hn,o,k): return (hn@objE[o]).topk(min(k,hn.shape[0])).indices.tolist()
ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
idxs=[i for i in range(len(ds)) if i%2==1][:a.n]
COND=["baseline","zero","ban","zero_ban"]; res={k:[] for k in COND}; nab=0
fout=open(a.out,"w"); print(f"[ban] n={len(idxs)} rk={a.rk}",flush=True)
for c,i in enumerate(idxs):
    img=ds[i]["image"].convert("RGB")
    vl,hn=prep(img); _svc.clear(); S["sel"]=None; cap=gen(vl)
    _,node,_,_=cu.caption_to_words(cap); objs=[o for o in set(node) if o in objE]
    bad=[o for o in objs if sv_no(img,o)>0]
    sel=set();
    for o in bad: sel|=set(support(hn,o,a.rk))
    sel=sorted(sel); banned=list({t for o in bad for t in canon2ft.get(o,set())})
    caps={"baseline":cap}
    if bad and sel:
        nab+=1
        S["sel"]=sel; caps["zero"]=gen(vl); S["sel"]=None
        S["sel"]=None; caps["ban"]=gen(vl,banned)
        S["sel"]=sel; caps["zero_ban"]=gen(vl,banned); S["sel"]=None
    else:
        for k in ["zero","ban","zero_ban"]: caps[k]=cap
    for k in COND:
        _,nv,_,_=cu.caption_to_words(caps[k]); res[k].append({"answer":list(ds[i]["gt_object"]),"pred":nv})
    fout.write(json.dumps({"i":i,"bad":bad})+"\n")
    if (c+1)%40==0: print(f"[ban] {c+1}/{len(idxs)} ablated={nab}",flush=True)
fout.close()
def agg(r): return (cu.coco_cap_chair_aggregate_results_chair_s(r),cu.coco_cap_chair_aggregate_results_chair_i(r),
                    cu.coco_cap_chair_aggregate_results_recall(r),sum(len(x['pred']) for x in r)/len(r))
b=agg(res["baseline"])
print(f"\n[ban] {'cond':10} CHAIR_s CHAIR_i recall objs  ΔCHAIR_s  (ablated={nab}, self-verify flag)")
for k in COND:
    m=agg(res[k]); print(f"[ban] {k:10} {m[0]:6.2f} {m[1]:6.2f} {m[2]:6.2f} {m[3]:5.2f}  {m[0]-b[0]:+.2f}")
