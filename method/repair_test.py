#!/usr/bin/env python3
"""图内视觉token修复 vs EAZY清零 头对头 (物体幻觉消除)。
同一检测(grounding gc<tau, 你的cos-sim信号)+ 同一批支撑token(logit-lens retrieval), 只换干预方式:
  zero  : 支撑token->0            (EAZY式)
  mean  : ->所有视觉token均值     (vtablate mean)
  repair: ->本图"高置信token"质心  (intra-image, 用真实内容重构松散/弱的幻觉支撑; 非steering)
高置信token = 对任一candidate object 的 max cos-sim 最高的 top-M(紧致强属于=真实内容锚)。
报 baseline/zero/mean/repair 的 CHAIR_s/i/recall。看 repair 是否比 zero 更低CHAIR+更保recall。"""
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
ap.add_argument("--tau",type=float,default=-0.06); ap.add_argument("--rk",type=int,default=20)
ap.add_argument("--M",type=int,default=40,help="高置信锚token数")
ap.add_argument("--n",type=int,default=250); ap.add_argument("--max_new_tokens",type=int,default=512)
ap.add_argument("--gpu",default="1"); ap.add_argument("--out",default="method/repair_out.jsonl")
ap.add_argument("--model",default="llava-hf/llava-1.5-7b-hf"); ap.add_argument("--cal",default="detect/calibration_tkb.json")
a=ap.parse_args(); cu=CU()
torch.set_grad_enabled(False)
proc=AutoProcessor.from_pretrained(a.model);tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained(a.model,torch_dtype=torch.float16,device_map="cuda:0",attn_implementation="eager").eval()
img_id=model.config.image_token_index;DEV="cuda:0";W=model.lm_head.weight;norm=model.model.language_model.norm
Llast=model.config.text_config.num_hidden_layers
cal=json.load(open(a.cal));cand=list(cal.keys());calmean=sum(cal.values())/len(cal)
objE={o:F.normalize(W[tok.encode(o,add_special_tokens=False)].float().mean(0),dim=-1).to(DEV) for o in cand}
OBJMAT=torch.stack([objE[o] for o in cand])  # (C,H_norm) 用于 max cos
YES=tok.encode("Yes",add_special_tokens=False)[0]; NO=tok.encode("No",add_special_tokens=False)[0]
yl_=tok.encode("yes",add_special_tokens=False)[0]; nl_=tok.encode("no",add_special_tokens=False)[0]
_svc={}
@torch.no_grad()
def sv_no(image,o):  # self-verify: >0 => 模型说不在(强检测器, flag多, 隔离干预对比)
    if o in _svc: return _svc[o]
    pr=f"{SYS} USER: <image>\nIs there a {o} in the image?\nAnswer the question using a single word or phrase. ASSISTANT:"
    inp=proc(images=image,text=pr,return_tensors="pt").to(DEV,torch.float16); S["mode"]=None; S["sel"]=None
    lg=model(**inp).logits[0,-1,:].float()
    v=float(torch.logsumexp(lg[[NO,nl_]],0)-torch.logsumexp(lg[[YES,yl_]],0)); _svc[o]=v; return v
# hook: capture / zero / mean / repair
S={"mode":None,"sel":None,"anchor":None,"proj":None}
def proj_hook(mod,inp,out):
    if S["mode"]=="capture":
        S["proj"]=out.detach().clone().view(-1,out.shape[-1]); return out
    if S["mode"] is None or S["sel"] is None: return out
    flat=out.view(-1,out.shape[-1]) if out.dim()==3 else out
    for i in S["sel"]:
        if i>=flat.shape[0]: continue
        if S["mode"]=="zero": flat[i]=0
        elif S["mode"]=="mean": flat[i]=flat.mean(0)
        elif S["mode"]=="repair": flat[i]=S["anchor"].to(flat.dtype)
    return out
model.multi_modal_projector.register_forward_hook(proj_hook)
@torch.no_grad()
def prep(image):
    S["mode"]="capture"; S["sel"]=None
    vl=proc(images=image,text=P_IMG,return_tensors="pt").to(DEV,torch.float16)
    vis=(vl.input_ids[0]==img_id).nonzero(as_tuple=True)[0]
    o=model(**vl,output_hidden_states=True,use_cache=False)
    S["mode"]=None
    hn=F.normalize(norm(o.hidden_states[Llast][0,vis,:]).float(),dim=-1)  # (P,H)
    # 高置信锚: 每token对任一object的max cos, 取top-M, 在projector输出空间取质心
    maxg=(hn@OBJMAT.T).max(1).values                     # (P,) 该token最强属于某object的程度
    conf=maxg.topk(min(a.M,hn.shape[0])).indices.tolist()
    proj=S["proj"]                                        # (P,H) projector输出
    anchor=proj[conf].mean(0)
    return vl,hn,anchor
@torch.no_grad()
def gen(vl):
    out=model.generate(**vl,max_new_tokens=a.max_new_tokens,do_sample=False,num_beams=1)
    return tok.decode(out[0,vl.input_ids.shape[1]:],skip_special_tokens=True).strip()
def gc_of(hn,o): return (hn@objE[o]).topk(min(10,hn.shape[0])).values.mean().item()-cal.get(o,calmean)
def support(hn,o,k): return (hn@objE[o]).topk(min(k,hn.shape[0])).indices.tolist()
ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
idxs=[i for i in range(len(ds)) if i%2==1][:a.n]
COND=["baseline","zero","mean","repair"]; res={k:[] for k in COND}; nab=0
fout=open(a.out,"w"); print(f"[repair] n={len(idxs)} tau={a.tau} rk={a.rk} M={a.M}",flush=True)
for c,i in enumerate(idxs):
    img=ds[i]["image"].convert("RGB"); gt=list(ds[i]["gt_object"])
    vl,hn,anchor=prep(img); S["anchor"]=anchor; _svc.clear()
    S["mode"]=None; S["sel"]=None; cap=gen(vl)
    _,node,_,_=cu.caption_to_words(cap); objs=[o for o in set(node) if o in objE]
    bad=[o for o in objs if sv_no(img,o)>0]   # self-verify flagging(强, flag多)
    sel=set()
    for o in bad: sel|=set(support(hn,o,a.rk))
    sel=sorted(sel)
    caps={"baseline":cap}
    if bad and sel:
        nab+=1
        for mode in ["zero","mean","repair"]:
            S["mode"]=mode; S["sel"]=sel; caps[mode]=gen(vl); S["mode"]=None; S["sel"]=None
    else:
        for mode in ["zero","mean","repair"]: caps[mode]=cap
    for k in COND:
        _,nv,_,_=cu.caption_to_words(caps[k]); res[k].append({"answer":gt,"pred":nv})
    fout.write(json.dumps({"i":i,"bad":bad,"nsel":len(sel)})+"\n")
    if (c+1)%40==0: print(f"[repair] {c+1}/{len(idxs)} ablated={nab}",flush=True)
fout.close()
def agg(r): return (cu.coco_cap_chair_aggregate_results_chair_s(r),cu.coco_cap_chair_aggregate_results_chair_i(r),
                    cu.coco_cap_chair_aggregate_results_recall(r),sum(len(x['pred']) for x in r)/len(r))
b=agg(res["baseline"])
print(f"\n[repair] {'cond':10} CHAIR_s CHAIR_i recall objs  ΔCHAIR_s   (ablated={nab})")
for k in COND:
    m=agg(res[k]); print(f"[repair] {k:10} {m[0]:6.2f} {m[1]:6.2f} {m[2]:6.2f} {m[3]:5.2f}  {m[0]-b[0]:+.2f}")
