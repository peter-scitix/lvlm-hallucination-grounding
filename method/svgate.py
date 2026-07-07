#!/usr/bin/env python3
"""svgate: 单次前向"信念→范数"门控 (make-or-break)。
思路: 用模型自己的 self-verify 判别信念(强, 0.94)决定抑制哪些 object, 用 NORM(唯一未被占的因果杠杆)做抑制,
grounding 仅作内部定位工具(找 object 的支撑视觉 token)。对幻觉 object 的支撑 token 缩范数(scale→0), 重生成一次。
同一杠杆下头对头比 4 个决策信号, 精确定位命门:
  baseline | gd-gate(grounding决策) | sv-gate(self-verify决策=新方法) | oracle-gate(GT决策=机制天花板)
若 oracle 强而 sv/gd 弱 => 瓶颈是决策信号; 若 oracle 也弱 => 范数杠杆不够(回退A)。"""
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
ap.add_argument("--tau",type=float,default=-0.06,help="grounding决策阈值 gc<tau=>幻觉")
ap.add_argument("--b",type=float,default=0.0,help="self-verify决策阈值 margin>b=>模型说不在")
ap.add_argument("--rk",type=int,default=20); ap.add_argument("--scale",type=float,default=0.0)
ap.add_argument("--mode",default="scale",choices=["scale","project"],help="scale=缩范数(因果杠杆); project=去object语义方向(保范数, 解离对照)")
ap.add_argument("--loc",default="grounding",choices=["grounding","attention","random"],help="定位消融: 用什么找支撑token(grounding=logit-lens / attention=竞品轴 / random=对照)")
ap.add_argument("--n",type=int,default=250); ap.add_argument("--max_new_tokens",type=int,default=512)
ap.add_argument("--gpu",default="1"); ap.add_argument("--out",default="method/svgate_out.jsonl")
ap.add_argument("--model",default="llava-hf/llava-1.5-7b-hf"); ap.add_argument("--cal",default="detect/calibration_tkb.json")
a=ap.parse_args(); cu=CU()
torch.set_grad_enabled(False)
proc=AutoProcessor.from_pretrained(a.model);tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained(a.model,torch_dtype=torch.float16,device_map="cuda:0",attn_implementation="eager").eval()
img_id=model.config.image_token_index;DEV="cuda:0"
W=model.lm_head.weight;norm=model.model.language_model.norm
Llast=model.config.text_config.num_hidden_layers
cal=json.load(open(a.cal));cand=list(cal.keys());calmean=sum(cal.values())/len(cal)
objE={o:F.normalize(W[tok.encode(o,add_special_tokens=False)].float().mean(0),dim=-1).to(DEV) for o in cand}
YES=tok.encode("Yes",add_special_tokens=False)[0]; NO=tok.encode("No",add_special_tokens=False)[0]
yl_=tok.encode("yes",add_special_tokens=False)[0]; nl_=tok.encode("no",add_special_tokens=False)[0]
# projector hook: scale=缩SEL行范数 / project=去其object语义方向(保范数)
S={"sel":None,"scale":0.0,"mode":"scale","dirs":{}}
def proj_hook(mod,inp,out):
    if S["sel"] is None: return out
    flat=out.view(-1,out.shape[-1]) if out.dim()==3 else out
    for i in S["sel"]:
        if i>=flat.shape[0]: continue
        if S["mode"]=="project":
            d=S["dirs"].get(i)
            if d is not None:
                dd=d.to(flat.dtype); flat[i]=flat[i]-(flat[i]@dd)*dd
        else: flat[i]=flat[i]*S["scale"]
    return out
model.multi_modal_projector.register_forward_hook(proj_hook)
_svc={}
@torch.no_grad()
def selfverify_no(image,o):
    if o in _svc: return _svc[o]
    pr=f"{SYS} USER: <image>\nIs there a {o} in the image?\nAnswer the question using a single word or phrase. ASSISTANT:"
    inp=proc(images=image,text=pr,return_tensors="pt").to(DEV,torch.float16)
    S["sel"]=None; lg=model(**inp).logits[0,-1,:].float()
    v=float(torch.logsumexp(lg[[NO,nl_]],0)-torch.logsumexp(lg[[YES,yl_]],0)); _svc[o]=v; return v
@torch.no_grad()
def prep(image):
    S["sel"]=None
    vl=proc(images=image,text=P_IMG,return_tensors="pt").to(DEV,torch.float16)
    vis=(vl.input_ids[0]==img_id).nonzero(as_tuple=True)[0]
    o=model(**vl,output_hidden_states=True,use_cache=False)
    hn=F.normalize(norm(o.hidden_states[Llast][0,vis,:]).float(),dim=-1)
    return vl,hn,len(vis)
@torch.no_grad()
def gen(vl):
    out=model.generate(**vl,max_new_tokens=a.max_new_tokens,do_sample=False,num_beams=1)
    return tok.decode(out[0,vl.input_ids.shape[1]:],skip_special_tokens=True).strip()
def gc_of(hn,o): return (hn@objE[o]).topk(min(10,hn.shape[0])).values.mean().item()-cal.get(o,calmean)
_rng=None
@torch.no_grad()
def sv_attn(image,o,P):
    """attention定位(竞品轴): 问'is there o'时, 末token对图像token的注意力(深层平均)topk patch。"""
    pr=f"{SYS} USER: <image>\nIs there a {o} in the image?\nAnswer the question using a single word or phrase. ASSISTANT:"
    inp=proc(images=image,text=pr,return_tensors="pt").to(DEV,torch.float16); S["sel"]=None
    out=model(**inp,output_attentions=True,use_cache=False)
    vis=(inp.input_ids[0]==img_id).nonzero(as_tuple=True)[0]
    att=torch.stack([out.attentions[l][0,:,-1,:][:,vis].mean(0) for l in range(20,Llast)]).mean(0)  # (P,)
    return att.topk(min(a.rk,att.shape[0])).indices.tolist()
def sel_tokens(hn,o,image,P):
    if a.loc=="grounding": return (hn@objE[o]).topk(min(a.rk,hn.shape[0])).indices.tolist()
    if a.loc=="attention": return sv_attn(image,o,P)
    import random as _r; _r.seed(hash(o)&0xffff)
    return _r.sample(range(P),min(a.rk,P))
@torch.no_grad()
def regen(vl,hn,bad,image=None,P=576):
    """对 bad object 的支撑 token 做扰动(scale缩范数 / project去方向), 重生成一次。定位方式由 --loc 决定。"""
    if not bad: return None
    sel=set(); dirs={}
    for o in bad:
        if o in objE:
            toks=sel_tokens(hn,o,image,P); sel|=set(toks)
            if a.mode=="project":
                for t in toks: dirs[t]=objE[o]
    if not sel: return None
    S["sel"]=sorted(sel); S["scale"]=a.scale; S["mode"]=a.mode; S["dirs"]=dirs
    cap=gen(vl); S["sel"]=None; return cap

ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
idxs=[i for i in range(len(ds)) if i%2==1][:a.n]
COND=["baseline","gd_gate","sv_gate","oracle_gate"]
res={k:[] for k in COND}; nab={k:0 for k in COND}
fout=open(a.out,"w")
print(f"[svgate] n={len(idxs)} tau={a.tau} b={a.b} rk={a.rk} scale={a.scale} model={a.model}",flush=True)
for c,i in enumerate(idxs):
    img=ds[i]["image"].convert("RGB"); gt=set(ds[i]["gt_object"])
    vl,hn,P=prep(img); cap=gen(vl); _svc.clear()
    _,node,_,_=cu.caption_to_words(cap); objs=[o for o in set(node) if o in objE]
    # 三种决策信号 flag 幻觉集(都用 grounding 定位支撑token, 只是"决定抑制谁"的信号不同)
    bad_gd=[o for o in objs if gc_of(hn,o)<a.tau]
    bad_sv=[o for o in objs if selfverify_no(img,o)>a.b]
    bad_or=[o for o in objs if o not in gt]
    caps={"baseline":cap}
    for k,bad in [("gd_gate",bad_gd),("sv_gate",bad_sv),("oracle_gate",bad_or)]:
        nc=regen(vl,hn,bad,img,P)
        if nc is not None: nab[k]+=1
        caps[k]=nc if nc is not None else cap
    for k in COND:
        _,nv,_,_=cu.caption_to_words(caps[k])
        res[k].append({"answer":list(gt),"pred":nv})
    fout.write(json.dumps({"i":i,"bad_gd":bad_gd,"bad_sv":bad_sv,"bad_or":bad_or})+"\n")
    if (c+1)%40==0: print(f"[svgate] {c+1}/{len(idxs)} ablated gd/sv/or={nab['gd_gate']}/{nab['sv_gate']}/{nab['oracle_gate']}",flush=True)
fout.close()
def agg(r): return (cu.coco_cap_chair_aggregate_results_chair_s(r),cu.coco_cap_chair_aggregate_results_chair_i(r),
                    cu.coco_cap_chair_aggregate_results_recall(r),sum(len(x['pred']) for x in r)/len(r))
base=agg(res["baseline"])
print(f"\n[svgate] {'cond':12} CHAIR_s CHAIR_i recall  objs  ablated  ΔCHAIR_s")
for k in COND:
    m=agg(res[k]); print(f"[svgate] {k:12} {m[0]:6.2f} {m[1]:6.2f} {m[2]:6.2f} {m[3]:5.2f}  {nab[k]:5d}   {m[0]-base[0]:+.2f}")
