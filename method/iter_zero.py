#!/usr/bin/env python3
"""迭代清除: detect(self-verify)->zero支撑->regen, 重复K轮(累积清零newly-detected的支撑)。
看能否逼近/微超单次oracle(-17.6), 即级联幻觉是否重要。"""
import argparse, importlib.util, json, os, sys
if "--gpu" in sys.argv: os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[sys.argv.index("--gpu")+1]
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
SYS=("A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.")
P_IMG=f"{SYS} USER: <image>\nPlease describe this image in detail. ASSISTANT:"
def CU():
    p="/volume/exploration/EvolvingLMMs/lmms-eval/lmms_eval/tasks/coco_cap_chair/utils.py"
    s=importlib.util.spec_from_file_location("cu",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ap=argparse.ArgumentParser()
ap.add_argument("--rk",type=int,default=20); ap.add_argument("--K",type=int,default=3); ap.add_argument("--n",type=int,default=250)
ap.add_argument("--max_new_tokens",type=int,default=512); ap.add_argument("--gpu",default="1"); ap.add_argument("--out",default="method/iter_out.jsonl")
ap.add_argument("--model",default="llava-hf/llava-1.5-7b-hf"); ap.add_argument("--cal",default="detect/calibration_tkb.json")
a=ap.parse_args(); cu=CU()
torch.set_grad_enabled(False)
proc=AutoProcessor.from_pretrained(a.model);tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained(a.model,torch_dtype=torch.float16,device_map="cuda:0",attn_implementation="eager").eval()
img_id=model.config.image_token_index;DEV="cuda:0";W=model.lm_head.weight;norm=model.model.language_model.norm
Llast=model.config.text_config.num_hidden_layers
cal=json.load(open(a.cal));cand=list(cal.keys())
objE={o:F.normalize(W[tok.encode(o,add_special_tokens=False)].float().mean(0),dim=-1).to(DEV) for o in cand}
YES=tok.encode("Yes",add_special_tokens=False)[0]; NO=tok.encode("No",add_special_tokens=False)[0]
yl_=tok.encode("yes",add_special_tokens=False)[0]; nl_=tok.encode("no",add_special_tokens=False)[0]
S={"sel":None}
def hook(m,i,o):
    if S["sel"] is None: return o
    f=o.view(-1,o.shape[-1]) if o.dim()==3 else o
    for k in S["sel"]:
        if k<f.shape[0]: f[k]=0
    return o
model.multi_modal_projector.register_forward_hook(hook)
_svc={}
@torch.no_grad()
def sv(image,o):
    if o in _svc: return _svc[o]
    pr=f"{SYS} USER: <image>\nIs there a {o} in the image?\nAnswer the question using a single word or phrase. ASSISTANT:"
    inp=proc(images=image,text=pr,return_tensors="pt").to(DEV,torch.float16); S["sel"]=None
    lg=model(**inp).logits[0,-1,:].float(); v=float(torch.logsumexp(lg[[NO,nl_]],0)-torch.logsumexp(lg[[YES,yl_]],0)); _svc[o]=v; return v
@torch.no_grad()
def prep(image):
    S["sel"]=None; vl=proc(images=image,text=P_IMG,return_tensors="pt").to(DEV,torch.float16)
    vis=(vl.input_ids[0]==img_id).nonzero(as_tuple=True)[0]
    o=model(**vl,output_hidden_states=True,use_cache=False)
    return vl,F.normalize(norm(o.hidden_states[Llast][0,vis,:]).float(),dim=-1)
@torch.no_grad()
def gen(vl):
    out=model.generate(**vl,max_new_tokens=a.max_new_tokens,do_sample=False,num_beams=1)
    return tok.decode(out[0,vl.input_ids.shape[1]:],skip_special_tokens=True).strip()
def sup(hn,o): return (hn@objE[o]).topk(min(a.rk,hn.shape[0])).indices.tolist()
ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
idxs=[i for i in range(len(ds)) if i%2==1][:a.n]
res={"baseline":[],"iter":[]}; fout=open(a.out,"w"); print(f"[iter] n={len(idxs)} K={a.K} rk={a.rk}",flush=True)
for c,i in enumerate(idxs):
    img=ds[i]["image"].convert("RGB"); vl,hn=prep(img); _svc.clear()
    S["sel"]=None; cap0=gen(vl); cur=cap0; sel=set()
    for r in range(a.K):
        _,node,_,_=cu.caption_to_words(cur); objs=[o for o in set(node) if o in objE]
        bad=[o for o in objs if sv(img,o)>0]
        new=set()
        for o in bad: new|=set(sup(hn,o))
        if new<=sel: break                 # 无新增, 收敛
        sel|=new; S["sel"]=sorted(sel); cur=gen(vl); S["sel"]=None
    res["baseline"].append({"answer":list(ds[i]["gt_object"]),"pred":cu.caption_to_words(cap0)[1]})
    res["iter"].append({"answer":list(ds[i]["gt_object"]),"pred":cu.caption_to_words(cur)[1]})
    fout.write(json.dumps({"i":i})+"\n")
    if (c+1)%40==0: print(f"[iter] {c+1}/{len(idxs)}",flush=True)
fout.close()
def agg(r): return (cu.coco_cap_chair_aggregate_results_chair_s(r),cu.coco_cap_chair_aggregate_results_chair_i(r),cu.coco_cap_chair_aggregate_results_recall(r))
b=agg(res["baseline"]); v=agg(res["iter"])
print(f"\n[iter] baseline {b[0]:.2f}/{b[1]:.2f}/{b[2]:.2f}")
print(f"[iter] iterative {v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f}  ΔCHAIR_s={v[0]-b[0]:+.2f}  (对比单次sv-gate -14.8, oracle -17.6)")
