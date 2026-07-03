#!/usr/bin/env python3
"""视觉侧消除(用户方向, EAZY-adjacent): 检测不变(grounding gc), 但消除在 VISUAL TOKEN 上做 ——
对检测到的幻觉object o, 用 logit-lens object->visual-token retrieval 找支撑它的 top-k visual token
(cos(LN(h_vis[p]), W_U[o]) 最高的那些, 即"声称自己是o"的patch), 把它们在 multi_modal_projector 输出里清零/均值化,
再重生成 caption。好处: 编辑输入侧=>模型自然重生成=>caption通顺; 只中和幻觉的支撑patch=>真实object不受影响。
检测=grounding+base-rate protect(不变)。对比 excise+repair / 输出侧方法。"""
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
ap.add_argument("--tau",type=float,default=-0.06); ap.add_argument("--rk",type=int,default=20,help="每个幻觉object中和的visual token数")
ap.add_argument("--mode",default="zero",choices=["zero","mean","project","scale"])
ap.add_argument("--scale",type=float,default=0.0,help="scale模式: 支撑token乘以该系数")
ap.add_argument("--ban",action="store_true",help="重生成时同时ban幻觉object输出token(视觉中和保通顺+ban保证移除)")
ap.add_argument("--protect",default="person,tennis racket,surfboard,sports ball,elephant,umbrella")
ap.add_argument("--split",default="test"); ap.add_argument("--n",type=int,default=250); ap.add_argument("--max_new_tokens",type=int,default=512)
ap.add_argument("--gpu",default="1"); ap.add_argument("--out",default="method/vt_out.jsonl"); ap.add_argument("--show",type=int,default=4)
ap.add_argument("--model",default="llava-hf/llava-1.5-7b-hf")
ap.add_argument("--cal",default="detect/calibration_tkb.json")
ap.add_argument("--random",action="store_true",help="范数特异性对照: 同数量随机patch做同样扰动(非幻觉支撑token)")
a=ap.parse_args(); cu=CU()
PROTECT=set(x.strip() for x in a.protect.split(",") if x.strip())
torch.set_grad_enabled(False)
proc=AutoProcessor.from_pretrained(a.model);tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained(a.model,torch_dtype=torch.float16,device_map="cuda:0",attn_implementation="eager").eval()
img_id=model.config.image_token_index;eos=tok.eos_token_id;DEV="cuda:0"
W=model.lm_head.weight;norm=model.model.language_model.norm
Llast=model.config.text_config.num_hidden_layers  # last-layer hidden_states index (7B=32, 13B=40)
cal=json.load(open(a.cal));cand=list(cal.keys());calmean=sum(cal.values())/len(cal)
objE={o:F.normalize(W[tok.encode(o,add_special_tokens=False)].float().mean(0),dim=-1).to(DEV) for o in cand}
IRREG={"man":"men","woman":"women","person":"people","child":"children","foot":"feet","tooth":"teeth"}
canon2ftoks=defaultdict(set)
for w,canon in cu.INVERSE_SYNONYM_DICT.items():
    for s in [w,w+"s",w.rstrip("s"),w.capitalize()]+([IRREG[w],IRREG[w].capitalize()] if w in IRREG else []):
        ids=tok.encode(s,add_special_tokens=False)
        if ids: canon2ftoks[canon].add(ids[0])
# projector hook: 把 SEL 行清零/均值
S={"sel":None,"mode":"zero","dirs":{}}
def proj_hook(mod,inp,out):
    if S["sel"] is None: return out
    o=out
    flat=o.view(-1,o.shape[-1]) if o.dim()==3 else o   # (P,H) or (B,P,H)->(B*P,H) 单图P=576
    idx=[i for i in S["sel"] if i<flat.shape[0]]
    if idx:
        if S["mode"]=="mean": flat[idx]=flat.mean(0,keepdim=True)
        elif S["mode"]=="scale":
            for i in idx: flat[i]=flat[i]*S.get("scale",0.0)
        elif S["mode"]=="project":   # 沿object语义方向投影掉(只去"指向o"的分量)
            for i in idx:
                d=S["dirs"].get(i)
                if d is not None:
                    dd=d.to(flat.dtype); flat[i]=flat[i]-(flat[i]@dd)*dd
        else: flat[idx]=0
    return o
model.multi_modal_projector.register_forward_hook(proj_hook)

@torch.no_grad()
def prep(image):
    vl=proc(images=image,text=P_IMG,return_tensors="pt").to(DEV,torch.float16)
    vis=(vl.input_ids[0]==img_id).nonzero(as_tuple=True)[0]
    o=model(**vl,output_hidden_states=True,use_cache=False)
    hn=F.normalize(norm(o.hidden_states[Llast][0,vis,:]).float(),dim=-1)  # (576,d)
    return vl,hn,len(vis)
@torch.no_grad()
def gen(vl,banned=None):
    kw={}
    if banned: kw["bad_words_ids"]=[[t] for t in banned]
    out=model.generate(**vl,max_new_tokens=a.max_new_tokens,do_sample=False,num_beams=1,**kw)
    return tok.decode(out[0,vl.input_ids.shape[1]:],skip_special_tokens=True).strip()
def gc_of(hn,o): return (hn@objE[o]).topk(min(10,hn.shape[0])).values.mean().item()-cal.get(o,calmean)
def support_tokens(hn,o,k):
    return (hn@objE[o]).topk(min(k,hn.shape[0])).indices.tolist()  # 最"声称是o"的patch
@torch.no_grad()
def fluency(caption):
    ids=tok(caption,return_tensors="pt").to(DEV)
    if ids.input_ids.shape[1]<2: return 0.0
    o=model(input_ids=ids.input_ids);lp=torch.log_softmax(o.logits[0,:-1].float(),-1);tgt=ids.input_ids[0,1:]
    return float(lp[range(len(tgt)),tgt].mean())
ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
idxs=[i for i in range(len(ds)) if a.split=="all" or (a.split=="test" and i%2==1) or (a.split=="cal" and i%2==0)][:a.n]
fout=open(a.out,"w");res_b=[];res_v=[];flu_b=[];flu_v=[];samples=[];nab=0
print(f"[vt] tau={a.tau} rk={a.rk} mode={a.mode} split={a.split} n={len(idxs)}",flush=True)
for c,i in enumerate(idxs):
    img=ds[i]["image"].convert("RGB")
    S["sel"]=None; vl,hn,P=prep(img); cap=gen(vl)
    _,node,_,_=cu.caption_to_words(cap)
    bad=[o for o in set(node) if o not in PROTECT and o in objE and gc_of(hn,o)<a.tau]
    if bad:
        sel=set(); dirs={}
        for o in bad:
            toks=support_tokens(hn,o,a.rk); sel|=set(toks)
            if a.mode=="project" and objE.get(o) is not None:
                for t in toks: dirs[t]=objE[o]   # 该patch沿其支撑object的语义方向投影
        if a.random and sel:   # 特异性对照: 换成同数量随机patch(非幻觉支撑), 同样扰动
            import random as _r; _r.seed(1000+i)
            sel=set(_r.sample(range(P), min(len(sel),P)))
        S["sel"]=sorted(sel); S["mode"]=a.mode; S["dirs"]=dirs; S["scale"]=a.scale
        banned=list({t for o in bad for t in canon2ftoks.get(o,set())}) if a.ban else None
        newcap=gen(vl,banned); S["sel"]=None; nab+=1
    else:
        newcap=cap
    _,nv,_,_=cu.caption_to_words(newcap)
    res_b.append({"answer":list(ds[i]["gt_object"]),"pred":node}); res_v.append({"answer":list(ds[i]["gt_object"]),"pred":nv})
    flu_b.append(fluency(cap)); flu_v.append(fluency(newcap))
    fout.write(json.dumps({"image_id":i,"bad":bad,"base":cap,"vtablate":newcap})+"\n")
    if bad and len(samples)<a.show: samples.append((i,bad,cap,newcap))
    if (c+1)%40==0: print(f"[vt] {c+1}/{len(idxs)} (ablated={nab})",flush=True)
fout.close()
def agg(res): return (cu.coco_cap_chair_aggregate_results_chair_s(res),cu.coco_cap_chair_aggregate_results_chair_i(res),
                      cu.coco_cap_chair_aggregate_results_recall(res),sum(len(x['pred']) for x in res)/len(res))
b=agg(res_b);v=agg(res_v)
print(f"\n[vt] baseline    {b[0]:.2f}/{b[1]:.2f}/{b[2]:.2f} objs={b[3]:.2f} fluency={np.mean(flu_b):.3f}")
print(f"[vt] 视觉侧中和  {v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f} objs={v[3]:.2f} fluency={np.mean(flu_v):.3f} (ablated={nab})")
gate="✓" if (v[3]>=6.6 and v[2]>=74) else "✗护栏"
print(f"[vt] 护栏={gate} (对比 excise+repair 40.8/12.0/74.5 fluent4.4)")
for iid,bad,old,new in samples:
    print(f"\n--- img{iid} 中和{bad}\n  原: {old[:200]}\n  新: {new[:200]}")
