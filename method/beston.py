#!/usr/bin/env python3
"""方法A(CHAIR提点, 非消除): best-of-N + 检测重排。
对每张图采样 N 个 caption, 用 grounding 检测给每个 caption 打分(其 object 里被判幻觉的个数/比例),
挑幻觉最少的那个。是"选择天生好的生成"而非"移除"=> 通顺+全recall, 且不受移除 frontier 约束
(能否提点取决于: 样本里是否存在天然低幻觉的caption + 检测能否挑出它)。
grounding gc(o) 只依赖 image+object => 每图一次 grounding forward, 各候选 caption 只是 object 集不同, 查表打分。
对比 greedy baseline。扫多种重排分数。"""
import argparse, importlib.util, json, os, sys
import torch, torch.nn.functional as F
if "--gpu" in sys.argv: os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[sys.argv.index("--gpu")+1]
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
ap.add_argument("--N",type=int,default=6); ap.add_argument("--temp",type=float,default=0.8)
ap.add_argument("--tau",type=float,default=-0.03,help="gc<tau 记为一个预测幻觉")
ap.add_argument("--split",default="test"); ap.add_argument("--n",type=int,default=250); ap.add_argument("--max_new_tokens",type=int,default=512)
ap.add_argument("--gpu",default="1"); ap.add_argument("--out",default="method/beston_out.jsonl")
a=ap.parse_args(); cu=CU()
torch.set_grad_enabled(False)
proc=AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf");tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",torch_dtype=torch.float16,device_map="cuda:0",attn_implementation="eager").eval()
img_id=model.config.image_token_index;DEV="cuda:0";W=model.lm_head.weight;norm=model.model.language_model.norm
cal=json.load(open("detect/calibration_tkb.json"));calmean=sum(cal.values())/len(cal)
YES=tok.encode("Yes",add_special_tokens=False)[0]; NO=tok.encode("No",add_special_tokens=False)[0]
yl_=tok.encode("yes",add_special_tokens=False)[0]; nl_=tok.encode("no",add_special_tokens=False)[0]
_svc={}
@torch.no_grad()
def selfverify_no(image,o):
    """返回 nl-yl (>0 => 模型说no => 预测幻觉). 每(image,o)缓存(同图内object复用)。"""
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
    hn=F.normalize(norm(o.hidden_states[31][0,vis,:]).float(),dim=-1)
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
ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
idxs=[i for i in range(len(ds)) if a.split=="all" or (a.split=="test" and i%2==1) or (a.split=="cal" and i%2==0)][:a.n]
# 多种重排score => 各自的pred集合
picks={"greedy":[],"minhall":[],"recall_aware":[],"minhall_SV":[],"recall_aware_SV":[],"combo_gc+SV":[]}
all_scored=[]  # oracle上界用
gt={};fout=open(a.out,"w")
print(f"[beston] N={a.N} temp={a.temp} tau={a.tau} split={a.split} n={len(idxs)}",flush=True)
for c,i in enumerate(idxs):
    img=ds[i]["image"].convert("RGB"); gt[i]=list(ds[i]["gt_object"])
    vl,hn=prep(img); g=greedy_cap(vl); caps=sample_caps(vl,g)
    _svc.clear()  # self-verify缓存按图清
    scored=[]
    for cap in caps:
        _,node,_,_=cu.caption_to_words(cap)
        objs=list(node); nh=sum(gc_of(hn,o)<a.tau for o in set(objs))  # grounding预测幻觉个数
        nh_sv=sum(selfverify_no(img,o)>0 for o in set(objs))            # self-verify预测幻觉个数
        no=len(set(objs)); rate=nh/max(no,1)
        scored.append({"cap":cap,"node":objs,"nh":nh,"nh_sv":nh_sv,"no":no,"rate":rate})
    picks["greedy"].append(scored[0])
    picks["minhall"].append(min(scored,key=lambda s:(s["nh"],-s["no"])))            # grounding: 幻觉最少
    picks["recall_aware"].append(min(scored,key=lambda s:(s["nh"]-0.5*s["no"])))    # grounding: 幻觉少+奖励object
    picks["minhall_SV"].append(min(scored,key=lambda s:(s["nh_sv"],-s["no"])))       # self-verify重排
    picks["recall_aware_SV"].append(min(scored,key=lambda s:(s["nh_sv"]-0.5*s["no"])))
    picks["combo_gc+SV"].append(min(scored,key=lambda s:(s["nh"]+s["nh_sv"]-0.5*s["no"]))) # 两检测器融合
    all_scored.append(scored)
    fout.write(json.dumps({"image_id":i,"scored":[{k:s[k] for k in['nh','no','rate']} for s in scored]})+"\n")
    if (c+1)%40==0: print(f"[beston] {c+1}/{len(idxs)}",flush=True)
fout.close()
def agg(sel):
    R=[{"answer":gt[i],"pred":s["node"]} for i,s in zip(idxs,sel)]
    return (cu.coco_cap_chair_aggregate_results_chair_s(R),cu.coco_cap_chair_aggregate_results_chair_i(R),
            cu.coco_cap_chair_aggregate_results_recall(R),sum(len(x['pred']) for x in R)/len(R))
print()
for k,sel in picks.items():
    m=agg(sel); print(f"[beston] {k:14} CHAIR_s={m[0]:.2f} CHAIR_i={m[1]:.2f} recall={m[2]:.2f} objs={m[3]:.2f}")
# oracle上界: 用GT挑每图真幻觉最少的样本(tiebreak: 多object) => best-of-N的天花板
orc=[]
for i,sc in zip(idxs,all_scored):
    g=set(gt[i]); orc.append(min(sc,key=lambda s:(sum(1 for o in set(s["node"]) if o not in g),-s["no"])))
m=agg(orc); print(f"[beston] {'ORACLE(上界)':14} CHAIR_s={m[0]:.2f} CHAIR_i={m[1]:.2f} recall={m[2]:.2f} objs={m[3]:.2f}")
