#!/usr/bin/env python3
"""忠实完整复现 EAZY(ICCV'25) 算法, 同协议 vs sv-gate 头对头。
EAZY算法(读官方码): 1)生成caption C0+attention; 2)所有object的attention-L14 top-k支撑; 3)清全部支撑->重生成C1;
4)反事实检测: C0里有、C1里消失的object=幻觉, 只留它们的支撑=final; 5)清final->重生成C2=EAZY输出。
(原码用minigpt4+transformers4.29自定义generate; 此为算法级忠实复现, 同数据/协议。)"""
import argparse, importlib.util, json, os, sys
from collections import defaultdict
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
ap.add_argument("--k",type=int,default=20); ap.add_argument("--layer",type=int,default=14)
ap.add_argument("--n",type=int,default=250); ap.add_argument("--max_new_tokens",type=int,default=512)
ap.add_argument("--gpu",default="1"); ap.add_argument("--out",default="method/eazy_full.jsonl")
ap.add_argument("--model",default="llava-hf/llava-1.5-7b-hf")
a=ap.parse_args(); cu=CU()
torch.set_grad_enabled(False)
proc=AutoProcessor.from_pretrained(a.model);tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained(a.model,torch_dtype=torch.float16,device_map="cuda:0",attn_implementation="eager").eval()
img_id=model.config.image_token_index;DEV="cuda:0"
IRREG={"man":"men","woman":"women","person":"people","child":"children"}
canon2ft=defaultdict(set)
for w,canon in cu.INVERSE_SYNONYM_DICT.items():
    for s in [w,w+"s",w.rstrip("s")," "+w," "+w+"s"]+([" "+IRREG[w]] if w in IRREG else []):
        ids=tok.encode(s,add_special_tokens=False)
        if ids: canon2ft[canon].add(ids[0])
S={"sel":None}
def hook(m,i,o):
    if S["sel"] is None: return o
    f=o.view(-1,o.shape[-1]) if o.dim()==3 else o
    for k in S["sel"]:
        if k<f.shape[0]: f[k]=0
    return o
model.multi_modal_projector.register_forward_hook(hook)
@torch.no_grad()
def gen(vl):
    S_prev=S["sel"]
    out=model.generate(**vl,max_new_tokens=a.max_new_tokens,do_sample=False,num_beams=1)
    return out[0],tok.decode(out[0,vl.input_ids.shape[1]:],skip_special_tokens=True).strip()
@torch.no_grad()
def attn_support(seq,pix,vis,objs,plen):
    """所有object的 layer-a.layer attention top-k 图像patch (EAZY定位)"""
    S["sel"]=None
    o=model(input_ids=seq.unsqueeze(0),pixel_values=pix,output_attentions=True,use_cache=False)
    att=o.attentions[a.layer][0]  # (heads,T,T)
    gen_ids=seq[plen:]; ngen=gen_ids.shape[0]; T=seq.shape[0]; base=T-ngen
    obj2sel={}
    for ob in objs:
        fts=canon2ft.get(ob,set()); sel=set()
        for j in range(ngen):
            if int(gen_ids[j]) in fts:
                a2img=att[:,base+j,:][:,vis].mean(0)
                for idx in a2img.topk(min(a.k,a2img.shape[0])).indices.tolist(): sel.add(idx)
        obj2sel[ob]=sel
    return obj2sel
ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
idxs=[i for i in range(len(ds)) if i%2==1][:a.n]
res={"baseline":[],"eazy":[]}; fout=open(a.out,"w"); print(f"[eazyfull] n={len(idxs)} k={a.k} layer={a.layer}",flush=True)
for c,i in enumerate(idxs):
    img=ds[i]["image"].convert("RGB")
    vl=proc(images=img,text=P_IMG,return_tensors="pt").to(DEV,torch.float16); pix=vl.pixel_values; plen=vl.input_ids.shape[1]
    vis=(vl.input_ids[0]==img_id).nonzero(as_tuple=True)[0]
    S["sel"]=None; seq0,C0=gen(vl)
    _,objs0,_,_=cu.caption_to_words(C0); objs0=list(set(objs0))
    if not objs0:
        res["baseline"].append({"answer":list(ds[i]["gt_object"]),"pred":[]}); res["eazy"].append({"answer":list(ds[i]["gt_object"]),"pred":[]}); continue
    obj2sel=attn_support(seq0,pix,vis,objs0,plen)
    allsel=sorted(set().union(*obj2sel.values())) if obj2sel else []
    # pass1: 清全部支撑 -> C1
    S["sel"]=allsel if allsel else None; _,C1=gen(vl); S["sel"]=None
    # 反事实检测: C0有C1无 = 幻觉
    _,objs1,_,_=cu.caption_to_words(C1); objs1=set(objs1)
    final=set()
    for ob in objs0:
        if ob not in objs1: final|=obj2sel.get(ob,set())
    # pass2: 只清final -> C2
    S["sel"]=sorted(final) if final else None; _,C2=gen(vl); S["sel"]=None
    res["baseline"].append({"answer":list(ds[i]["gt_object"]),"pred":objs0})
    res["eazy"].append({"answer":list(ds[i]["gt_object"]),"pred":cu.caption_to_words(C2)[1]})
    fout.write(json.dumps({"i":i,"n_final":len(final)})+"\n")
    if (c+1)%40==0: print(f"[eazyfull] {c+1}/{len(idxs)}",flush=True)
fout.close()
def agg(r): return (cu.coco_cap_chair_aggregate_results_chair_s(r),cu.coco_cap_chair_aggregate_results_chair_i(r),cu.coco_cap_chair_aggregate_results_recall(r),sum(len(x['pred']) for x in r)/len(r))
b=agg(res["baseline"]); e=agg(res["eazy"])
print(f"\n[eazyfull] baseline {b[0]:.2f}/{b[1]:.2f}/{b[2]:.2f}")
print(f"[eazyfull] EAZY(复现) {e[0]:.2f}/{e[1]:.2f}/{e[2]:.2f}  ΔCHAIR_s={e[0]-b[0]:+.2f}  (对比 sv-gate -14.8/recall76)")
