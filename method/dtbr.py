#!/usr/bin/env python3
"""DTBR: 检测器触发的局部回溯-重采样 (干预维度白区; 训练-free, self-verify 当触发器/价值函数)。
思路(区别于 编辑移除 与 全caption best-of-N):
  1. greedy 生成 caption; 用 self-verify 检出幻觉 object;
  2. 回溯到"最早的幻觉 object"的首 token 位置 p, 保留好前缀 prefix[:p];
  3. 从 prefix 局部重采样 K 个"续写"(temperature), 用 self-verify 给每个续写打分(新幻觉最少);
  4. 拼 prefix + 最优续写。 -> 改变了 mention set(跳出 removal frontier), 只重解码flag后的部分(保recall+便宜)。
对比 baseline / DTBR (+ 参考: 清零-14.8, best-of-N-21.6, oracle-17.6)。"""
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
ap.add_argument("--K",type=int,default=4,help="局部重采样候选数"); ap.add_argument("--temp",type=float,default=0.8)
ap.add_argument("--rounds",type=int,default=2,help="回溯轮数(处理续写里的新幻觉)")
ap.add_argument("--n",type=int,default=250); ap.add_argument("--max_new_tokens",type=int,default=512)
ap.add_argument("--gpu",default="1"); ap.add_argument("--out",default="method/dtbr_out.jsonl")
ap.add_argument("--model",default="llava-hf/llava-1.5-7b-hf")
a=ap.parse_args(); cu=CU()
torch.set_grad_enabled(False)
proc=AutoProcessor.from_pretrained(a.model);tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained(a.model,torch_dtype=torch.float16,device_map="cuda:0",attn_implementation="eager").eval()
img_id=model.config.image_token_index;DEV="cuda:0"
YES=tok.encode("Yes",add_special_tokens=False)[0]; NO=tok.encode("No",add_special_tokens=False)[0]
yl_=tok.encode("yes",add_special_tokens=False)[0]; nl_=tok.encode("no",add_special_tokens=False)[0]
IRREG={"man":"men","woman":"women","person":"people","child":"children"}
canon2ft=defaultdict(set)
for w,canon in cu.INVERSE_SYNONYM_DICT.items():
    for s in [w,w+"s",w.rstrip("s")," "+w," "+w+"s"]+([" "+IRREG[w]] if w in IRREG else []):
        ids=tok.encode(s,add_special_tokens=False)
        if ids: canon2ft[canon].add(ids[0])
_svc={}
@torch.no_grad()
def sv_no(image,o):
    if o in _svc: return _svc[o]
    pr=f"{SYS} USER: <image>\nIs there a {o} in the image?\nAnswer the question using a single word or phrase. ASSISTANT:"
    inp=proc(images=image,text=pr,return_tensors="pt").to(DEV,torch.float16)
    lg=model(**inp).logits[0,-1,:].float()
    v=float(torch.logsumexp(lg[[NO,nl_]],0)-torch.logsumexp(lg[[YES,yl_]],0)); _svc[o]=v; return v
def flagged_objs(image,cap):
    _,node,_,_=cu.caption_to_words(cap); return [o for o in set(node) if sv_no(image,o)>0]
@torch.no_grad()
def greedy_from(base_ids,pix):
    out=model.generate(input_ids=base_ids,pixel_values=pix,max_new_tokens=a.max_new_tokens,do_sample=False,num_beams=1)
    return out[0,base_ids.shape[1]:]
@torch.no_grad()
def sample_from(base_ids,pix,K):
    out=model.generate(input_ids=base_ids,pixel_values=pix,max_new_tokens=a.max_new_tokens,do_sample=True,temperature=a.temp,top_p=0.95,num_return_sequences=K)
    return [out[j,base_ids.shape[1]:] for j in range(out.shape[0])]
def first_pos(gen_ids,objs):
    """gen_ids 里最早属于 objs 之一的 token 位置"""
    fts=set().union(*[canon2ft.get(o,set()) for o in objs]) if objs else set()
    for j in range(gen_ids.shape[0]):
        if int(gen_ids[j]) in fts: return j
    return None
@torch.no_grad()
def dtbr(image):
    vl=proc(images=image,text=P_IMG,return_tensors="pt").to(DEV,torch.float16)
    pix=vl.pixel_values; prompt=vl.input_ids
    gen=greedy_from(prompt,pix); _svc.clear()
    cap=tok.decode(gen,skip_special_tokens=True).strip()
    for r in range(a.rounds):
        bad=flagged_objs(image,cap)
        if not bad: break
        p=first_pos(gen,bad)
        if p is None or p==0: break
        base=torch.cat([prompt[0],gen[:p]]).unsqueeze(0)   # 保留好前缀
        cands=sample_from(base,pix,a.K)
        # 每个候选: 完整caption = gen[:p]+cand; 数其(整体)幻觉object; 选最少(recall tiebreak:更多object)
        best=None;bs=(99,0)
        for cand in cands:
            full=torch.cat([gen[:p],cand]); fcap=tok.decode(full,skip_special_tokens=True).strip()
            fb=flagged_objs(image,fcap); _,fn,_,_=cu.caption_to_words(fcap)
            sc=(len(fb),-len(set(fn)))
            if sc<bs: bs=sc; best=(full,fcap)
        if best is None: break
        gen,cap=best
    return cap
ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
idxs=[i for i in range(len(ds)) if i%2==1][:a.n]
res={"baseline":[],"dtbr":[]}; fout=open(a.out,"w"); print(f"[dtbr] n={len(idxs)} K={a.K} rounds={a.rounds}",flush=True)
for c,i in enumerate(idxs):
    img=ds[i]["image"].convert("RGB")
    vl=proc(images=img,text=P_IMG,return_tensors="pt").to(DEV,torch.float16)
    _svc.clear(); g=greedy_from(vl.input_ids,vl.pixel_values); base_cap=tok.decode(g,skip_special_tokens=True).strip()
    dt_cap=dtbr(img)
    res["baseline"].append({"answer":list(ds[i]["gt_object"]),"pred":cu.caption_to_words(base_cap)[1]})
    res["dtbr"].append({"answer":list(ds[i]["gt_object"]),"pred":cu.caption_to_words(dt_cap)[1]})
    fout.write(json.dumps({"i":i,"base":base_cap,"dtbr":dt_cap})+"\n")
    if (c+1)%40==0: print(f"[dtbr] {c+1}/{len(idxs)}",flush=True)
fout.close()
def agg(r): return (cu.coco_cap_chair_aggregate_results_chair_s(r),cu.coco_cap_chair_aggregate_results_chair_i(r),cu.coco_cap_chair_aggregate_results_recall(r),sum(len(x['pred']) for x in r)/len(r))
b=agg(res["baseline"]); d=agg(res["dtbr"])
print(f"\n[dtbr] baseline  {b[0]:.2f}/{b[1]:.2f}/{b[2]:.2f} objs={b[3]:.2f}")
print(f"[dtbr] DTBR      {d[0]:.2f}/{d[1]:.2f}/{d[2]:.2f} objs={d[3]:.2f}  ΔCHAIR_s={d[0]-b[0]:+.2f}")
print(f"[dtbr] 参考: 清零 -14.8(recall76) | best-of-N -21.6(recall71) | oracle -17.6")
