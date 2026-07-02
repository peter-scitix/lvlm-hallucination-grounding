#!/usr/bin/env python3
"""方法B: 指令自我改写(training-free自我纠错, 通顺by construction)。
输入 baseline caption(F_base) + grounding-flagged(base-rate-aware)的幻觉object;
指令模型: "重写这段描述, 去掉对 {flagged} 的提及, 保留其余细节"; 模型条件在图像+原caption上产出通顺新caption。
对比 baseline / 删名词excise。报 CHAIR + fluency(mean token logprob, 越高越通顺) + objs/cap护栏。
flagged决策复用 excise_gc.json(gc<tau) + protect名单。"""
import argparse, importlib.util, json, os, sys
from collections import defaultdict
if "--gpu" in sys.argv: os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[sys.argv.index("--gpu")+1]
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
SYS=("A chat between a curious user and an artificial intelligence assistant. "
     "The assistant gives helpful, detailed, and polite answers to the user's questions.")
def CU():
    p="/volume/exploration/EvolvingLMMs/lmms-eval/lmms_eval/tasks/coco_cap_chair/utils.py"
    s=importlib.util.spec_from_file_location("cu",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ap=argparse.ArgumentParser()
ap.add_argument("--tau",type=float,default=-0.06)
ap.add_argument("--flagger",default="grounding",choices=["grounding","selfcheck"])
ap.add_argument("--svthr",type=float,default=0.0,help="selfcheck: flag if (nl-yl)>svthr (模型倾向说no)")
ap.add_argument("--ban",action="store_true",help="改写时hard-ban flagged object的token(保证移除+通顺)")
ap.add_argument("--protect",default="person,tennis racket,surfboard,sports ball,elephant,umbrella")
ap.add_argument("--split",default="test",choices=["all","test","cal"])
ap.add_argument("--n",type=int,default=250); ap.add_argument("--max_new_tokens",type=int,default=512)
ap.add_argument("--gpu",default="2"); ap.add_argument("--out",default="method/rw_out.jsonl"); ap.add_argument("--show",type=int,default=4)
a=ap.parse_args(); cu=CU()
PROTECT=set(x.strip() for x in a.protect.split(",") if x.strip())
torch.set_grad_enabled(False)
proc=AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf");tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",torch_dtype=torch.float16,device_map="cuda:0",attn_implementation="eager").eval()
img_id=model.config.image_token_index;eos=tok.eos_token_id;DEV="cuda:0"
gc=json.load(open("method/excise_gc.json"))
sv=json.load(open("detect/selfverify.json")) if a.flagger=="selfcheck" else {}
def is_flagged(i,o):
    if o in PROTECT: return False
    if a.flagger=="selfcheck":
        d=sv.get(str(i),{}).get(o); return (d is not None) and (d["nl"]-d["yl"])>a.svthr
    return gc.get(str(i),{}).get(o,1.0)<a.tau
canon2surf={}
for w,canon in cu.INVERSE_SYNONYM_DICT.items(): canon2surf.setdefault(canon,set()).add(w)
IRREG={"man":"men","woman":"women","person":"people","child":"children","foot":"feet","tooth":"teeth"}
canon2ftoks=defaultdict(set)
for w,canon in cu.INVERSE_SYNONYM_DICT.items():
    for s in [w,w+"s",w.rstrip("s"),w.capitalize()]+([IRREG[w],IRREG[w].capitalize()] if w in IRREG else []):
        ids=tok.encode(s,add_special_tokens=False)
        if ids: canon2ftoks[canon].add(ids[0])
ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)

@torch.no_grad()
def rewrite(image, caption, flagged_surfaces, flagged_canons):
    fl=", ".join(sorted(flagged_surfaces))
    prompt=(f"{SYS} USER: <image>\nHere is a description of the image:\n\"{caption}\"\n\n"
            f"Rewrite this description so that it does NOT mention {fl}. "
            f"Keep every other detail and object exactly as described, and keep it fluent. ASSISTANT:")
    inp=proc(images=image,text=prompt,return_tensors="pt").to(DEV,torch.float16)
    if not a.ban:
        out=model.generate(**inp,max_new_tokens=a.max_new_tokens,do_sample=False,num_beams=1)
        return tok.decode(out[0,inp.input_ids.shape[1]:],skip_special_tokens=True).strip()
    banned=list({t for o in flagged_canons for t in canon2ftoks.get(o,set())})
    o=model(**inp,use_cache=True); pk=o.past_key_values; lg=o.logits[0,-1,:].float(); gen=[]
    for _ in range(a.max_new_tokens):
        if banned: lg[banned]=-float("inf")
        nxt=int(lg.argmax())
        if nxt==eos: break
        gen.append(nxt)
        o=model(input_ids=torch.tensor([[nxt]],device=DEV),past_key_values=pk,use_cache=True)
        pk=o.past_key_values; lg=o.logits[0,-1,:].float()
    return tok.decode(gen,skip_special_tokens=True).strip()

@torch.no_grad()
def fluency(caption):
    """mean token logprob of the caption under the LM (higher=更通顺). 用describe prompt前缀无所谓, 只测caption自身。"""
    ids=tok(caption,return_tensors="pt").to(DEV)
    if ids.input_ids.shape[1]<2: return 0.0
    o=model(input_ids=ids.input_ids)
    lp=torch.log_softmax(o.logits[0,:-1].float(),-1)
    tgt=ids.input_ids[0,1:]
    return float(lp[range(len(tgt)),tgt].mean())

idxs=[i for i in range(len(ds)) if a.split=="all" or (a.split=="test" and i%2==1) or (a.split=="cal" and i%2==0)]
base={json.loads(l)["image_id"]:json.loads(l) for l in open("method/F_base.jsonl")}
idxs=[i for i in idxs if i in base][:a.n]
fout=open(a.out,"w");res_b=[];res_r=[];flu_b=[];flu_r=[];samples=[];nrw=0
print(f"[rw] tau={a.tau} protect={len(PROTECT)} split={a.split} n={len(idxs)}",flush=True)
for c,i in enumerate(idxs):
    r=base[i]; cap=r["caption"]
    flagged=[o for o in set(r["pred"]) if is_flagged(i,o)]
    if flagged:
        surfs=set()
        for o in flagged: surfs|=(canon2surf.get(o,{o}) & set())  # only surfaces actually in caption
        # use canonical + present surfaces
        low=cap.lower(); present=set()
        for o in flagged:
            for sf in canon2surf.get(o,{o}):
                if sf in low: present.add(sf)
            present.add(o)
        newcap=rewrite(ds[i]["image"].convert("RGB"), cap, present, flagged); nrw+=1
    else:
        newcap=cap
    _,nb,_,_=cu.caption_to_words(cap); _,nr,_,_=cu.caption_to_words(newcap)
    res_b.append({"answer":list(ds[i]["gt_object"]),"pred":nb}); res_r.append({"answer":list(ds[i]["gt_object"]),"pred":nr})
    flu_b.append(fluency(cap)); flu_r.append(fluency(newcap))
    fout.write(json.dumps({"image_id":i,"flagged":flagged,"base":cap,"rewrite":newcap})+"\n")
    if flagged and len(samples)<a.show: samples.append((i,flagged,cap,newcap))
    if (c+1)%40==0: print(f"[rw] {c+1}/{len(idxs)} (rewrites={nrw})",flush=True)
fout.close()
def agg(res): return (cu.coco_cap_chair_aggregate_results_chair_s(res),cu.coco_cap_chair_aggregate_results_chair_i(res),
                      cu.coco_cap_chair_aggregate_results_recall(res),sum(len(x['pred']) for x in res)/len(res))
b=agg(res_b); r=agg(res_r)
import numpy as np
print(f"\n[rw] baseline   CHAIR_s={b[0]:.2f} CHAIR_i={b[1]:.2f} recall={b[2]:.2f} objs={b[3]:.2f} fluency={np.mean(flu_b):.3f}")
print(f"[rw] 指令自改写 CHAIR_s={r[0]:.2f} CHAIR_i={r[1]:.2f} recall={r[2]:.2f} objs={r[3]:.2f} fluency={np.mean(flu_r):.3f} (rewrites={nrw})")
gate="✓" if (r[3]>=6.6 and r[2]>=74) else "✗护栏"
print(f"[rw] 护栏={gate} (objs≥6.6 & recall≥74); 目标CHAIR_s≤38 recall≥74 通顺")
for iid,fl,old,new in samples:
    print(f"\n--- img{iid} 去除{fl}\n  原: {old[:200]}\n  新: {new[:200]}")
