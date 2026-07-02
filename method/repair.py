#!/usr/bin/env python3
"""解耦方案: excise决定删什么(最优CHAIR/recall) + 语法修复pass补通顺(ban被删object, 只修语法不改内容)。
思路: 用grounding+base-rate-protect excise(已知test 40.0/11.9/72.6)删掉幻觉object的名词短语->破碎文本
-> 喂LM"修正语法, 不新增任何object/细节"+ ban被删object的token -> 通顺文本(object集不变=CHAIR/recall保持)。
关键: repair只补语法, 因ban+指令不会重新引入被删object, 也因input已是excised故少漂移(比从原caption重生成faithful)。
对比 excise(破碎) / 原ban-rewrite(漂移)。报CHAIR+fluency+recall。"""
import argparse, importlib.util, json, os, re, sys
from collections import defaultdict
if "--gpu" in sys.argv: os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[sys.argv.index("--gpu")+1]
import torch, numpy as np
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
SYS=("A chat between a curious user and an artificial intelligence assistant. "
     "The assistant gives helpful, detailed, and polite answers to the user's questions.")
def CU():
    p="/volume/exploration/EvolvingLMMs/lmms-eval/lmms_eval/tasks/coco_cap_chair/utils.py"
    s=importlib.util.spec_from_file_location("cu",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ap=argparse.ArgumentParser()
ap.add_argument("--tau",type=float,default=-0.06)
ap.add_argument("--protect",default="person,tennis racket,surfboard,sports ball,elephant,umbrella")
ap.add_argument("--split",default="test"); ap.add_argument("--n",type=int,default=250)
ap.add_argument("--faithful",action="store_true",help="repair时ban掉所有缺席COCO object=>CHAIR严格=excise")
ap.add_argument("--gpu",default="1"); ap.add_argument("--out",default="method/repair_out.jsonl"); ap.add_argument("--show",type=int,default=4)
a=ap.parse_args(); cu=CU()
PROTECT=set(x.strip() for x in a.protect.split(",") if x.strip())
torch.set_grad_enabled(False)
proc=AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf");tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",torch_dtype=torch.float16,device_map="cuda:0",attn_implementation="eager").eval()
eos=tok.eos_token_id;DEV="cuda:0"
gc=json.load(open("method/excise_gc.json"))
canon2surf=defaultdict(set); IRREG={"man":"men","woman":"women","person":"people","child":"children","foot":"feet","tooth":"teeth"}
canon2ftoks=defaultdict(set)
for w,canon in cu.INVERSE_SYNONYM_DICT.items():
    canon2surf[canon].add(w)
    for s in [w,w+"s",w.rstrip("s"),w.capitalize()]+([IRREG[w],IRREG[w].capitalize()] if w in IRREG else []):
        ids=tok.encode(s,add_special_tokens=False)
        if ids: canon2ftoks[canon].add(ids[0])
def excise_text(cap, flagged):
    out=cap; surfs=set()
    for c in flagged: surfs|=canon2surf.get(c,{c})
    for sf in sorted(surfs,key=lambda x:-len(x)):
        pat=r'\b(?:a|an|the|some|several|two|three|four|many|various)?\s*(?:\w+\s+){0,2}'+re.escape(sf)+r's?\b'
        out=re.sub(pat,' ',out,flags=re.IGNORECASE)
    out=re.sub(r'\s+',' ',out); out=re.sub(r'\s+([.,;])',r'\1',out); return out.strip()
@torch.no_grad()
def repair(broken, banned):
    prompt=(f"{SYS} USER: The following image description has some words removed and reads awkwardly. "
            f"Fix only the grammar and flow so it reads naturally. Do NOT add any new objects, people, or details. "
            f"Text: \"{broken}\" ASSISTANT: Here is the corrected description:")
    inp=tok(prompt,return_tensors="pt").to(DEV)
    o=model(input_ids=inp.input_ids,use_cache=True); pk=o.past_key_values; lg=o.logits[0,-1,:].float(); gen=[]
    bl=list(banned)
    for _ in range(300):
        if bl: lg[bl]=-float("inf")
        nxt=int(lg.argmax())
        if nxt==eos: break
        gen.append(nxt)
        o=model(input_ids=torch.tensor([[nxt]],device=DEV),past_key_values=pk,use_cache=True); pk=o.past_key_values; lg=o.logits[0,-1,:].float()
    t=tok.decode(gen,skip_special_tokens=True).strip().strip('"')
    return t
@torch.no_grad()
def fluency(caption):
    ids=tok(caption,return_tensors="pt").to(DEV)
    if ids.input_ids.shape[1]<2: return 0.0
    o=model(input_ids=ids.input_ids); lp=torch.log_softmax(o.logits[0,:-1].float(),-1); tgt=ids.input_ids[0,1:]
    return float(lp[range(len(tgt)),tgt].mean())
ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
gt={i:list(ds[i]["gt_object"]) for i in range(len(ds))}
base={json.loads(l)["image_id"]:json.loads(l) for l in open("method/F_base.jsonl")}
idxs=[i for i in range(len(ds)) if a.split=="all" or (a.split=="test" and i%2==1) or (a.split=="cal" and i%2==0)]
idxs=[i for i in idxs if i in base][:a.n]
res_b=[];res_e=[];res_r=[];flu_e=[];flu_r=[];samples=[];fout=open(a.out,"w")
print(f"[repair] tau={a.tau} split={a.split} n={len(idxs)}",flush=True)
for c,i in enumerate(idxs):
    r=base[i]; cap=r["caption"]; d=gc.get(str(i),{})
    flagged=[o for o in set(r["pred"]) if o not in PROTECT and d.get(o,1.0)<a.tau]
    if flagged:
        broken=excise_text(cap,set(flagged))
        if a.faithful:
            # ban 掉所有"不在破碎文本里"的COCO object => repair不能引入任何新object => CHAIR严格=excise
            _,present,_,_=cu.caption_to_words(broken); present=set(present)
            banned={t for c in canon2ftoks for t in canon2ftoks[c] if c not in present}
        else:
            banned={t for o in flagged for t in canon2ftoks.get(o,set())}
        rep=repair(broken,banned)
    else:
        broken=cap; rep=cap
    _,nb,_,_=cu.caption_to_words(cap);_,ne,_,_=cu.caption_to_words(broken);_,nr,_,_=cu.caption_to_words(rep)
    res_b.append({"answer":gt[i],"pred":nb});res_e.append({"answer":gt[i],"pred":ne});res_r.append({"answer":gt[i],"pred":nr})
    flu_e.append(fluency(broken));flu_r.append(fluency(rep))
    fout.write(json.dumps({"image_id":i,"flagged":flagged,"broken":broken,"repaired":rep})+"\n")
    if flagged and len(samples)<a.show: samples.append((i,flagged,broken,rep))
    if (c+1)%40==0: print(f"[repair] {c+1}/{len(idxs)}",flush=True)
fout.close()
def agg(res): return (cu.coco_cap_chair_aggregate_results_chair_s(res),cu.coco_cap_chair_aggregate_results_chair_i(res),
                      cu.coco_cap_chair_aggregate_results_recall(res),sum(len(x['pred']) for x in res)/len(res))
b=agg(res_b);e=agg(res_e);rp=agg(res_r)
print(f"\n[repair] baseline      {b[0]:.2f}/{b[1]:.2f}/{b[2]:.2f} objs={b[3]:.2f}")
print(f"[repair] excise(破碎)   {e[0]:.2f}/{e[1]:.2f}/{e[2]:.2f} objs={e[3]:.2f} fluency={np.mean(flu_e):.3f}")
print(f"[repair] excise+修复    {rp[0]:.2f}/{rp[1]:.2f}/{rp[2]:.2f} objs={rp[3]:.2f} fluency={np.mean(flu_r):.3f}")
gate="✓" if (rp[3]>=6.6 and rp[2]>=74 and rp[0]<=38) else ("recall/objs ok" if (rp[3]>=6.6 and rp[2]>=74) else "✗")
print(f"[repair] 判据={gate}")
for iid,fl,bk,rp2 in samples:
    print(f"\n--- img{iid} 删{fl}\n  破碎: {bk[:200]}\n  修复: {rp2[:200]}")
