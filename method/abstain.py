#!/usr/bin/env python3
"""Idea E: grounding-gated selective abstention (基于 gground.py 加 ~20 行)。
soft 只压 flagged object 的 logit,模型往往换个同义 object 继续断言。abstention 给它一个「退出」出口:
当【压制前】的 argmax 落在某个 flagged(低grounding)object 的首token 时,给 hedge 词
(objects/something/items/...)加 bonus,让模型说泛化词而非断言假 object。hedge 词不在 CHAIR 词表 => 不计幻觉。
护栏(workflow 要求): 报 objs/cap 与 recall; objs/cap<6.6 或 recall<74 判 cheap-baseline(缩分母刷分)不算数。
abstain=0 时退化为纯 soft(同子集参照)。"""
import argparse, importlib.util, json, os
from collections import defaultdict
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
SYS=("A chat between a curious user and an artificial intelligence assistant. "
     "The assistant gives helpful, detailed, and polite answers to the user's questions.")
P_IMG=f"{SYS} USER: <image>\nPlease describe this image in detail. ASSISTANT:"
def CU():
    p="/volume/exploration/EvolvingLMMs/lmms-eval/lmms_eval/tasks/coco_cap_chair/utils.py"
    s=importlib.util.spec_from_file_location("cu",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ap=argparse.ArgumentParser()
ap.add_argument("--tau",type=float,default=-0.012); ap.add_argument("--beta",type=float,default=12.0)
ap.add_argument("--abstain",type=float,default=8.0,help="hedge 词 logit bonus (0=纯soft)")
ap.add_argument("--glayer",type=int,default=31); ap.add_argument("--gtopk",type=int,default=10)
ap.add_argument("--n",type=int,default=120); ap.add_argument("--max_new_tokens",type=int,default=512)
ap.add_argument("--split",default="all",choices=["all","test","cal"])
ap.add_argument("--gpu",default="1"); ap.add_argument("--out",default="method/ab_out.jsonl"); ap.add_argument("--label",default="abstain")
a=ap.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"]=a.gpu
cu=CU()
proc=AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf");tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",torch_dtype=torch.float16,device_map="cuda:0").eval()
W=model.lm_head.weight;norm=model.model.language_model.norm;img_id=model.config.image_token_index;eos=tok.eos_token_id;DEV="cuda:0"
cal=json.load(open("/volume/exploration/EvolvingLMMs/detect/calibration_tkb.json"));cand=list(cal.keys())
objE=torch.stack([F.normalize(W[tok.encode(o,add_special_tokens=False)].float().mean(0),dim=-1) for o in cand]).to(DEV)
IRREG={"man":"men","woman":"women","person":"people","child":"children","foot":"feet","tooth":"teeth"}
canon2ftoks=defaultdict(set)
for w,canon in cu.INVERSE_SYNONYM_DICT.items():
    for s in [w,w+"s",w.rstrip("s"),w.capitalize()]+([IRREG[w],IRREG[w].capitalize()] if w in IRREG else []):
        ids=tok.encode(s,add_special_tokens=False)
        if ids: canon2ftoks[canon].add(ids[0])
# hedge 词 (泛化/退出词, 非 COCO object)
HEDGE=["something","objects","items","some","several","various","other","things","stuff","more"]
hedge_ids=set()
for hw in HEDGE:
    for form in [hw," "+hw,hw.capitalize()]:
        ids=tok.encode(form,add_special_tokens=False)
        if ids: hedge_ids.add(ids[-1] if form==" "+hw else ids[0])
hedge_ids=list(hedge_ids)
@torch.inference_mode()
def run(image):
    vl=proc(images=image,text=P_IMG,return_tensors="pt").to(DEV,torch.float16)
    o=model(**vl,output_hidden_states=True,use_cache=True)
    vis=(vl.input_ids[0]==img_id).nonzero(as_tuple=True)[0]
    hn=F.normalize(norm(o.hidden_states[a.glayer][0,vis,:]).float(),dim=-1)
    g=(hn@objE.T).topk(min(a.gtopk,hn.shape[0]),dim=0).values.mean(0)
    pen=torch.zeros(W.shape[0],device=DEV); flagged=set()
    for i,oo in enumerate(cand):
        gc=g[i].item()-cal[oo]; u=a.tau-gc
        if u>0:
            for t in canon2ftoks.get(oo,()): pen[t]=max(pen[t].item(),a.beta*u); flagged.add(t)
    pk=o.past_key_values;lg=o.logits[0,-1,:].float();out=[];nred=0
    for _ in range(a.max_new_tokens):
        raw_top=int(lg.argmax())                       # 压制前 argmax
        z=lg.clone()
        if a.abstain>0 and raw_top in flagged:         # 模型想说一个 flagged(低grounding) object
            z[hedge_ids]=z[hedge_ids]+a.abstain; nred+=1
        nxt=int((z-pen).argmax())
        if nxt==eos: break
        out.append(nxt);t=torch.tensor([[nxt]],device=DEV)
        o2=model(input_ids=t,past_key_values=pk,use_cache=True);pk=o2.past_key_values;lg=o2.logits[0,-1,:].float()
    return tok.decode(out,skip_special_tokens=True).strip(),nred
ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
idxs=[i for i in range(len(ds)) if a.split=="all" or (a.split=="test" and i%2==1) or (a.split=="cal" and i%2==0)][:a.n]
fout=open(a.out,"w");res=[];n=len(idxs);tot=0
print(f"[ab] {a.label} tau={a.tau} beta={a.beta} abstain={a.abstain} split={a.split} n={n} |hedge|={len(hedge_ids)}",flush=True)
for c,i in enumerate(idxs):
    cap,nred=run(ds[i]["image"].convert("RGB"));_,node,_,_=cu.caption_to_words(cap);tot+=nred
    res.append({"answer":list(ds[i]["gt_object"]),"pred":node});fout.write(json.dumps({"image_id":i,"caption":cap,"pred":node,"nred":nred})+"\n")
    if (c+1)%40==0:print(f"[ab] {c+1}/{n}",flush=True)
fout.close()
cs=cu.coco_cap_chair_aggregate_results_chair_s(res);ci=cu.coco_cap_chair_aggregate_results_chair_i(res);rc=cu.coco_cap_chair_aggregate_results_recall(res)
objs=sum(len(x['pred']) for x in res)/len(res)
flag="✓" if (objs>=6.6 and rc>=74) else "✗ cheap-baseline(缩分母)"
print(f"\n[ab] {a.label} CHAIR_s={cs:.2f} CHAIR_i={ci:.2f} recall={rc:.2f} objs={objs:.2f} redirects/img={tot/n:.2f} 护栏={flag} (soft参照43.4/13.2/74.5, base50.4/15.4/76.9)",flush=True)
