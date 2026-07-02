#!/usr/bin/env python3
"""Detection-guided suppression sweep: per-object CALIBRATED grounding gc(o)=g(o)-cal[o]; flag gc<tau;
ban flagged objects' first-tokens during greedy decode. Load once, grounding computed once per image,
sweep tau. Reports CHAIR_s/i/recall/avg_objs + mean #flagged per tau."""
import argparse, importlib.util, json, os, copy
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration

SYS = ("A chat between a curious user and an artificial intelligence assistant. "
       "The assistant gives helpful, detailed, and polite answers to the user's questions.")
P_IMG = f"{SYS} USER: <image>\nPlease describe this image in detail. ASSISTANT:"
def CU():
    p="/volume/exploration/EvolvingLMMs/lmms-eval/lmms_eval/tasks/coco_cap_chair/utils.py"
    s=importlib.util.spec_from_file_location("cu",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

ap=argparse.ArgumentParser()
ap.add_argument("--n",type=int,default=80)
ap.add_argument("--taus",default="-1,0.0,0.005,0.01,0.015")   # -1 = baseline (flag none)
ap.add_argument("--glayer",type=int,default=31); ap.add_argument("--gtopk",type=int,default=10)
ap.add_argument("--max_new_tokens",type=int,default=110); ap.add_argument("--gpu",default="1")
a=ap.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"]=a.gpu
taus=[float(x) for x in a.taus.split(",")]
cu=CU()
proc=AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf");tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",torch_dtype=torch.float16,device_map="cuda:0").eval()
W=model.lm_head.weight;norm=model.model.language_model.norm;img_id=model.config.image_token_index;eos=tok.eos_token_id;DEV="cuda:0"
calj=json.load(open("/volume/exploration/EvolvingLMMs/detect/calibration.json"))
cal=calj["per_object_mean_grounding"];cand=list(cal.keys())
objE=[];ftok={}
for o in cand:
    ids=tok.encode(" "+o,add_special_tokens=False) or tok.encode(o,add_special_tokens=False)
    objE.append(F.normalize(W[ids].float().mean(0),dim=-1));ftok[o]=ids[0]
objE=torch.stack(objE).to(DEV)

@torch.inference_mode()
def gen_with_ban(vl, ban_ids):
    banmask=torch.zeros(W.shape[0],device=DEV)
    for t in ban_ids: banmask[t]=1e9
    o=model(**vl,use_cache=True);pk=o.past_key_values;lg=o.logits[:,-1,:].float()
    out=[]
    for _ in range(a.max_new_tokens):
        nxt=int((lg-banmask).argmax())
        if nxt==eos:break
        out.append(nxt);t=torch.tensor([[nxt]],device=DEV)
        o=model(input_ids=t,past_key_values=pk,use_cache=True);pk=o.past_key_values;lg=o.logits[:,-1,:].float()
    return tok.decode(out,skip_special_tokens=True).strip()

@torch.inference_mode()
def grounding(vl):
    o=model(**vl,output_hidden_states=True,use_cache=False)
    vis=(vl.input_ids[0]==img_id).nonzero(as_tuple=True)[0]
    hn=F.normalize(norm(o.hidden_states[a.glayer][0,vis,:]).float(),dim=-1)
    cos=hn@objE.T
    return cos.topk(min(a.gtopk,cos.shape[0]),dim=0).values.mean(0)   # (C,)

ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
n=min(a.n,len(ds));res={t:[] for t in taus};nflag={t:0 for t in taus}
print(f"[dg] n={n} taus={taus}",flush=True)
for i in range(n):
    img=ds[i]["image"].convert("RGB")
    vl=proc(images=img,text=P_IMG,return_tensors="pt").to(DEV,torch.float16)
    g=grounding(vl)
    gc={o:(g[ci].item()-cal[o]) for ci,o in enumerate(cand)}
    for t in taus:
        ban=[] if t==-1 else [ftok[o] for o in cand if gc[o]<t]
        nflag[t]+=len(ban)
        cap=gen_with_ban(vl,set(ban))
        _,node,_,_=cu.caption_to_words(cap)
        res[t].append({"answer":list(ds[i]["gt_object"]),"pred":node})
    if (i+1)%20==0:print(f"[dg] {i+1}/{n}",flush=True)
print(f"\n{'tau':>7} {'CHAIR_s':>8} {'CHAIR_i':>8} {'recall':>7} {'objs':>6} {'flagged/img':>11}")
for t in taus:
    r=res[t];cs=cu.coco_cap_chair_aggregate_results_chair_s(r);ci=cu.coco_cap_chair_aggregate_results_chair_i(r);rc=cu.coco_cap_chair_aggregate_results_recall(r)
    al=sum(len(x["pred"]) for x in r)/len(r)
    print(f"{t:>7} {cs:>8.2f} {ci:>8.2f} {rc:>7.2f} {al:>6.2f} {nflag[t]/n:>11.1f}")
print("(baseline ref full-500: 52.20/16.18/77.19)")
