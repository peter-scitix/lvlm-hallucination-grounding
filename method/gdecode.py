#!/usr/bin/env python3
"""
Grounding-GATED decoding (our novel family) for LLaVA-1.5 (llava_hf, transformers 4.57). Training-free, per-image.

Per image, compute per-object CALIBRATED logit-lens grounding once at prefill:
    g(o)  = topk-mean cosine( norm(visual_token_hidden_L), unembed_row(o) )      (our detector, AUROC 0.82)
    gc(o) = g(o) - cal[o]        (per-object reference shift; cal from detect/calibration.json)
    u(o)  = relu(tau - gc(o))    (ungroundedness; >0 when below threshold)
Then intervene ONLY on object tokens, GATED by grounding (vs prior methods' blanket intervention):
    --ban           logit[first_tok(o)] = -inf           for gc(o) < tau
    --soft BETA     logit[first_tok(o)] -= BETA * u(o)    (continuous, proportional to ungroundedness)
    --gcontrast W   logit[first_tok(o)] = (1+W)*vl - W*txt  ONLY for gc(o)<tau (selective text-only contrast)
Flags stack. Scores CHAIR_s/i/recall (+ avg objs, POPE-safe later) with lmms-eval caption_to_words.
"""
import argparse, importlib.util, json, os
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration

SYS = ("A chat between a curious user and an artificial intelligence assistant. "
       "The assistant gives helpful, detailed, and polite answers to the user's questions.")
P_IMG = f"{SYS} USER: <image>\nPlease describe this image in detail. ASSISTANT:"
P_TXT = f"{SYS} USER: \nPlease describe this image in detail. ASSISTANT:"
def CU():
    p="/volume/exploration/EvolvingLMMs/lmms-eval/lmms_eval/tasks/coco_cap_chair/utils.py"
    s=importlib.util.spec_from_file_location("cu",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

ap=argparse.ArgumentParser()
ap.add_argument("--mode",default="gated")                 # label
ap.add_argument("--ban",action="store_true")
ap.add_argument("--soft",type=float,default=0.0)          # BETA (0=off)
ap.add_argument("--gcontrast",type=float,default=0.0)     # W (0=off)
ap.add_argument("--tau",type=float,default=0.0)
ap.add_argument("--glayer",type=int,default=31); ap.add_argument("--gtopk",type=int,default=10)
ap.add_argument("--syn",action="store_true",help="also gate synonyms' first-tokens (broader suppression)")
ap.add_argument("--n",type=int,default=120); ap.add_argument("--max_new_tokens",type=int,default=256)
ap.add_argument("--gpu",default="1"); ap.add_argument("--out",default="method/gd_out.jsonl")
a=ap.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"]=a.gpu
cu=CU()
proc=AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf");tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",torch_dtype=torch.float16,device_map="cuda:0").eval()
W=model.lm_head.weight;norm=model.model.language_model.norm;img_id=model.config.image_token_index;eos=tok.eos_token_id;DEV="cuda:0"
cal=json.load(open("/volume/exploration/EvolvingLMMs/detect/calibration.json"))["per_object_mean_grounding"]
cand=list(cal.keys())
objE=[];ftok={}
for o in cand:
    ids=tok.encode(" "+o,add_special_tokens=False) or tok.encode(o,add_special_tokens=False)
    objE.append(F.normalize(W[ids].float().mean(0),dim=-1));ftok[o]=ids[0]
objE=torch.stack(objE).to(DEV)

@torch.inference_mode()
def prep(image):
    vl=proc(images=image,text=P_IMG,return_tensors="pt").to(DEV,torch.float16)
    o=model(**vl,output_hidden_states=True,use_cache=True)
    vis=(vl.input_ids[0]==img_id).nonzero(as_tuple=True)[0]
    hn=F.normalize(norm(o.hidden_states[a.glayer][0,vis,:]).float(),dim=-1)
    cos=hn@objE.T
    g=cos.topk(min(a.gtopk,cos.shape[0]),dim=0).values.mean(0)            # (C,)
    gc={cand[i]:(g[i].item()-cal[cand[i]]) for i in range(len(cand))}
    pen=torch.zeros(W.shape[0],device=DEV); cmask=torch.zeros(W.shape[0],dtype=torch.bool,device=DEV)
    for o2 in cand:
        u=a.tau-gc[o2]
        if u>0:                                                          # flagged (ungrounded)
            t=ftok[o2]
            if a.ban: pen[t]=1e9
            if a.soft>0: pen[t]=max(pen[t].item(), a.soft*u)
            if a.gcontrast>0: cmask[t]=True
    return vl, o.past_key_values, o.logits[:,-1,:].float(), pen, cmask

@torch.inference_mode()
def gen(image):
    vl,vpk,lvl,pen,cmask=prep(image)
    if a.gcontrast>0:
        tt=tok(P_TXT,return_tensors="pt").to(DEV);to=model(input_ids=tt.input_ids,attention_mask=tt.attention_mask,use_cache=True)
        tpk=to.past_key_values; ltx=to.logits[:,-1,:].float()
    out=[]
    for _ in range(a.max_new_tokens):
        logits=lvl.clone()
        if a.gcontrast>0:
            contrasted=(1+a.gcontrast)*lvl - a.gcontrast*ltx
            logits=torch.where(cmask, contrasted, logits)
        logits=logits-pen
        nxt=int(logits.argmax())
        if nxt==eos: break
        out.append(nxt); t=torch.tensor([[nxt]],device=DEV)
        o=model(input_ids=t,past_key_values=vpk,use_cache=True);vpk=o.past_key_values;lvl=o.logits[:,-1,:].float()
        if a.gcontrast>0:
            o2=model(input_ids=t,past_key_values=tpk,use_cache=True);tpk=o2.past_key_values;ltx=o2.logits[:,-1,:].float()
    return tok.decode(out,skip_special_tokens=True).strip()

ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
fout=open(a.out,"w");res=[];n=min(a.n,len(ds))
print(f"[gd] mode={a.mode} ban={a.ban} soft={a.soft} gcontrast={a.gcontrast} tau={a.tau} n={n}",flush=True)
for i in range(n):
    cap=gen(ds[i]["image"].convert("RGB")); _,node,_,_=cu.caption_to_words(cap)
    res.append({"answer":list(ds[i]["gt_object"]),"pred":node});fout.write(json.dumps({"image_id":i,"caption":cap,"pred":node})+"\n")
    if (i+1)%30==0: print(f"[gd] {i+1}/{n}",flush=True)
fout.close()
cs=cu.coco_cap_chair_aggregate_results_chair_s(res);ci=cu.coco_cap_chair_aggregate_results_chair_i(res);rc=cu.coco_cap_chair_aggregate_results_recall(res)
al=sum(len(x["pred"]) for x in res)/len(res)
print(f"\n[gd] mode={a.mode} CHAIR_s={cs:.2f} CHAIR_i={ci:.2f} recall={rc:.2f} objs/cap={al:.2f} (base 53.3/15.6/80.2)",flush=True)
