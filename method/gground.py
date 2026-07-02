#!/usr/bin/env python3
"""
Grounding-guided SOFT suppression (single-pass) for LLaVA-1.5 (llava_hf). Training-free, per-image.
Fixes baked in: correct emitted-token ids (no leading space), tkB-scale per-object calibration,
synonym+irregular-plural coverage. SOFT (not hard-ban) so confident/real objects survive -> recall preserved.

Per image at prefill: g(o)=topk-mean cosine(norm(visual_hidden_L), unembed(o)); gc(o)=g(o)-cal[o].
Build vocab penalty: for each candidate object o, for each of its surface-form tokens t:
    pen[t] = max(pen[t], beta * relu(tau - gc(o)))      (proportional to ungroundedness; 0 if grounded)
Greedy decode with logits -= pen. (--hard makes it a hard ban instead, for comparison.)
"""
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
ap.add_argument("--tau",type=float,default=-0.015); ap.add_argument("--beta",type=float,default=4.0)
ap.add_argument("--hard",action="store_true")
ap.add_argument("--glayer",type=int,default=31); ap.add_argument("--gtopk",type=int,default=10)
ap.add_argument("--n",type=int,default=120); ap.add_argument("--max_new_tokens",type=int,default=256)
ap.add_argument("--split",default="all",choices=["all","test","cal"],help="test=odd img idx, cal=even (matches conformal split)")
ap.add_argument("--gpu",default="1"); ap.add_argument("--out",default="method/gg_out.jsonl"); ap.add_argument("--label",default="gground")
ap.add_argument("--protect",default="",help="逗号分隔canonical object,永不抑制(base-rate-aware)")
a=ap.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"]=a.gpu
PROTECT=set(x.strip() for x in a.protect.split(",") if x.strip())
cu=CU()
proc=AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf");tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",torch_dtype=torch.float16,device_map="cuda:0").eval()
W=model.lm_head.weight;norm=model.model.language_model.norm;img_id=model.config.image_token_index;eos=tok.eos_token_id;DEV="cuda:0"
cal=json.load(open("/volume/exploration/EvolvingLMMs/detect/calibration_tkb.json"));cand=list(cal.keys())
objE=torch.stack([F.normalize(W[tok.encode(o,add_special_tokens=False)].float().mean(0),dim=-1) for o in cand]).to(DEV)
IRREG={"man":"men","woman":"women","person":"people","child":"children","foot":"feet","tooth":"teeth"}
canon2ftoks=defaultdict(set)
for w,canon in cu.INVERSE_SYNONYM_DICT.items():
    forms=[w,w+"s",w.rstrip("s"),w.capitalize()]+([IRREG[w],IRREG[w].capitalize()] if w in IRREG else [])
    for s in forms:
        ids=tok.encode(s,add_special_tokens=False)
        if ids: canon2ftoks[canon].add(ids[0])
@torch.inference_mode()
def run(image):
    vl=proc(images=image,text=P_IMG,return_tensors="pt").to(DEV,torch.float16)
    o=model(**vl,output_hidden_states=True,use_cache=True)
    vis=(vl.input_ids[0]==img_id).nonzero(as_tuple=True)[0]
    hn=F.normalize(norm(o.hidden_states[a.glayer][0,vis,:]).float(),dim=-1)
    g=(hn@objE.T).topk(min(a.gtopk,hn.shape[0]),dim=0).values.mean(0)
    pen=torch.zeros(W.shape[0],device=DEV)
    for i,oo in enumerate(cand):
        if oo in PROTECT: continue
        gc=g[i].item()-cal[oo]; u=a.tau-gc
        if u>0:
            val=1e9 if a.hard else a.beta*u
            for t in canon2ftoks.get(oo,()): pen[t]=max(pen[t].item(),val)
    pk=o.past_key_values;lg=o.logits[:,-1,:].float();out=[]
    for _ in range(a.max_new_tokens):
        nxt=int((lg-pen).argmax())
        if nxt==eos: break
        out.append(nxt);t=torch.tensor([[nxt]],device=DEV)
        o2=model(input_ids=t,past_key_values=pk,use_cache=True);pk=o2.past_key_values;lg=o2.logits[:,-1,:].float()
    return tok.decode(out,skip_special_tokens=True).strip()
ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
idxs=[i for i in range(len(ds)) if a.split=="all" or (a.split=="test" and i%2==1) or (a.split=="cal" and i%2==0)][:a.n]
fout=open(a.out,"w");res=[];n=len(idxs)
print(f"[gg] {a.label} tau={a.tau} beta={a.beta} hard={a.hard} split={a.split} n={n}",flush=True)
for c,i in enumerate(idxs):
    cap=run(ds[i]["image"].convert("RGB"));_,node,_,_=cu.caption_to_words(cap)
    res.append({"answer":list(ds[i]["gt_object"]),"pred":node});fout.write(json.dumps({"image_id":i,"caption":cap,"pred":node})+"\n")
    if (c+1)%30==0:print(f"[gg] {c+1}/{n}",flush=True)
fout.close()
cs=cu.coco_cap_chair_aggregate_results_chair_s(res);ci=cu.coco_cap_chair_aggregate_results_chair_i(res);rc=cu.coco_cap_chair_aggregate_results_recall(res)
print(f"\n[gg] {a.label} CHAIR_s={cs:.2f} CHAIR_i={ci:.2f} recall={rc:.2f} objs={sum(len(x['pred']) for x in res)/len(res):.2f} (base 53.3/15.6/80.2)",flush=True)
