#!/usr/bin/env python3
"""
Grounding-guided TWO-PASS elimination (our method) for LLaVA-1.5 (llava_hf). Training-free, per-image.

Pass 1: greedy caption.
Detect: extract MENTIONED COCO objects (caption_to_words); compute each one's calibrated logit-lens
        grounding gc(o)=g(o)-cal[o] (our AUROC-0.82 detector); FLAG those with gc(o) < tau.
        (diagnosis: lowest-grounding mentioned object is hallucinated 71% of the time; within-img AUROC 0.80)
Pass 2: regenerate greedily with ALL surface forms (synonyms + plurals) of flagged objects banned
        (synonym-aware via INVERSE_SYNONYM_DICT — fixes the canonical-only ban that escaped via synonyms).
Compares CHAIR_s/i/recall/objs to baseline.  --bottomk K  alternatively flags the K lowest-grounding mentioned.
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
ap.add_argument("--tau",type=float,default=0.0)
ap.add_argument("--bottomk",type=int,default=0,help="if>0, flag K lowest-grounding mentioned objs instead of tau")
ap.add_argument("--glayer",type=int,default=31); ap.add_argument("--gtopk",type=int,default=10)
ap.add_argument("--n",type=int,default=120); ap.add_argument("--max_new_tokens",type=int,default=256)
ap.add_argument("--gpu",default="1"); ap.add_argument("--out",default="method/tp_out.jsonl")
ap.add_argument("--label",default="twopass")
a=ap.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"]=a.gpu
cu=CU()
proc=AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf");tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",torch_dtype=torch.float16,device_map="cuda:0").eval()
W=model.lm_head.weight;norm=model.model.language_model.norm;img_id=model.config.image_token_index;eos=tok.eos_token_id;DEV="cuda:0"
cal=json.load(open("/volume/exploration/EvolvingLMMs/detect/calibration_tkb.json"))  # tkB-scale per-object mean
cand=list(cal.keys())
# NOTE: encode(word) WITHOUT leading space -> the merged "▁word" token the model actually emits.
# encode(" word") wrongly prepends a standalone "▁" (id 29871); do NOT use that.
objE=torch.stack([F.normalize(W[tok.encode(o,add_special_tokens=False)].float().mean(0),dim=-1) for o in cand]).to(DEV)
# canonical -> all surface-form emitted-token ids (synonym + plural + irregular + capitalized aware)
IRREG={"man":"men","woman":"women","person":"people","child":"children","foot":"feet","tooth":"teeth","mouse":"mice","goose":"geese"}
canon2ftoks=defaultdict(set)
for w,canon in cu.INVERSE_SYNONYM_DICT.items():
    forms=[w, w+"s", w.rstrip("s"), w.capitalize()]
    if w in IRREG: forms+=[IRREG[w], IRREG[w].capitalize()]
    for surf in forms:
        ids=tok.encode(surf,add_special_tokens=False)
        if ids: canon2ftoks[canon].add(ids[0])

@torch.inference_mode()
def prefill(image):
    vl=proc(images=image,text=P_IMG,return_tensors="pt").to(DEV,torch.float16)
    o=model(**vl,output_hidden_states=True,use_cache=True)
    vis=(vl.input_ids[0]==img_id).nonzero(as_tuple=True)[0]
    hn=F.normalize(norm(o.hidden_states[a.glayer][0,vis,:]).float(),dim=-1)
    g=(hn@objE.T).topk(min(a.gtopk,hn.shape[0]),dim=0).values.mean(0)
    graw={cand[i]:g[i].item() for i in range(len(cand))}                 # raw grounding (consistent scale)
    gc={cand[i]:(g[i].item()-cal[cand[i]]) for i in range(len(cand))}    # calibrated (note: cal is combined-scale)
    return vl,o.past_key_values,o.logits[:,-1,:].float(),graw,gc

@torch.inference_mode()
def decode(vl,pk,lg,ban):
    banv=torch.zeros(W.shape[0],device=DEV)
    for t in ban: banv[t]=1e9
    out=[]
    for _ in range(a.max_new_tokens):
        nxt=int((lg-banv).argmax())
        if nxt==eos: break
        out.append(nxt);t=torch.tensor([[nxt]],device=DEV)
        o=model(input_ids=t,past_key_values=pk,use_cache=True);pk=o.past_key_values;lg=o.logits[:,-1,:].float()
    return tok.decode(out,skip_special_tokens=True).strip()

ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
fout=open(a.out,"w");base_res=[];tp_res=[];n=min(a.n,len(ds));n_flag=0
print(f"[tp] {a.label} tau={a.tau} bottomk={a.bottomk} n={n}",flush=True)
for i in range(n):
    img=ds[i]["image"].convert("RGB")
    vl,pk,lg,graw,gc=prefill(img)
    cap1=decode(vl,pk,lg,set())                          # pass1 (need fresh cache for pass2 -> re-prefill)
    _,node1,_,_=cu.caption_to_words(cap1)
    mentioned=list(dict.fromkeys(node1))
    if a.bottomk>0:                                       # flag K lowest RAW-grounding mentioned objs
        flagged=[o for o,_ in sorted(((o,graw.get(o,9)) for o in mentioned),key=lambda x:x[1])[:a.bottomk]]
    else:                                                 # flag mentioned objs with CALIBRATED grounding gc < tau
        flagged=[o for o in mentioned if gc.get(o,9)<a.tau]
    n_flag+=len(flagged)
    ban=set();[ban.update(canon2ftoks.get(o,set())) for o in flagged]
    vl2,pk2,lg2,_,_=prefill(img)                          # fresh cache
    cap2=decode(vl2,pk2,lg2,ban)
    _,node2,_,_=cu.caption_to_words(cap2)
    base_res.append({"answer":list(ds[i]["gt_object"]),"pred":node1})
    tp_res.append({"answer":list(ds[i]["gt_object"]),"pred":node2})
    fout.write(json.dumps({"image_id":i,"cap1":cap1,"cap2":cap2,"flagged":flagged})+"\n")
    if (i+1)%30==0: print(f"[tp] {i+1}/{n}",flush=True)
fout.close()
def rep(tag,r):
    return f"{tag}: CHAIR_s={cu.coco_cap_chair_aggregate_results_chair_s(r):.2f} CHAIR_i={cu.coco_cap_chair_aggregate_results_chair_i(r):.2f} recall={cu.coco_cap_chair_aggregate_results_recall(r):.2f} objs={sum(len(x['pred']) for x in r)/len(r):.2f}"
print("\n"+rep("PASS1(base)",base_res))
print(rep(f"TWOPASS",tp_res))
print(f"flagged/img={n_flag/n:.2f}",flush=True)
