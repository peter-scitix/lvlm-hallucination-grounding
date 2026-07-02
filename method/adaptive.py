#!/usr/bin/env python3
"""
Detector-driven CONTINUOUS-adaptive method (OUR novel method) for LLaVA-1.5 (llava_hf, tf4.57). Training-free.

Pass 1: baseline caption (no intervention) + per-object logit-lens grounding gc(o)=g(o)-cal[o].
Risk:   risk_n = # MENTIONED objects with gc<tau  (how many hallucinated objects the model actually produced).
        alpha_img = alpha_max * min(1, risk_n / K)   -> CLEAN image (risk_n=0) => alpha=0 (keep baseline, full
        recall); DIRTY image => strong attention boost. (continuous, vs the failed binary gate that treated ~80%)
Pass 2 (only if alpha_img>0): regenerate with per-image alpha_img attention boost + grounding SOFT-suppression
        (penalty -soft*relu(stau-gc) on ungrounded objects' synonym tokens). Clean images keep pass-1 caption.
"""
import argparse, importlib.util, json, os, sys
if "--gpu" in sys.argv: os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[sys.argv.index("--gpu")+1]
import torch, torch.nn.functional as F
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
import transformers.models.llama.modeling_llama as LM
SYS=("A chat between a curious user and an artificial intelligence assistant. "
     "The assistant gives helpful, detailed, and polite answers to the user's questions.")
P_IMG=f"{SYS} USER: <image>\nPlease describe this image in detail. ASSISTANT:"
def CU():
    p="/volume/exploration/EvolvingLMMs/lmms-eval/lmms_eval/tasks/coco_cap_chair/utils.py"
    s=importlib.util.spec_from_file_location("cu",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ap=argparse.ArgumentParser()
ap.add_argument("--alpha_max",type=float,default=0.6); ap.add_argument("--K",type=float,default=3.0)
ap.add_argument("--tau",type=float,default=-0.012); ap.add_argument("--soft",type=float,default=15.0); ap.add_argument("--stau",type=float,default=-0.010)
ap.add_argument("--layer_min",type=int,default=2); ap.add_argument("--glayer",type=int,default=31); ap.add_argument("--gtopk",type=int,default=10)
ap.add_argument("--n",type=int,default=120); ap.add_argument("--max_new_tokens",type=int,default=256)
ap.add_argument("--gpu",default="1"); ap.add_argument("--out",default="method/adapt_out.jsonl"); ap.add_argument("--label",default="adapt")
a=ap.parse_args(); cu=CU()
proc=AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf");tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",torch_dtype=torch.float16,device_map="cuda:0",attn_implementation="eager").eval()
img_id=model.config.image_token_index;eos=tok.eos_token_id;DEV="cuda:0";W=model.lm_head.weight;norm=model.model.language_model.norm
S={"on":False,"lo":0,"hi":0,"alpha":0.0,"lmin":a.layer_min}
def pai_eager(module,query,key,value,attention_mask,scaling,dropout=0.0,**kw):
    ks=LM.repeat_kv(key,module.num_key_value_groups);vs=LM.repeat_kv(value,module.num_key_value_groups)
    aw=torch.matmul(query,ks.transpose(2,3))*scaling
    if attention_mask is not None: aw=aw+attention_mask[:,:,:,:ks.shape[-2]]
    if S["on"] and S["alpha"]>0 and getattr(module,"layer_idx",0)>=S["lmin"] and S["hi"]>S["lo"]:
        seg=aw[...,-1:,S["lo"]:S["hi"]]; aw[...,-1:,S["lo"]:S["hi"]]=seg+S["alpha"]*seg.abs()
    aw=F.softmax(aw,dim=-1,dtype=torch.float32).to(query.dtype);aw=F.dropout(aw,p=dropout,training=module.training)
    return torch.matmul(aw,vs).transpose(1,2).contiguous(),aw
LM.eager_attention_forward=pai_eager
cal=json.load(open("/volume/exploration/EvolvingLMMs/detect/calibration_tkb.json"));cobj=list(cal.keys())
objE=torch.stack([F.normalize(W[tok.encode(o,add_special_tokens=False)].float().mean(0),dim=-1) for o in cobj]).to(DEV)
IRREG={"man":"men","woman":"women","person":"people","child":"children","foot":"feet","tooth":"teeth"}
canon2ftoks=defaultdict(set)
for w,canon in cu.INVERSE_SYNONYM_DICT.items():
    for s in [w,w+"s",w.rstrip("s"),w.capitalize()]+([IRREG[w],IRREG[w].capitalize()] if w in IRREG else []):
        ids=tok.encode(s,add_special_tokens=False)
        if ids: canon2ftoks[canon].add(ids[0])
@torch.inference_mode()
def decode(pk,lg,pen=None):
    out=[]
    for _ in range(a.max_new_tokens):
        logits=lg if pen is None else lg-pen
        nxt=int(logits.argmax())
        if nxt==eos: break
        out.append(nxt);t=torch.tensor([[nxt]],device=DEV)
        o=model(input_ids=t,past_key_values=pk,use_cache=True);pk=o.past_key_values;lg=o.logits[:,-1,:].float()
    return tok.decode(out,skip_special_tokens=True).strip()
@torch.inference_mode()
def run(image):
    vl=proc(images=image,text=P_IMG,return_tensors="pt").to(DEV,torch.float16)
    vis=(vl.input_ids[0]==img_id).nonzero(as_tuple=True)[0];S["lo"]=int(vis.min());S["hi"]=int(vis.max())+1
    S["on"]=False;S["alpha"]=0.0
    o=model(**vl,output_hidden_states=True,use_cache=True)
    hn=F.normalize(norm(o.hidden_states[a.glayer][0,vis,:]).float(),dim=-1)
    g=(hn@objE.T).topk(min(a.gtopk,hn.shape[0]),dim=0).values.mean(0)
    gc={cobj[i]:g[i].item()-cal[cobj[i]] for i in range(len(cobj))}
    cap1=decode(o.past_key_values,o.logits[:,-1,:].float())
    _,node1,_,_=cu.caption_to_words(cap1)
    risk_n=sum(1 for o2 in dict.fromkeys(node1) if gc.get(o2,9)<a.tau)
    alpha_img=a.alpha_max*min(1.0,risk_n/a.K)
    if alpha_img<=0:                                       # clean image: keep baseline (full recall)
        _,node,_,_=cu.caption_to_words(cap1); return cap1,node,0.0
    pen=torch.zeros(W.shape[0],device=DEV)                 # soft-suppression penalty for pass2
    for i,oo in enumerate(cobj):
        u=a.stau-gc[oo]
        if u>0:
            for t in canon2ftoks.get(oo,()): pen[t]=max(pen[t].item(), a.soft*u)
    S["on"]=True;S["alpha"]=alpha_img
    o2=model(**vl,use_cache=True)
    cap=decode(o2.past_key_values,o2.logits[:,-1,:].float(),pen=pen)
    S["on"]=False
    _,node,_,_=cu.caption_to_words(cap); return cap,node,alpha_img
ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
fout=open(a.out,"w");res=[];n=min(a.n,len(ds));ntreat=0;asum=0.0
print(f"[ad] {a.label} alpha_max={a.alpha_max} K={a.K} tau={a.tau} soft={a.soft} n={n}",flush=True)
for i in range(n):
    cap,node,ai=run(ds[i]["image"].convert("RGB"));ntreat+=(ai>0);asum+=ai
    res.append({"answer":list(ds[i]["gt_object"]),"pred":node});fout.write(json.dumps({"image_id":i,"caption":cap,"pred":node,"alpha":round(ai,3)})+"\n")
    if (i+1)%30==0:print(f"[ad] {i+1}/{n} treated={ntreat}",flush=True)
fout.close()
cs=cu.coco_cap_chair_aggregate_results_chair_s(res);ci=cu.coco_cap_chair_aggregate_results_chair_i(res);rc=cu.coco_cap_chair_aggregate_results_recall(res)
print(f"\n[ad] {a.label} CHAIR_s={cs:.2f} CHAIR_i={ci:.2f} recall={rc:.2f} objs={sum(len(x['pred']) for x in res)/len(res):.2f} treated={ntreat}/{n} avg_alpha={asum/n:.3f} (base 50.4/15.4/76.9 @mnt512; PAI-frontier)",flush=True)
