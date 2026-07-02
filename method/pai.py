#!/usr/bin/env python3
"""
PAI-style attention intervention (+ optional grounding gating = OUR novelty) for LLaVA-1.5 (llava_hf, tf4.57).
Training-free. Eq.3: at deep layers, boost each query's PRE-softmax attention to IMAGE tokens by alpha*|A|.
Eq.4 (optional logit refine, gamma): logits = gamma*vl - (gamma-1)*txt   (gentle text-only CFG, gamma~1.1).

Novelty knob --ggate: scale alpha per IMAGE-TOKEN by how much that token grounds SOME object
(grounding saliency from our logit-lens), focusing the boost on grounded patches rather than uniform.
"""
import argparse, importlib.util, json, os, sys
# MUST set CUDA_VISIBLE_DEVICES BEFORE importing torch (else it binds to physical GPU0).
if "--gpu" in sys.argv: os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[sys.argv.index("--gpu")+1]
import torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
import transformers.models.llama.modeling_llama as LM

SYS=("A chat between a curious user and an artificial intelligence assistant. "
     "The assistant gives helpful, detailed, and polite answers to the user's questions.")
P_IMG=f"{SYS} USER: <image>\nPlease describe this image in detail. ASSISTANT:"
P_TXT=f"{SYS} USER: \nPlease describe this image in detail. ASSISTANT:"
def CU():
    p="/volume/exploration/EvolvingLMMs/lmms-eval/lmms_eval/tasks/coco_cap_chair/utils.py"
    s=importlib.util.spec_from_file_location("cu",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

ap=argparse.ArgumentParser()
ap.add_argument("--alpha",type=float,default=0.5); ap.add_argument("--gamma",type=float,default=1.0)
ap.add_argument("--locore",type=int,default=0,help="reproduce LVLMs-Saliency LocoRE: boost last-query attn to the K most-recent predecessor tokens (0=off => PAI image-token boost)")
ap.add_argument("--layer_min",type=int,default=2); ap.add_argument("--layer_max",type=int,default=32)
ap.add_argument("--ggate",action="store_true",help="grounding-gated: scale boost per image token by grounding saliency")
ap.add_argument("--rbonus",type=float,default=0.0,help="grounding recall-recovery: +rbonus*gc(o) logit bonus to well-grounded objects' tokens")
ap.add_argument("--soft",type=float,default=0.0,help="grounding soft-suppression: -soft*relu(stau-gc) penalty on ungrounded objects' tokens")
ap.add_argument("--stau",type=float,default=-0.012,help="suppression flag threshold (gc<stau)")
ap.add_argument("--n",type=int,default=120); ap.add_argument("--max_new_tokens",type=int,default=256)
ap.add_argument("--split",default="all",choices=["all","test","cal"])
ap.add_argument("--gpu",default="1"); ap.add_argument("--out",default="method/pai_out.jsonl"); ap.add_argument("--label",default="pai")
a=ap.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"]=a.gpu
cu=CU()
proc=AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf");tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",torch_dtype=torch.float16,device_map="cuda:0",attn_implementation="eager").eval()
img_id=model.config.image_token_index;eos=tok.eos_token_id;DEV="cuda:0"
W=model.lm_head.weight;norm=model.model.language_model.norm

# global PAI state used by the patched attention
S={"on":False,"lo":0,"hi":0,"alpha":a.alpha,"lmin":a.layer_min,"lmax":a.layer_max,"sal":None,"locore":a.locore}
_orig=LM.eager_attention_forward
def pai_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    key_states=LM.repeat_kv(key, module.num_key_value_groups)
    value_states=LM.repeat_kv(value, module.num_key_value_groups)
    aw=torch.matmul(query, key_states.transpose(2,3))*scaling
    if attention_mask is not None:
        aw=aw+attention_mask[:,:,:,:key_states.shape[-2]]
    li=getattr(module,"layer_idx",0)
    if S["on"] and S["lmin"]<=li<S["lmax"]:
        if S.get("locore",0)>0:
            # LocoRE reproduction: boost last-query attn to the K most-recent PREDECESSOR tokens (text coherence)
            kv=aw.shape[-1]; K=S["locore"]; a0=max(S["hi"],kv-1-K); a1=kv-1   # recent predecessors (after image block)
            if a1>a0:
                seg=aw[...,-1:,a0:a1]; aw[...,-1:,a0:a1]=seg+S["alpha"]*seg.abs()
        elif S["hi"]>S["lo"]:
            # PAI: only the LAST (currently-generating) query row attends-more-to-image; never touch prefill image rows
            seg=aw[...,-1:,S["lo"]:S["hi"]]
            boost=S["alpha"]*seg.abs()
            if S["sal"] is not None: boost=boost*S["sal"]  # grounding-gated per-image-token scaling
            aw[...,-1:,S["lo"]:S["hi"]]=seg+boost
    aw=F.softmax(aw,dim=-1,dtype=torch.float32).to(query.dtype)
    aw=F.dropout(aw,p=dropout,training=module.training)
    out=torch.matmul(aw,value_states).transpose(1,2).contiguous()
    return out, aw
LM.eager_attention_forward=pai_eager

cal=json.load(open("/volume/exploration/EvolvingLMMs/detect/calibration_tkb.json"));cobj=list(cal.keys())
objE=torch.stack([F.normalize(W[tok.encode(o,add_special_tokens=False)].float().mean(0),dim=-1) for o in cobj]).to(DEV)
from collections import defaultdict
IRREG={"man":"men","woman":"women","person":"people","child":"children","foot":"feet","tooth":"teeth"}
canon2ftoks=defaultdict(set)
for w,canon in cu.INVERSE_SYNONYM_DICT.items():
    for s in [w,w+"s",w.rstrip("s"),w.capitalize()]+([IRREG[w],IRREG[w].capitalize()] if w in IRREG else []):
        ids=tok.encode(s,add_special_tokens=False)
        if ids: canon2ftoks[canon].add(ids[0])

@torch.inference_mode()
def saliency(vl,vis):
    """per-image-token grounding saliency: max over objects of cosine(token, object) -> [1,1,1,n_img]."""
    o=model(**vl,output_hidden_states=True,use_cache=False)
    hn=F.normalize(norm(o.hidden_states[31][0,vis,:]).float(),dim=-1)
    s=(hn@objE.T).max(dim=1).values                      # (n_img,) how strongly each patch points to some object
    s=(s-s.min())/(s.max()-s.min()+1e-6)                 # normalize 0..1
    return s.to(torch.float16).view(1,1,1,-1)

@torch.inference_mode()
def gen(image):
    vl=proc(images=image,text=P_IMG,return_tensors="pt").to(DEV,torch.float16)
    ids=vl.input_ids[0];vis=(ids==img_id).nonzero(as_tuple=True)[0]
    S["lo"]=int(vis.min());S["hi"]=int(vis.max())+1
    bonus=None; pen=None
    if a.ggate or a.rbonus>0 or a.soft>0:                      # one grounding forward -> saliency / recall-bonus / suppression
        o0=model(**vl,output_hidden_states=True,use_cache=False)
        hn=F.normalize(norm(o0.hidden_states[31][0,vis,:]).float(),dim=-1)
        cos=hn@objE.T                                          # (n_img, C)
        if a.ggate:
            s=cos.max(dim=1).values; s=(s-s.min())/(s.max()-s.min()+1e-6); S["sal"]=s.to(torch.float16).view(1,1,1,-1)
        else: S["sal"]=None
        if a.rbonus>0 or a.soft>0:
            g=cos.topk(min(10,cos.shape[0]),dim=0).values.mean(0)   # (C,) per-object grounding
            if a.rbonus>0: bonus=torch.zeros(W.shape[0],device=DEV)
            if a.soft>0: pen=torch.zeros(W.shape[0],device=DEV)
            for i,oo in enumerate(cobj):
                gc=g[i].item()-cal[oo]
                if a.rbonus>0 and gc>0:
                    for t in canon2ftoks.get(oo,()): bonus[t]=max(bonus[t].item(), a.rbonus*gc)
                if a.soft>0 and gc<a.stau:                      # ungrounded -> suppress its tokens
                    for t in canon2ftoks.get(oo,()): pen[t]=max(pen[t].item(), a.soft*(a.stau-gc))
    else:
        S["sal"]=None
    if a.gamma>1: tt=tok(P_TXT,return_tensors="pt").to(DEV)
    S["on"]=True
    o=model(**vl,use_cache=True);pk=o.past_key_values;lvl=o.logits[:,-1,:].float()
    if a.gamma>1:
        S["on"]=False  # text-only branch: no image, no boost
        ot=model(input_ids=tt.input_ids,attention_mask=tt.attention_mask,use_cache=True);tpk=ot.past_key_values;ltx=ot.logits[:,-1,:].float()
        S["on"]=True
    out=[]
    for _ in range(a.max_new_tokens):
        logits=a.gamma*lvl-(a.gamma-1)*ltx if a.gamma>1 else lvl
        if bonus is not None: logits=logits+bonus
        if pen is not None: logits=logits-pen
        nxt=int(logits.argmax())
        if nxt==eos: break
        out.append(nxt);t=torch.tensor([[nxt]],device=DEV)
        o=model(input_ids=t,past_key_values=pk,use_cache=True);pk=o.past_key_values;lvl=o.logits[:,-1,:].float()
        if a.gamma>1:
            S["on"]=False;ot=model(input_ids=t,past_key_values=tpk,use_cache=True);tpk=ot.past_key_values;ltx=ot.logits[:,-1,:].float();S["on"]=True
    S["on"]=False
    return tok.decode(out,skip_special_tokens=True).strip()

ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
_sp=getattr(a,"split","all")
idxs=[i for i in range(len(ds)) if _sp=="all" or (_sp=="test" and i%2==1) or (_sp=="cal" and i%2==0)][:a.n]
fout=open(a.out,"w");res=[];n=len(idxs)
print(f"[pai] {a.label} alpha={a.alpha} gamma={a.gamma} locore={a.locore} split={_sp} n={n}",flush=True)
for c,i in enumerate(idxs):
    cap=gen(ds[i]["image"].convert("RGB"));_,node,_,_=cu.caption_to_words(cap)
    res.append({"answer":list(ds[i]["gt_object"]),"pred":node});fout.write(json.dumps({"image_id":i,"caption":cap,"pred":node})+"\n")
    if (c+1)%30==0:print(f"[pai] {c+1}/{n}",flush=True)
fout.close()
cs=cu.coco_cap_chair_aggregate_results_chair_s(res);ci=cu.coco_cap_chair_aggregate_results_chair_i(res);rc=cu.coco_cap_chair_aggregate_results_recall(res)
print(f"\n[pai] {a.label} CHAIR_s={cs:.2f} CHAIR_i={ci:.2f} recall={rc:.2f} objs={sum(len(x['pred']) for x in res)/len(res):.2f} (base 53.3/15.6/80.2)",flush=True)
