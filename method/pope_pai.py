#!/usr/bin/env python3
"""Faithful PAI on the STANDARD POPE protocol (fair comparison).
Our validated standard POPE prompt ("Answer ... single word or phrase", short decode) — baseline ~0.84 —
plus PAI's TWO components run faithfully: Eq.3 attention boost to image tokens (alpha, layers 2-32, last-query row)
and Eq.4 logit CFG (gamma): logits = gamma*full - (gamma-1)*text_only. README POPE config: alpha=0.2, gamma=1.1.
Rationale: PAI's OFFICIAL pope_eval.py omits the yes/no instruction + decodes 512 tokens -> LLaVA-1.5 over-affirms
(baseline degenerates to ~0.50, 99% yes). That is a prompt artifact affecting baseline AND PAI equally, so the
fair faithful PAI POPE number is on the standard prompt. This reuses the exact attention monkeypatch validated in
method/pope_eval.py (baseline 0.840/0.868/0.870)."""
import argparse, os, sys, re
if "--gpu" in sys.argv: os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[sys.argv.index("--gpu")+1]
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
import transformers.models.llama.modeling_llama as LM
SYS=("A chat between a curious user and an artificial intelligence assistant. "
     "The assistant gives helpful, detailed, and polite answers to the user's questions.")
ap=argparse.ArgumentParser()
ap.add_argument("--alpha",type=float,default=0.2); ap.add_argument("--gamma",type=float,default=1.1)
ap.add_argument("--layer_min",type=int,default=2); ap.add_argument("--layer_max",type=int,default=32)
ap.add_argument("--n",type=int,default=500); ap.add_argument("--gpu",default="1")
ap.add_argument("--splits",default="adversarial,popular,random")
ap.add_argument("--label",default="pai")
a=ap.parse_args()
proc=AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf");tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",torch_dtype=torch.float16,device_map="cuda:0",attn_implementation="eager").eval()
img_id=model.config.image_token_index;eos=tok.eos_token_id;DEV="cuda:0"
S={"on":False,"lo":0,"hi":0,"alpha":a.alpha,"lmin":a.layer_min,"lmax":a.layer_max}
def pai_eager(module,query,key,value,attention_mask,scaling,dropout=0.0,**kw):
    ks=LM.repeat_kv(key,module.num_key_value_groups);vs=LM.repeat_kv(value,module.num_key_value_groups)
    aw=torch.matmul(query,ks.transpose(2,3))*scaling
    if attention_mask is not None: aw=aw+attention_mask[:,:,:,:ks.shape[-2]]
    li=getattr(module,"layer_idx",0)
    if S["on"] and S["lmin"]<=li<S["lmax"] and S["hi"]>S["lo"]:
        seg=aw[...,-1:,S["lo"]:S["hi"]]; aw[...,-1:,S["lo"]:S["hi"]]=seg+S["alpha"]*seg.abs()
    aw=F.softmax(aw,dim=-1,dtype=torch.float32).to(query.dtype);aw=F.dropout(aw,p=dropout,training=module.training)
    return torch.matmul(aw,vs).transpose(1,2).contiguous(),aw
LM.eager_attention_forward=pai_eager

@torch.inference_mode()
def answer(image,question):
    prompt=f"{SYS} USER: <image>\n{question}\nAnswer the question using a single word or phrase. ASSISTANT:"
    inp=proc(images=image,text=prompt,return_tensors="pt").to(DEV,torch.float16)
    ids=inp.input_ids[0];vis=(ids==img_id).nonzero(as_tuple=True)[0];S["lo"]=int(vis.min());S["hi"]=int(vis.max())+1
    use_attn=a.alpha>0; use_cfg=a.gamma>1
    if not use_attn and not use_cfg:                          # plain baseline
        out=model.generate(**inp,max_new_tokens=5,do_sample=False,num_beams=1)
        return tok.decode(out[0,inp.input_ids.shape[1]:],skip_special_tokens=True).strip().lower()
    # manual greedy decode (needed for CFG); attention boost via S["on"]
    if use_cfg:
        # text-only branch: SAME textual prompt, NO image (CFG negative prompt = question without vision)
        tp=f"{SYS} USER: \n{question}\nAnswer the question using a single word or phrase. ASSISTANT:"
        tt=proc(text=tp,return_tensors="pt").to(DEV)
    S["on"]=use_attn
    o=model(**inp,use_cache=True);pk=o.past_key_values;lvl=o.logits[:,-1,:].float()
    if use_cfg:
        S["on"]=False;ot=model(input_ids=tt.input_ids,attention_mask=tt.attention_mask,use_cache=True);tpk=ot.past_key_values;ltx=ot.logits[:,-1,:].float();S["on"]=use_attn
    out=[]
    for _ in range(5):
        logits=a.gamma*lvl-(a.gamma-1)*ltx if use_cfg else lvl
        nxt=int(logits.argmax())
        if nxt==eos: break
        out.append(nxt);t=torch.tensor([[nxt]],device=DEV)
        o=model(input_ids=t,past_key_values=pk,use_cache=True);pk=o.past_key_values;lvl=o.logits[:,-1,:].float()
        if use_cfg:
            S["on"]=False;ot=model(input_ids=t,past_key_values=tpk,use_cache=True);tpk=ot.past_key_values;ltx=ot.logits[:,-1,:].float();S["on"]=use_attn
    S["on"]=False
    return tok.decode(out,skip_special_tokens=True).strip().lower()

def yn(t):
    t=t.lower()
    if t.startswith("yes") or re.search(r"\byes\b",t): return "yes"
    if t.startswith("no") or re.search(r"\bno\b",t): return "no"
    return "no"
print(f"[pope-pai] {a.label} alpha={a.alpha} gamma={a.gamma} layers=[{a.layer_min},{a.layer_max}) n/split={a.n}",flush=True)
ALL=load_dataset("lmms-lab/POPE",split="test",token=False)
for split in a.splits.split(","):
    ds=ALL.filter(lambda x:x["category"]==split)
    n=min(a.n,len(ds));tp=fp=tn=fn=0;correct=0;yes=0
    for i in range(n):
        ex=ds[i];gt=ex["answer"].strip().lower();pred=yn(answer(ex["image"].convert("RGB"),ex["question"]))
        correct+=(pred==gt);yes+=(pred=="yes")
        if gt=="yes" and pred=="yes": tp+=1
        elif gt=="yes" and pred=="no": fn+=1
        elif gt=="no" and pred=="yes": fp+=1
        else: tn+=1
    acc=correct/n; prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); f1=2*prec*rec/max(prec+rec,1e-9)
    print(f"[pope-pai] {a.label} {split:12} acc={acc:.4f} f1={f1:.4f} prec={prec:.4f} rec={rec:.4f} yes_rate={yes/n:.3f} (n={n})",flush=True)
