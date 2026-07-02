#!/usr/bin/env python3
"""POPE no-regression check for our method (PAI attention; soft-suppression is inactive on yes/no answers).
Eval LLaVA-1.5-7B on lmms-lab/POPE 3 splits, yes/no accuracy, baseline (alpha=0) vs ours (alpha)."""
import argparse, os, sys, re
if "--gpu" in sys.argv: os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[sys.argv.index("--gpu")+1]
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
import transformers.models.llama.modeling_llama as LM
SYS=("A chat between a curious user and an artificial intelligence assistant. "
     "The assistant gives helpful, detailed, and polite answers to the user's questions.")
ap=argparse.ArgumentParser()
ap.add_argument("--alpha",type=float,default=0.0); ap.add_argument("--layer_min",type=int,default=2)
ap.add_argument("--n",type=int,default=500); ap.add_argument("--gpu",default="1")
a=ap.parse_args()
proc=AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf");tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",torch_dtype=torch.float16,device_map="cuda:0",attn_implementation="eager").eval()
img_id=model.config.image_token_index;eos=tok.eos_token_id;DEV="cuda:0"
S={"on":a.alpha>0,"lo":0,"hi":0,"alpha":a.alpha,"lmin":a.layer_min}
def pai_eager(module,query,key,value,attention_mask,scaling,dropout=0.0,**kw):
    ks=LM.repeat_kv(key,module.num_key_value_groups);vs=LM.repeat_kv(value,module.num_key_value_groups)
    aw=torch.matmul(query,ks.transpose(2,3))*scaling
    if attention_mask is not None: aw=aw+attention_mask[:,:,:,:ks.shape[-2]]
    if S["on"] and getattr(module,"layer_idx",0)>=S["lmin"] and S["hi"]>S["lo"]:
        seg=aw[...,-1:,S["lo"]:S["hi"]]; aw[...,-1:,S["lo"]:S["hi"]]=seg+S["alpha"]*seg.abs()
    aw=F.softmax(aw,dim=-1,dtype=torch.float32).to(query.dtype);aw=F.dropout(aw,p=dropout,training=module.training)
    return torch.matmul(aw,vs).transpose(1,2).contiguous(),aw
LM.eager_attention_forward=pai_eager

@torch.inference_mode()
def answer(image,question):
    prompt=f"{SYS} USER: <image>\n{question}\nAnswer the question using a single word or phrase. ASSISTANT:"
    inp=proc(images=image,text=prompt,return_tensors="pt").to(DEV,torch.float16)
    vis=(inp.input_ids[0]==img_id).nonzero(as_tuple=True)[0];S["lo"]=int(vis.min());S["hi"]=int(vis.max())+1
    out=model.generate(**inp,max_new_tokens=5,do_sample=False,num_beams=1)
    return tok.decode(out[0,inp.input_ids.shape[1]:],skip_special_tokens=True).strip().lower()

def yn(t):
    t=t.lower()
    if t.startswith("yes") or re.search(r"\byes\b",t): return "yes"
    if t.startswith("no") or re.search(r"\bno\b",t): return "no"
    return "no"
print(f"[pope] alpha={a.alpha} n/split={a.n}",flush=True)
ALL=load_dataset("lmms-lab/POPE",split="test",token=False)
for split in ["adversarial","popular","random"]:
    ds=ALL.filter(lambda x:x["category"]==split)
    n=min(a.n,len(ds));tp=fp=tn=fn=0;correct=0
    for i in range(n):
        ex=ds[i];gt=ex["answer"].strip().lower();pred=yn(answer(ex["image"].convert("RGB"),ex["question"]))
        correct+=(pred==gt)
        if gt=="yes" and pred=="yes": tp+=1
        elif gt=="yes" and pred=="no": fn+=1
        elif gt=="no" and pred=="yes": fp+=1
        else: tn+=1
    acc=correct/n; prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); f1=2*prec*rec/max(prec+rec,1e-9)
    print(f"[pope] {split:12} acc={acc:.4f} f1={f1:.4f} prec={prec:.4f} rec={rec:.4f} (n={n})",flush=True)
