#!/usr/bin/env python3
"""
Faithful reimplementation of LVLMs-Saliency LocoRE (Eq.8) on LLaVA-1.5 (llava_hf) — the module MISSING from
their released code. Eq.8: for the last (generating) query, multiply its POST-softmax attention weights to the
most-recent w_s OUTPUT tokens by gain (1+beta), then renormalize the row; applied at layers [lmin,lmax) (paper: 9-14).
Paper full-method params: beta=0.15. (SGRS is a proven no-op, so LocoRE must carry their 51->35.6 headline.)
Tests beta=0.15 (paper) + a sweep. Scores CHAIR on first-N CHAIR images; compare to our llava_hf baseline.
"""
import argparse, importlib.util, json, os, sys
if "--gpu" in sys.argv: os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[sys.argv.index("--gpu")+1]
import torch, torch.nn.functional as F
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
ap.add_argument("--beta",type=float,default=0.15); ap.add_argument("--ws",type=int,default=5)
ap.add_argument("--lmin",type=int,default=9); ap.add_argument("--lmax",type=int,default=15)  # layers 9-14
ap.add_argument("--n",type=int,default=100); ap.add_argument("--max_new_tokens",type=int,default=512)
ap.add_argument("--gpu",default="1"); ap.add_argument("--out",default="method/locore_out.jsonl"); ap.add_argument("--label",default="locore")
a=ap.parse_args(); cu=CU()
proc=AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf");tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",torch_dtype=torch.float16,device_map="cuda:0",attn_implementation="eager").eval()
img_id=model.config.image_token_index;eos=tok.eos_token_id;DEV="cuda:0"
S={"on":False,"imgend":0}
def locore_eager(module,query,key,value,attention_mask,scaling,dropout=0.0,**kw):
    ks=LM.repeat_kv(key,module.num_key_value_groups);vs=LM.repeat_kv(value,module.num_key_value_groups)
    aw=torch.matmul(query,ks.transpose(2,3))*scaling
    if attention_mask is not None: aw=aw+attention_mask[:,:,:,:ks.shape[-2]]
    aw=F.softmax(aw,dim=-1,dtype=torch.float32)
    li=getattr(module,"layer_idx",0)
    if S["on"] and a.lmin<=li<a.lmax:
        kv=aw.shape[-1]; P=kv-1                      # last key index = current pos during decode
        lo=max(S["imgend"], P-a.ws)                  # recent w_s OUTPUT tokens (after image block), exclude self
        if P>lo:
            row=aw[...,-1:, lo:P]                     # last query -> recent w_s keys
            aw[...,-1:, lo:P]=row*(1.0+a.beta)        # Eq.8 gain
            aw[...,-1,:]=aw[...,-1,:]/aw[...,-1,:].sum(-1,keepdim=True)   # renormalize the row
    aw=aw.to(query.dtype)
    out=torch.matmul(aw,vs).transpose(1,2).contiguous()
    return out, aw
LM.eager_attention_forward=locore_eager
@torch.inference_mode()
def gen(image):
    vl=proc(images=image,text=P_IMG,return_tensors="pt").to(DEV,torch.float16)
    vis=(vl.input_ids[0]==img_id).nonzero(as_tuple=True)[0]; S["imgend"]=int(vis.max())+1
    S["on"]=True
    out=model.generate(**vl,max_new_tokens=a.max_new_tokens,do_sample=False,num_beams=1)
    S["on"]=False
    return tok.decode(out[0,vl.input_ids.shape[1]:],skip_special_tokens=True).strip()
ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
fout=open(a.out,"w");res=[];n=min(a.n,len(ds))
print(f"[loco] {a.label} beta={a.beta} ws={a.ws} layers=[{a.lmin},{a.lmax}) n={n}",flush=True)
for i in range(n):
    cap=gen(ds[i]["image"].convert("RGB")); _,node,_,_=cu.caption_to_words(cap)
    res.append({"answer":list(ds[i]["gt_object"]),"pred":node}); fout.write(json.dumps({"image_id":i,"caption":cap,"pred":node})+"\n")
    if (i+1)%30==0: print(f"[loco] {i+1}/{n}",flush=True)
fout.close()
cs=cu.coco_cap_chair_aggregate_results_chair_s(res);ci=cu.coco_cap_chair_aggregate_results_chair_i(res);rc=cu.coco_cap_chair_aggregate_results_recall(res)
print(f"\n[loco] {a.label} CHAIR_s={cs:.2f} CHAIR_i={ci:.2f} recall={rc:.2f} objs={sum(len(x['pred']) for x in res)/len(res):.2f}",flush=True)
