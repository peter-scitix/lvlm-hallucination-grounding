#!/usr/bin/env python3
"""跨架构验证 (Qwen2.5-VL-7B): grounding(logit-lens) + self-verify(gen-verification gap) 检测是否迁移。
流程同 crossmodel: 生成caption -> 每个mentioned COCO object 算 grounding(视觉token->词表投影 topk cos, cal分片per-obj demean)
+ self-verify(Is there X) -> 幻觉标签(object∉GT) -> 报 AUROC。验证"检测跨架构成立"。"""
import argparse, importlib.util, json, os, sys
from collections import defaultdict
if "--gpu" in sys.argv: os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[sys.argv.index("--gpu")+1]
import torch, torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
def CU():
    p="/volume/exploration/EvolvingLMMs/lmms-eval/lmms_eval/tasks/coco_cap_chair/utils.py"
    s=importlib.util.spec_from_file_location("cu",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ap=argparse.ArgumentParser()
ap.add_argument("--model",default="Qwen/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--n",type=int,default=500); ap.add_argument("--max_new_tokens",type=int,default=512)
ap.add_argument("--gpu",default="1"); ap.add_argument("--out",default="method/qwen_detect.json")
a=ap.parse_args(); cu=CU()
ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
gt={i:set(ds[i]["gt_object"]) for i in range(len(ds))}
torch.set_grad_enabled(False)
proc=AutoProcessor.from_pretrained(a.model);tok=proc.tokenizer
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map="cuda:0",attn_implementation="eager").eval()  # Qwen2.5-VL 必须 bf16, fp16 生成乱码
DEV="cuda:0"; IMG_ID=model.config.image_token_id
Llast=model.config.num_hidden_layers if hasattr(model.config,"num_hidden_layers") else model.config.text_config.num_hidden_layers
W=model.lm_head.weight
# 防御性定位最终 RMSNorm
def find_norm(m):
    for path in ["model.language_model.norm","model.model.norm","model.norm","language_model.norm"]:
        o=m
        try:
            for p in path.split("."): o=getattr(o,p)
            if o is not None: print(f"[qwen] norm @ {path}",flush=True); return o
        except AttributeError: continue
    raise RuntimeError("norm not found")
norm=find_norm(model)
YES=tok.encode("Yes",add_special_tokens=False)[0]; NO=tok.encode("No",add_special_tokens=False)[0]
yl_=tok.encode("yes",add_special_tokens=False)[0]; nl_=tok.encode("no",add_special_tokens=False)[0]
_emb={}
def emb(o):
    if o not in _emb:
        ids=tok.encode(o,add_special_tokens=False); _emb[o]=F.normalize(W[ids].float().mean(0),dim=-1).to(DEV) if ids else None
    return _emb[o]
def build_inputs(image,text):
    msgs=[{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":text}]}]
    chat=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    imgs,vids=process_vision_info(msgs)
    return proc(text=[chat],images=imgs,videos=vids,return_tensors="pt",padding=True).to(DEV)
@torch.no_grad()
def gen_and_hidden(image):
    inp=build_inputs(image,"Describe this image.")
    out=model.generate(**inp,max_new_tokens=a.max_new_tokens,do_sample=False,num_beams=1)
    cap=tok.decode(out[0,inp.input_ids.shape[1]:],skip_special_tokens=True).strip()
    o=model(**inp,output_hidden_states=True,use_cache=False)
    vis=(inp.input_ids[0]==IMG_ID).nonzero(as_tuple=True)[0]
    hn=F.normalize(norm(o.hidden_states[Llast][0,vis,:]).float(),dim=-1)
    return cap,hn
@torch.no_grad()
def selfverify_no(image,o):
    inp=build_inputs(image,f"Is there a {o} in the image? Answer the question using a single word or phrase.")
    lg=model(**inp).logits[0,-1,:].float()
    return float(torch.logsumexp(lg[[NO,nl_]],0)-torch.logsumexp(lg[[YES,yl_]],0))
def gc_raw(hn,o):
    e=emb(o); return (hn@e).topk(min(10,hn.shape[0])).values.mean().item() if e is not None else None

recs=[]
print(f"[qwen] n={min(a.n,len(ds))} model={a.model} Llast={Llast} IMG_ID={IMG_ID}",flush=True)
for i in range(min(a.n,len(ds))):
    img=ds[i]["image"].convert("RGB")
    cap,hn=gen_and_hidden(img)
    _,node,_,_=cu.caption_to_words(cap)
    for o in set(node):
        g=gc_raw(hn,o)
        if g is None: continue
        sv=selfverify_no(img,o); hall=int(o not in gt[i])
        recs.append({"i":i,"o":o,"hall":hall,"g":g,"sv":sv})
    if (i+1)%50==0: print(f"[qwen] {i+1}/{min(a.n,len(ds))} recs={len(recs)}",flush=True)
json.dump({"model":a.model,"recs":recs},open(a.out,"w"))

# AUROC: grounding raw / per-obj demean(cal=偶i) / self-verify
def auroc(y,s):
    y=np.array(y); s=np.array(s); pos=s[y==1]; neg=s[y==0]
    if len(pos)==0 or len(neg)==0: return float("nan")
    return float((sum((pos[:,None]>neg[None,:]).sum(1))+0.5*sum((pos[:,None]==neg[None,:]).sum(1)))/(len(pos)*len(neg)))
cal=defaultdict(list)
for r in recs:
    if r["i"]%2==0: cal[r["o"]].append(r["g"])
calm={o:np.mean(v) for o,v in cal.items()}; gm=np.mean([r["g"] for r in recs])
te=[r for r in recs if r["i"]%2==1]
y=[r["hall"] for r in te]
# 幻觉分数: 越"是幻觉"越高 => 用 -g (grounding低=幻觉), sv (no-yes高=幻觉)
print(f"\n[qwen] test recs={len(te)} hall_rate={np.mean(y):.3f}")
print(f"[qwen] grounding raw AUROC   = {auroc(y,[-r['g'] for r in te]):.3f}")
print(f"[qwen] grounding demean AUROC= {auroc(y,[-(r['g']-calm.get(r['o'],gm)) for r in te]):.3f}")
print(f"[qwen] self-verify AUROC     = {auroc(y,[r['sv'] for r in te]):.3f}")
