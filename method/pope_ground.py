#!/usr/bin/env python3
"""我们的 logit-lens grounding 检测器直接用于 POPE 判别(POPE = object 存在性判别 = 检测器强项)。
一次 forward(POPE prompt)同时取: (i) visual token hidden@L31 -> grounding gc(obj); (ii) 首 token 的 yes/no logit。
评估三种用法(阈值/λ 在 cal split 定, test split 评, 不偷看):
  A 纯 grounding 判别:  pred = yes if gc>t
  B 模型 baseline:       pred = argmax(yes,no) logit
  C grounding-gated:     在模型 yes/no logit 上加 λ*gc (gc 高->偏 yes)
输出 AUROC + 各法 test-acc, 对比 baseline。"""
import argparse,json,os,sys,re
if "--gpu" in sys.argv: os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[sys.argv.index("--gpu")+1]
import torch,torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoProcessor,LlavaForConditionalGeneration
SYS=("A chat between a curious user and an artificial intelligence assistant. "
     "The assistant gives helpful, detailed, and polite answers to the user's questions.")
ap=argparse.ArgumentParser()
ap.add_argument("--split",default="adversarial"); ap.add_argument("--n",type=int,default=500)
ap.add_argument("--gpu",default="1"); ap.add_argument("--topk",type=int,default=10)
ap.add_argument("--out",default="method/pope_ground_adv.jsonl")
a=ap.parse_args()
proc=AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf");tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",torch_dtype=torch.float16,device_map="cuda:0",attn_implementation="eager").eval()
img_id=model.config.image_token_index;DEV="cuda:0"
W=model.lm_head.weight;norm=model.model.language_model.norm
cal=json.load(open("detect/calibration_tkb.json")); calmean=float(np.mean(list(cal.values())))
# yes/no token ids (首 token)
YES=tok.encode("Yes",add_special_tokens=False)[0]; NO=tok.encode("No",add_special_tokens=False)[0]
yes_lc=tok.encode("yes",add_special_tokens=False)[0]; no_lc=tok.encode("no",add_special_tokens=False)[0]

def obj_embed(word):
    ids=tok.encode(word,add_special_tokens=False)
    if not ids: return None
    return F.normalize(W[ids].float().mean(0),dim=-1).to(DEV)
def extract_obj(q):
    m=re.search(r"[Ii]s there (?:a |an |the )?(.+?) in the image",q)
    return m.group(1).strip().lower() if m else None

@torch.inference_mode()
def probe(image,question,obj):
    prompt=f"{SYS} USER: <image>\n{question}\nAnswer the question using a single word or phrase. ASSISTANT:"
    inp=proc(images=image,text=prompt,return_tensors="pt").to(DEV,torch.float16)
    ids=inp.input_ids[0];vis=(ids==img_id).nonzero(as_tuple=True)[0]
    o=model(**inp,output_hidden_states=True,use_cache=False)
    hn=F.normalize(norm(o.hidden_states[31][0,vis,:]).float(),dim=-1)          # (576,d)
    lg=o.logits[0,-1,:].float()
    e=obj_embed(obj)
    g=(hn@e).topk(min(a.topk,hn.shape[0])).values.mean().item() if e is not None else None
    # 模型 yes/no logit (合并大小写)
    yl=float(torch.logsumexp(lg[[YES,yes_lc]],0)); nl=float(torch.logsumexp(lg[[NO,no_lc]],0))
    return g, yl, nl

ds=load_dataset("lmms-lab/POPE",split="test",token=False).filter(lambda x:x["category"]==a.split)
n=min(a.n,len(ds));rows=[];fout=open(a.out,"w")
print(f"[pope-ground] {a.split} n={n} 探测中...",flush=True)
for i in range(n):
    ex=ds[i];q=ex["question"];gt=1 if ex["answer"].strip().lower()=="yes" else 0;obj=extract_obj(q)
    if obj is None: continue
    g,yl,nl=probe(ex["image"].convert("RGB"),q,obj)
    if g is None: continue
    gc=g-cal.get(obj,calmean)
    rows.append((gc,yl,nl,gt)); fout.write(json.dumps({"obj":obj,"gc":gc,"yl":yl,"nl":nl,"gt":gt})+"\n")
    if (i+1)%100==0: print(f"  {i+1}/{n}",flush=True)
fout.close()
gc=np.array([r[0] for r in rows]);yl=np.array([r[1] for r in rows]);nl=np.array([r[2] for r in rows]);y=np.array([r[3] for r in rows])
def auroc(s,y):
    o=np.argsort(s);r=np.empty(len(s));r[o]=np.arange(1,len(s)+1)
    npos=y.sum();nneg=len(y)-npos
    return (r[y==1].sum()-npos*(npos+1)/2)/(npos*nneg+1e-9)
def acc_at(pred,y,m): return (pred[m].astype(int)==y[m]).mean()
ci=np.arange(len(gc));calm=ci%2==0;testm=ci%2==1
# A 纯 grounding: cal 上扫阈值
best_t,best=None,-1
for t in np.unique(gc[calm]):
    ac=acc_at((gc>t),y,calm)
    if ac>best: best,best_t=ac,t
A_test=acc_at((gc>best_t),y,testm)
# B baseline: 模型 yes/no argmax
mb=(yl>nl); B_test=acc_at(mb,y,testm); B_all=acc_at(mb,y,ci>=0)
# C gated: (yl-nl)+λ*gc >0 ; cal 上扫 λ
d=yl-nl; bestl,bestc=0,-1
for lam in np.linspace(0,20,201):
    ac=acc_at((d+lam*gc>0),y,calm)
    if ac>bestc: bestc,bestl=ac,lam
C_test=acc_at((d+bestl*gc>0),y,testm)
print(f"\n[pope-ground] {a.split} n={len(gc)}")
print(f"  grounding AUROC (gc->present) = {auroc(gc,y):.4f}")
print(f"  A 纯grounding判别   test-acc={A_test:.4f}  (cal阈值={best_t:.4f}, cal-acc={best:.4f})")
print(f"  B 模型baseline      test-acc={B_test:.4f}  全量acc={B_all:.4f}")
print(f"  C grounding-gated   test-acc={C_test:.4f}  (最优λ={bestl:.2f}, cal-acc={bestc:.4f})")
print(f"  => C比B {'提升' if C_test>B_test else '未提升'} {C_test-B_test:+.4f}")
