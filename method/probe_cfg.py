#!/usr/bin/env python3
"""正交检测信号: CFG 图像因果效应。teacher-force baseline caption, 对每个 object 词的 token 位置计算
Δ = logprob_full(token) - logprob_textonly(token) —— 图像把这个词推高了多少。
真实object: 图像证据推高 (Δ>0); 幻觉: 语言先验独自吐 (Δ≈0或<0)。与 grounding(cosine)正交。
join detect/probe_full.jsonl 的 hallucinated 标签 + tkB_L31(grounding), 测:
  gc 单独 / Δ 单独 / gc+Δ 组合(z-sum 训练无关 & logreg-CV 上界) 的 AUROC。
若组合 > 0.82 => 检测器可提升 => 整条消除frontier下移(见 method/excise.py)。"""
import argparse, importlib.util, json, os, re, sys
if "--gpu" in sys.argv: os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[sys.argv.index("--gpu")+1]
import torch, torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
SYS=("A chat between a curious user and an artificial intelligence assistant. "
     "The assistant gives helpful, detailed, and polite answers to the user's questions.")
P_IMG=f"{SYS} USER: <image>\nPlease describe this image in detail. ASSISTANT:"
P_TXT=f"{SYS} USER: \nPlease describe this image in detail. ASSISTANT:"
def CU():
    p="/volume/exploration/EvolvingLMMs/lmms-eval/lmms_eval/tasks/coco_cap_chair/utils.py"
    s=importlib.util.spec_from_file_location("cu",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ap=argparse.ArgumentParser()
ap.add_argument("--gpu",default="1"); ap.add_argument("--base",default="method/F_base.jsonl")
ap.add_argument("--probe",default="detect/probe_full.jsonl"); ap.add_argument("--out",default="detect/cfg_signal.json")
ap.add_argument("--recompute",action="store_true"); ap.add_argument("--n",type=int,default=100000)
a=ap.parse_args(); cu=CU()
ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
base=[json.loads(l) for l in open(a.base)][:a.n]

if a.recompute or not os.path.exists(a.out):
    torch.set_grad_enabled(False)
    proc=AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf");tok=proc.tokenizer
    model=LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",torch_dtype=torch.float16,device_map="cuda:0",attn_implementation="eager").eval()
    img_id=model.config.image_token_index;DEV="cuda:0"
    @torch.no_grad()
    def delta_per_caption(image,caption):
        """返回 caption 各 token 的 Δ=logprob_full-logprob_txt, 以及 caption token ids + char offsets。"""
        # full (with image)
        pf=proc(images=image,text=P_IMG,return_tensors="pt").to(DEV,torch.float16); Lp=pf.input_ids.shape[1]
        ff=proc(images=image,text=P_IMG+" "+caption,return_tensors="pt").to(DEV,torch.float16)
        of=model(**ff); logf=torch.log_softmax(of.logits[0].float(),dim=-1)     # (T,V)
        cap_ids=ff.input_ids[0,Lp:]                                              # caption tokens
        # text-only
        pt=tok(P_TXT,return_tensors="pt").to(DEV); Lpt=pt.input_ids.shape[1]
        ft=tok(P_TXT+" "+caption,return_tensors="pt").to(DEV)
        ot=model(input_ids=ft.input_ids,attention_mask=ft.attention_mask); logt=torch.log_softmax(ot.logits[0].float(),dim=-1)
        cap_ids_t=ft.input_ids[0,Lpt:]
        m=min(cap_ids.shape[0],cap_ids_t.shape[0])
        deltas=[]
        for j in range(m):
            tk=int(cap_ids[j])
            lf=float(logf[Lp+j-1,tk]); lt=float(logt[Lpt+j-1,tk])
            deltas.append(lf-lt)
        # char offsets for caption tokens (for object->token mapping)
        offs=[len(tok.decode(cap_ids[:j+1],skip_special_tokens=True)) for j in range(m)]
        return deltas, offs
    out={}
    for c,r in enumerate(base):
        i=r["image_id"]; cap=r["caption"]
        words,node,idxs,_=cu.caption_to_words(cap)
        if not node: out[str(i)]={}; continue
        deltas,offs=delta_per_caption(ds[i]["image"].convert("RGB"),cap)
        low=cap.lower(); od={}
        # 每个 canonical object: 找它的surface出现的char位置->token->聚合Δ(取min=最像幻觉的那次)
        surf_by_canon={}
        for k in range(len(node)):
            surf_by_canon.setdefault(node[k],set()).add(words[k])
        for o,surfs in surf_by_canon.items():
            dd=[]
            for sf in surfs:
                for mt in re.finditer(r'\b'+re.escape(sf)+r's?\b',low):
                    cp=mt.start()
                    # token index containing char cp
                    ti=next((t for t,ov in enumerate(offs) if ov>cp),None)
                    if ti is not None and ti<len(deltas): dd.append(deltas[ti])
            if dd: od[o]={"mean":float(np.mean(dd)),"min":float(np.min(dd))}
        out[str(i)]=od
        if (c+1)%50==0: print(f"cfg {c+1}/{len(base)}",flush=True)
    json.dump(out,open(a.out,"w")); print("saved",a.out,flush=True)
else:
    out=json.load(open(a.out)); print("loaded",a.out,flush=True)

# ==== join with labels + AUROC ====
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
probe=[json.loads(l) for l in open(a.probe)]
rows=[]
for p in probe:
    i=p["image_id"]; o=p["object"]
    d=out.get(str(i),{}).get(o)
    if d is None: continue
    rows.append({"y":int(p["hallucinated"]),"gc":float(p["tkB_L31"]),"loose":float(p["looseB_L31"]),
                 "dmean":d["mean"],"dmin":d["min"]})
y=np.array([r["y"] for r in rows])
def au(s):
    s=np.array(s); m=np.isfinite(s); a_=roc_auc_score(y[m],s[m]); return max(a_,1-a_)
gc=np.array([r["gc"] for r in rows]); dmean=np.array([r["dmean"] for r in rows]); dmin=np.array([r["dmin"] for r in rows])
print(f"\n[probe-cfg] n={len(rows)} hallucinated={int(y.sum())} base_rate={y.mean():.3f}")
print(f"  grounding gc(tkB_L31)      AUROC={au(-gc):.4f}   (低gc=幻觉)")
print(f"  CFG image-effect Δmean      AUROC={au(-dmean):.4f}  (低Δ=幻觉)")
print(f"  CFG image-effect Δmin       AUROC={au(-dmin):.4f}")
def zs(v):
    v=np.array(v,float); return (v-np.nanmean(v))/(np.nanstd(v)+1e-9)
combo_z=-(zs(gc))-(zs(dmean))   # 都是低=幻觉,取负z相加
print(f"  z-sum(gc+Δmean) [训练无关]  AUROC={au(combo_z):.4f}")
# logreg CV 上界
for feats,name in [(["gc"],"gc"),(["gc","dmean"],"gc+Δmean"),(["gc","dmean","dmin","loose"],"gc+Δmean+Δmin+loose")]:
    X=np.array([[r[f] for f in feats] for r in rows])
    X=(X-X.mean(0))/(X.std(0)+1e-9)
    pr=cross_val_predict(LogisticRegression(max_iter=1000),X,y,cv=5,method="predict_proba")[:,1]
    print(f"  logreg-CV [{name:22}] AUROC={roc_auc_score(y,pr):.4f}")
