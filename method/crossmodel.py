#!/usr/bin/env python3
"""跨模型验证 THEORY: 在任意 llava-hf LVLM 上, 检验 self-verify(gen-verification gap) 是否在 CHAIR_i-recall
frontier 上 dominate grounding(logit-lens) —— 即 T1 推论(高ROC检测器→更低frontier)是否跨模型成立。
流程: 生成baseline caption -> 每个mentioned object 算 grounding(topk cos, raw+per-obj-demean) + self-verify(is-there-X)
-> 幻觉标签(object∉GT) -> 报 ROC(grounding/self-verify) + CHAIR_i-recall frontier对比。"""
import argparse, importlib.util, json, os, sys
from collections import defaultdict
if "--gpu" in sys.argv: os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[sys.argv.index("--gpu")+1]
import torch, torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
SYS=("A chat between a curious user and an artificial intelligence assistant. "
     "The assistant gives helpful, detailed, and polite answers to the user's questions.")
P_IMG=f"{SYS} USER: <image>\nPlease describe this image in detail. ASSISTANT:"
def CU():
    p="/volume/exploration/EvolvingLMMs/lmms-eval/lmms_eval/tasks/coco_cap_chair/utils.py"
    s=importlib.util.spec_from_file_location("cu",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ap=argparse.ArgumentParser()
ap.add_argument("--model",default="llava-hf/llava-1.5-13b-hf")
ap.add_argument("--n",type=int,default=500); ap.add_argument("--max_new_tokens",type=int,default=512)
ap.add_argument("--gpu",default="1"); ap.add_argument("--out",default="method/crossmodel_out.json"); ap.add_argument("--recompute",action="store_true")
a=ap.parse_args(); cu=CU()
ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
gt={i:set(ds[i]["gt_object"]) for i in range(len(ds))}

if a.recompute or not os.path.exists(a.out):
    torch.set_grad_enabled(False)
    proc=AutoProcessor.from_pretrained(a.model);tok=proc.tokenizer
    model=LlavaForConditionalGeneration.from_pretrained(a.model,torch_dtype=torch.float16,device_map="cuda:0",attn_implementation="eager").eval()
    img_id=model.config.image_token_index;eos=tok.eos_token_id;DEV="cuda:0"
    W=model.lm_head.weight; norm=model.model.language_model.norm
    Llast=model.config.text_config.num_hidden_layers  # hidden_states index of last layer
    YES=tok.encode("Yes",add_special_tokens=False)[0]; NO=tok.encode("No",add_special_tokens=False)[0]
    yl_=tok.encode("yes",add_special_tokens=False)[0]; nl_=tok.encode("no",add_special_tokens=False)[0]
    _emb={}
    def emb(o):
        if o not in _emb:
            ids=tok.encode(o,add_special_tokens=False); _emb[o]=F.normalize(W[ids].float().mean(0),dim=-1).to(DEV) if ids else None
        return _emb[o]
    @torch.no_grad()
    def gen_and_hidden(image):
        vl=proc(images=image,text=P_IMG,return_tensors="pt").to(DEV,torch.float16)
        out=model.generate(**vl,max_new_tokens=a.max_new_tokens,do_sample=False,num_beams=1)
        cap=tok.decode(out[0,vl.input_ids.shape[1]:],skip_special_tokens=True).strip()
        vis=(vl.input_ids[0]==img_id).nonzero(as_tuple=True)[0]
        o=model(**vl,output_hidden_states=True,use_cache=False)
        hn=F.normalize(norm(o.hidden_states[Llast][0,vis,:]).float(),dim=-1)
        return cap,hn
    @torch.no_grad()
    def selfcheck(image,obj):
        pr=f"{SYS} USER: <image>\nIs there a {obj} in the image?\nAnswer the question using a single word or phrase. ASSISTANT:"
        inp=proc(images=image,text=pr,return_tensors="pt").to(DEV,torch.float16)
        lg=model(**inp).logits[0,-1,:].float()
        return float(torch.logsumexp(lg[[NO,nl_]],0)-torch.logsumexp(lg[[YES,yl_]],0))  # 高=model说no=幻觉
    recs=[]
    n=min(a.n,len(ds))
    print(f"[xm] {a.model} n={n} Llast={Llast}",flush=True)
    for i in range(n):
        img=ds[i]["image"].convert("RGB")
        cap,hn=gen_and_hidden(img)
        _,node,_,_=cu.caption_to_words(cap)
        for o in set(node):
            e=emb(o)
            g=(hn@e).topk(min(10,hn.shape[0])).values.mean().item() if e is not None else None
            sv=selfcheck(img,o)
            recs.append({"i":i,"o":o,"hall":0 if o in gt[i] else 1,"g":g,"sv":sv,"pred":node.count(o)})
        if (i+1)%50==0: print(f"[xm] {i+1}/{n} ({len(recs)} objs)",flush=True)
    json.dump({"model":a.model,"recs":recs},open(a.out,"w")); print("saved",a.out,flush=True)
else:
    recs=json.load(open(a.out))["recs"]; print("loaded",flush=True)

# ==== analyze ====
from sklearn.metrics import roc_auc_score
recs=[r for r in recs if r["g"] is not None]
y=np.array([r["hall"] for r in recs])
# per-object demean grounding
gmean=defaultdict(list)
for r in recs: gmean[r["o"]].append(r["g"])
gm={o:np.mean(v) for o,v in gmean.items()}
sg_raw=np.array([-r["g"] for r in recs])                      # 高=幻觉
sg_dem=np.array([-(r["g"]-gm[r["o"]]) for r in recs])
ssv=np.array([r["sv"] for r in recs])
def au(s): a_=roc_auc_score(y,s); return max(a_,1-a_)
print(f"\n[xm] n={len(recs)} base_rate={y.mean():.3f}")
print(f"  ROC grounding(raw)={au(sg_raw):.4f} grounding(demean)={au(sg_dem):.4f} self-verify={au(ssv):.4f}")
# CHAIR_i-recall frontier对比 (per-image, 移除score>=t的object所有mention)
base=defaultdict(list)
for r in recs: base[r["i"]].append(r)
def frontier(scoref):
    sc={(r["i"],r["o"]):scoref(r) for r in recs}
    allv=sorted(sc.values())
    out=[]
    for q in np.linspace(0.55,1.0,19):
        t=np.quantile(allv,q); preds=[]
        for i,rs in base.items():
            kept=[]
            for r in rs:
                if sc[(i,r["o"])]<t: kept+= [r["o"]]*r["pred"]
            preds.append((i,kept))
        R=[{"answer":list(gt[i]),"pred":pp} for i,pp in preds]
        out.append((cu.coco_cap_chair_aggregate_results_recall(R),cu.coco_cap_chair_aggregate_results_chair_i(R)))
    return sorted(out)
fg=frontier(lambda r:-(r["g"]-gm[r["o"]])); fs=frontier(lambda r:r["sv"])
def ci_at(fr,rt):
    rs=[x[0] for x in fr]; cs=[x[1] for x in fr]
    return float(np.interp(rt,rs,cs)) if min(rs)<=rt<=max(rs) else None
print(f"\n  CHAIR_i @ recall (grounding-demean vs self-verify):")
for rt in [72,70,68,66,64]:
    x=ci_at(fg,rt); z=ci_at(fs,rt)
    if x and z: print(f"    recall={rt}: grounding {x:.2f} | self-verify {z:.2f}  {'SV dominate✓' if z<x-0.1 else '≈/GR'}")
