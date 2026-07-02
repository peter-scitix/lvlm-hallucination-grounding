#!/usr/bin/env python3
"""Idea A':外科式切除 (recall-safe)。不重生成、不改真实object;只把高置信幻觉object的提及删掉。
先对 F_base.jsonl 的 baseline caption,逐(image,object)算 grounding gc,缓存;然后扫阈值 tau:
从每张图的 pred 里移除 gc<tau 的 object(其余原样保留),重算 CHAIR_s/i/recall。
这测的是「检测器当过滤器」的 CHAIR frontier —— recall-safe by construction(只删被flag的)。
对比 soft(43.4/13.2/74.5)与 baseline(50.4/15.4/76.9),看能否在匹配recall下把CHAIR_s压得更低。
"""
import argparse, importlib.util, json, os, sys
if "--gpu" in sys.argv: os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[sys.argv.index("--gpu")+1]
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
ap.add_argument("--topk",type=int,default=10); ap.add_argument("--gpu",default="1")
ap.add_argument("--base",default="method/F_base.jsonl")
ap.add_argument("--gccache",default="method/excise_gc.json")
ap.add_argument("--recompute",action="store_true")
ap.add_argument("--protect",default="",help="逗号分隔的canonical object, 永不移除(base-rate-aware: 低幻觉率类)")
a=ap.parse_args(); cu=CU()
PROTECT=set(x.strip() for x in a.protect.split(",") if x.strip())
ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
gt={i:list(ds[i]["gt_object"]) for i in range(len(ds))}
base=[json.loads(l) for l in open(a.base)]

if a.recompute or not os.path.exists(a.gccache):
    proc=AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf");tok=proc.tokenizer
    model=LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",torch_dtype=torch.float16,device_map="cuda:0",attn_implementation="eager").eval()
    torch.set_grad_enabled(False)
    img_id=model.config.image_token_index;DEV="cuda:0"
    W=model.lm_head.weight;norm=model.model.language_model.norm
    cal=json.load(open("/volume/exploration/EvolvingLMMs/detect/calibration_tkb.json"));calmean=sum(cal.values())/len(cal)
    _emb={}
    def emb(o):
        if o not in _emb:
            ids=tok.encode(o,add_special_tokens=False);_emb[o]=F.normalize(W[ids].float().mean(0),dim=-1).to(DEV) if ids else None
        return _emb[o]
    @torch.no_grad()
    def hn_of(image):
        vl=proc(images=image,text=P_IMG,return_tensors="pt").to(DEV,torch.float16)
        vis=(vl.input_ids[0]==img_id).nonzero(as_tuple=True)[0]
        o=model(**vl,output_hidden_states=True,use_cache=False)
        return F.normalize(norm(o.hidden_states[31][0,vis,:]).float(),dim=-1)
    cache={}
    for c,r in enumerate(base):
        i=r["image_id"]; uniq=set(r["pred"])
        if not uniq: cache[str(i)]={}; continue
        hn=hn_of(ds[i]["image"].convert("RGB"))
        d={}
        for o in uniq:
            e=emb(o)
            d[o]=((hn@e).topk(min(a.topk,hn.shape[0])).values.mean().item()-cal.get(o,calmean)) if e is not None else 0.0
        cache[str(i)]=d
        if (c+1)%50==0: print(f"gc {c+1}/{len(base)}",flush=True)
    json.dump(cache,open(a.gccache,"w")); print("saved",a.gccache,flush=True)
else:
    cache=json.load(open(a.gccache)); print("loaded cached gc",flush=True)

def agg(preds):
    R=[{"answer":gt[r["image_id"]],"pred":p} for r,p in zip(base,preds)]
    return (cu.coco_cap_chair_aggregate_results_chair_s(R),cu.coco_cap_chair_aggregate_results_chair_i(R),
            cu.coco_cap_chair_aggregate_results_recall(R),sum(len(x) for x in preds)/len(preds))
# baseline
b=agg([r["pred"] for r in base]); print(f"\nbaseline  CHAIR_s={b[0]:.2f} CHAIR_i={b[1]:.2f} recall={b[2]:.2f} objs={b[3]:.2f}  (n={len(base)})")
print("\nexcise frontier (删 gc<tau 的 object 提及):")
print(f"{'tau':>7} {'CHAIR_s':>8} {'CHAIR_i':>8} {'recall':>7} {'objs':>6} {'removed/img':>11}")
for tau in [-0.10,-0.08,-0.06,-0.05,-0.04,-0.03,-0.025,-0.02,-0.015,-0.01,0.0,0.01]:
    preds=[]; nrem=0
    for r in base:
        d=cache.get(str(r["image_id"]),{})
        kept=[o for o in r["pred"] if (o in PROTECT or d.get(o,1.0)>=tau)]
        nrem+=len(r["pred"])-len(kept); preds.append(kept)
    m=agg(preds)
    print(f"{tau:>7.3f} {m[0]:>8.2f} {m[1]:>8.2f} {m[2]:>7.2f} {m[3]:>6.2f} {nrem/len(base):>11.2f}")
print("\n对比 soft(β12): 43.4/13.2/74.5")
