#!/usr/bin/env python3
"""方法C(survey#1): 句级 entity-locked 约束重生成 (training-free, post-hoc)。
Pass1 baseline caption -> 按句切分 -> 每句用模型自我验证(selfverify.json, A的0.89检测)找幻觉object ->
只对含幻觉的句子: prefix-force前文, ban幻觉object的token, 重生成这一句(到句号), 干净句逐字保留 ->
gate: 若新句仍含幻觉object 或 丢了原句的真实object => 回退原句(保证不变差)。拼回。
穿墙点: post-hoc(避开轨迹耦合) + 整句重生(通顺,不破碎) + 干净句verbatim+gate(recall结构性保护)。
novelty: 训练无关 entity-locked 句级重生成无人发表(vs Woodpecker外部LLM/LURE训练/CGD decode-time)。
flagger默认self-check(0.89); grounding仅可选。"""
import argparse, importlib.util, json, os, re, sys
from collections import defaultdict
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
ap.add_argument("--flagger",default="selfcheck",choices=["selfcheck","grounding"])
ap.add_argument("--svthr",type=float,default=0.0); ap.add_argument("--tau",type=float,default=-0.06)
ap.add_argument("--protect",default="person,tennis racket,surfboard,sports ball,elephant,umbrella")
ap.add_argument("--split",default="test",choices=["all","test","cal"])
ap.add_argument("--n",type=int,default=250); ap.add_argument("--max_new_tokens",type=int,default=512)
ap.add_argument("--gpu",default="1"); ap.add_argument("--out",default="method/sr_out.jsonl"); ap.add_argument("--show",type=int,default=4)
a=ap.parse_args(); cu=CU()
PROTECT=set(x.strip() for x in a.protect.split(",") if x.strip())
torch.set_grad_enabled(False)
proc=AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf");tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",torch_dtype=torch.float16,device_map="cuda:0",attn_implementation="eager").eval()
img_id=model.config.image_token_index;eos=tok.eos_token_id;DEV="cuda:0"
sv=json.load(open("detect/selfverify.json")); gc=json.load(open("method/excise_gc.json"))
canon2surf=defaultdict(set); IRREG={"man":"men","woman":"women","person":"people","child":"children","foot":"feet","tooth":"teeth"}
canon2ftoks=defaultdict(set)
for w,canon in cu.INVERSE_SYNONYM_DICT.items():
    canon2surf[canon].add(w)
    for s in [w,w+"s",w.rstrip("s"),w.capitalize()]+([IRREG[w],IRREG[w].capitalize()] if w in IRREG else []):
        ids=tok.encode(s,add_special_tokens=False)
        if ids: canon2ftoks[canon].add(ids[0])
def is_hall(i,o):
    if o in PROTECT: return False
    if a.flagger=="selfcheck":
        d=sv.get(str(i),{}).get(o); return (d is not None) and (d["nl"]-d["yl"])>a.svthr
    return gc.get(str(i),{}).get(o,1.0)<a.tau

@torch.no_grad()
def prefill(image):
    vl=proc(images=image,text=P_IMG,return_tensors="pt").to(DEV,torch.float16)
    return vl
@torch.no_grad()
def regen_sentence(vl, prefix_text, banned):
    """从 prompt+prefix_text 续写, ban banned首token, 生成到句末(第一个'. '或换行或eos)。返回新句text。"""
    pre_ids=tok(prefix_text,add_special_tokens=False).input_ids if prefix_text else []
    ids=torch.cat([vl.input_ids,torch.tensor([pre_ids],device=DEV,dtype=vl.input_ids.dtype)],dim=1) if pre_ids else vl.input_ids
    am=torch.ones_like(ids)
    o=model(input_ids=ids,pixel_values=vl.pixel_values,attention_mask=am,use_cache=True)
    pk=o.past_key_values; lg=o.logits[0,-1,:].float(); gen=[]
    bl=list(banned)
    for _ in range(80):
        if bl: lg[bl]=-float("inf")
        nxt=int(lg.argmax())
        if nxt==eos: break
        gen.append(nxt)
        txt=tok.decode(gen)
        if len(gen)>3 and (txt.rstrip().endswith(".") ): break   # 到句末停
        o=model(input_ids=torch.tensor([[nxt]],device=DEV),past_key_values=pk,use_cache=True)
        pk=o.past_key_values; lg=o.logits[0,-1,:].float()
    return tok.decode(gen,skip_special_tokens=True).strip()

def split_sents(cap):
    parts=re.split(r'(?<=[.!?])\s+',cap.strip())
    return [p for p in parts if p.strip()]
def sent_objs(sent): _,node,_,_=cu.caption_to_words(sent); return node
def real_surfaces(i,sent,objs):
    # 原句里真实(非幻觉)object的surface, 用于gate检查是否被丢
    rs=set()
    for o in set(objs):
        if not is_hall(i,o):
            low=sent.lower()
            for sf in canon2surf.get(o,{o}):
                if re.search(r'\b'+re.escape(sf)+r's?\b',low): rs.add(sf)
    return rs

@torch.no_grad()
def run(image,i,cap):
    sents=split_sents(cap); vl=prefill(image); nrw=0
    out_sents=[]
    for sent in sents:
        objs=sent_objs(sent); bad=[o for o in set(objs) if is_hall(i,o)]
        if not bad:
            out_sents.append(sent); continue
        prefix=" ".join(out_sents)+(" " if out_sents else "")
        banned={t for o in bad for t in canon2ftoks.get(o,set())}
        news=regen_sentence(vl, prefix, banned); nrw+=1
        # gate: 新句不能含幻觉object, 且不能丢原句真实object
        n_objs=set(sent_objs(news)); still_bad=[o for o in n_objs if is_hall(i,o)]
        kept_real=real_surfaces(i,sent,objs); low=news.lower()
        lost=[sf for sf in kept_real if not re.search(r'\b'+re.escape(sf)+r's?\b',low)]
        if news and not still_bad and not lost:
            out_sents.append(news)
        else:
            out_sents.append(sent)   # 回退: 保证不变差(此句维持原样, 仍含幻觉但recall不丢)
            nrw-=1
    return " ".join(out_sents), nrw

ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
base={json.loads(l)["image_id"]:json.loads(l) for l in open("method/F_base.jsonl")}
idxs=[i for i in range(len(ds)) if a.split=="all" or (a.split=="test" and i%2==1) or (a.split=="cal" and i%2==0)]
idxs=[i for i in idxs if i in base][:a.n]
fout=open(a.out,"w");res_b=[];res_r=[];tot=0;samples=[]
print(f"[sr] flagger={a.flagger} svthr={a.svthr} split={a.split} n={len(idxs)}",flush=True)
for c,i in enumerate(idxs):
    cap=base[i]["caption"]; new,nrw=run(ds[i]["image"].convert("RGB"),i,cap); tot+=nrw
    _,nb,_,_=cu.caption_to_words(cap); _,nr,_,_=cu.caption_to_words(new)
    res_b.append({"answer":list(ds[i]["gt_object"]),"pred":nb}); res_r.append({"answer":list(ds[i]["gt_object"]),"pred":nr})
    fout.write(json.dumps({"image_id":i,"base":cap,"rewrite":new,"nrw":nrw})+"\n")
    if nrw>0 and len(samples)<a.show: samples.append((i,cap,new))
    if (c+1)%40==0: print(f"[sr] {c+1}/{len(idxs)} (accepted rewrites={tot})",flush=True)
fout.close()
def agg(res): return (cu.coco_cap_chair_aggregate_results_chair_s(res),cu.coco_cap_chair_aggregate_results_chair_i(res),
                      cu.coco_cap_chair_aggregate_results_recall(res),sum(len(x['pred']) for x in res)/len(res))
b=agg(res_b); r=agg(res_r)
print(f"\n[sr] baseline    CHAIR_s={b[0]:.2f} CHAIR_i={b[1]:.2f} recall={b[2]:.2f} objs={b[3]:.2f}")
print(f"[sr] 句级重生成  CHAIR_s={r[0]:.2f} CHAIR_i={r[1]:.2f} recall={r[2]:.2f} objs={r[3]:.2f} (accepted={tot})")
gate="✓" if (r[3]>=6.6 and r[2]>=74 and r[0]<=38) else ("recall/objs ok" if (r[3]>=6.6 and r[2]>=74) else "✗")
print(f"[sr] 判据(CHAIR_s≤38 & recall≥74 & objs≥6.6)={gate}")
for iid,old,new in samples:
    print(f"\n--- img{iid}\n  原: {old[:210]}\n  新: {new[:210]}")
