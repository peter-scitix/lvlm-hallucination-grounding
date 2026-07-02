#!/usr/bin/env python3
"""Idea A: training-free post-hoc 自我修正 (grounding 引导的局部约束重生成)。
Pass1 正常 greedy 生成 (recall 拉满) -> 用 logit-lens grounding gc(o) 检测每个 mentioned object ->
只对「高置信幻觉」(gc < tau_strict, 检测器高 precision 端) 做局部修正: 把已生成序列截断到该 object 词之前,
把该 object 及其同义词的首 token 加入 ban 集, 从截断点续写到 EOS; 迭代至多 max_iters 次。
关键: 只动最确信的幻觉 (少误伤 -> 保 recall), 其余原样保留 Pass1 的完整 recall。
对比: Woodpecker(GPT+外部检测)/LURE/Volcano(需训练) —— 我们 training-free + 自己的 grounding + 局部续写。
"""
import argparse, importlib.util, json, os, re, sys
if "--gpu" in sys.argv: os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[sys.argv.index("--gpu")+1]
import torch, torch.nn.functional as F
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
SYS=("A chat between a curious user and an artificial intelligence assistant. "
     "The assistant gives helpful, detailed, and polite answers to the user's questions.")
P_IMG=f"{SYS} USER: <image>\nPlease describe this image in detail. ASSISTANT:"
def CU():
    p="/volume/exploration/EvolvingLMMs/lmms-eval/lmms_eval/tasks/coco_cap_chair/utils.py"
    s=importlib.util.spec_from_file_location("cu",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ap=argparse.ArgumentParser()
ap.add_argument("--tau_strict",type=float,default=-0.03,help="只修正 gc<tau_strict 的高置信幻觉")
ap.add_argument("--max_iters",type=int,default=3)
ap.add_argument("--topk",type=int,default=10)
ap.add_argument("--n",type=int,default=100); ap.add_argument("--max_new_tokens",type=int,default=512)
ap.add_argument("--split",default="all",choices=["all","test","cal"])
ap.add_argument("--gpu",default="1"); ap.add_argument("--out",default="method/sc_out.jsonl"); ap.add_argument("--label",default="selfcorrect")
a=ap.parse_args(); cu=CU()
proc=AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf");tok=proc.tokenizer
model=LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",torch_dtype=torch.float16,device_map="cuda:0",attn_implementation="eager").eval()
img_id=model.config.image_token_index;eos=tok.eos_token_id;DEV="cuda:0"
W=model.lm_head.weight;norm=model.model.language_model.norm
cal=json.load(open("/volume/exploration/EvolvingLMMs/detect/calibration_tkb.json"))
calmean=sum(cal.values())/len(cal)
# 同义词 -> 首token(ban用) + 规范词 -> 同义surface(定位用)
IRREG={"man":"men","woman":"women","person":"people","child":"children","foot":"feet","tooth":"teeth"}
canon2ftoks=defaultdict(set); canon2surf=defaultdict(set)
for w,canon in cu.INVERSE_SYNONYM_DICT.items():
    canon2surf[canon].add(w)
    for s in [w,w+"s",w.rstrip("s"),w.capitalize()]+([IRREG[w],IRREG[w].capitalize()] if w in IRREG else []):
        ids=tok.encode(s,add_special_tokens=False)
        if ids: canon2ftoks[canon].add(ids[0])
_embcache={}
def obj_embed(canon):
    if canon not in _embcache:
        ids=tok.encode(canon,add_special_tokens=False)
        _embcache[canon]=F.normalize(W[ids].float().mean(0),dim=-1).to(DEV) if ids else None
    return _embcache[canon]

@torch.inference_mode()
def prep(image):
    vl=proc(images=image,text=P_IMG,return_tensors="pt").to(DEV,torch.float16)
    vis=(vl.input_ids[0]==img_id).nonzero(as_tuple=True)[0]
    o=model(**vl,output_hidden_states=True,use_cache=False)
    hn=F.normalize(norm(o.hidden_states[31][0,vis,:]).float(),dim=-1)   # (576,d)
    return vl,hn
def gc_of(hn,canon):
    e=obj_embed(canon)
    if e is None: return 0.0
    g=(hn@e).topk(min(a.topk,hn.shape[0])).values.mean().item()
    return g-cal.get(canon,calmean)

@torch.inference_mode()
def decode_from(vl,prefix_ids,banned):
    """从 prompt(含image) + prefix_ids 续写 greedy 到 EOS, 过程中 ban 掉 banned 首token。返回生成token列表(含prefix)。"""
    if prefix_ids:
        ids=torch.cat([vl.input_ids,torch.tensor([prefix_ids],device=DEV)],dim=1)
        am=torch.ones_like(ids)
        o=model(input_ids=ids,pixel_values=vl.pixel_values,attention_mask=am,use_cache=True)
    else:
        o=model(**vl,use_cache=True)
    pk=o.past_key_values; last=o.logits[0,-1,:].float()
    gen=list(prefix_ids); bl=list(banned)
    for _ in range(a.max_new_tokens-len(gen)):
        lg=last.clone()
        if bl: lg[bl]=-float("inf")
        nxt=int(lg.argmax())
        if nxt==eos: break
        gen.append(nxt)
        o=model(input_ids=torch.tensor([[nxt]],device=DEV),past_key_values=pk,use_cache=True)
        pk=o.past_key_values; last=o.logits[0,-1,:].float()
    return gen

def tok_char_offsets(ids):
    return [len(tok.decode(ids[:i+1],skip_special_tokens=True)) for i in range(len(ids))]
def char_to_tok(offs,cp):
    for i,o in enumerate(offs):
        if o>cp: return i
    return len(offs)
def earliest_char(cap_low,surfs):
    best=None
    for sf in surfs:
        for m in re.finditer(r'\b'+re.escape(sf)+r's?\b',cap_low):
            if best is None or m.start()<best: best=m.start()
    return best

@torch.inference_mode()
def run(image):
    vl,hn=prep(image)
    banned=set()
    gen=decode_from(vl,[],banned)
    ncorr=0
    for _ in range(a.max_iters):
        cap=tok.decode(gen,skip_special_tokens=True)
        words,node,idxs,_=cu.caption_to_words(cap)
        if not node: break
        uniq=set(node)
        bad=[o for o in uniq if gc_of(hn,o)<a.tau_strict]
        if not bad: break
        low=cap.lower(); cand=[]
        for o in bad:
            surfs={words[k] for k in range(len(node)) if node[k]==o}|canon2surf.get(o,set())
            c=earliest_char(low,surfs)
            if c is not None: cand.append((c,o))
        if not cand: break
        cand.sort(); start_char,target=cand[0]
        offs=tok_char_offsets(gen); t=char_to_tok(offs,start_char)
        newban=canon2ftoks.get(target,set())
        if newban<=banned and t>=len(gen): break        # 无进展保护
        banned|=newban
        gen=decode_from(vl,gen[:t],banned); ncorr+=1
    return tok.decode(gen,skip_special_tokens=True).strip(),ncorr

ds=load_dataset("tsunghanwu/mscoco_chair",split="test",token=False)
_sp=a.split
idxs=[i for i in range(len(ds)) if _sp=="all" or (_sp=="test" and i%2==1) or (_sp=="cal" and i%2==0)][:a.n]
fout=open(a.out,"w");res=[];tot_corr=0;n=len(idxs)
print(f"[sc] {a.label} tau_strict={a.tau_strict} max_iters={a.max_iters} split={_sp} n={n}",flush=True)
for c,i in enumerate(idxs):
    cap,nc=run(ds[i]["image"].convert("RGB"));_,node,_,_=cu.caption_to_words(cap);tot_corr+=nc
    res.append({"answer":list(ds[i]["gt_object"]),"pred":node});fout.write(json.dumps({"image_id":i,"caption":cap,"pred":node,"ncorr":nc})+"\n")
    if (c+1)%25==0:print(f"[sc] {c+1}/{n} (avg corrections={tot_corr/(c+1):.2f})",flush=True)
fout.close()
cs=cu.coco_cap_chair_aggregate_results_chair_s(res);ci=cu.coco_cap_chair_aggregate_results_chair_i(res);rc=cu.coco_cap_chair_aggregate_results_recall(res)
print(f"\n[sc] {a.label} CHAIR_s={cs:.2f} CHAIR_i={ci:.2f} recall={rc:.2f} objs={sum(len(x['pred']) for x in res)/len(res):.2f} avg_corr={tot_corr/n:.2f} (base 50.4/15.4/76.9)",flush=True)
