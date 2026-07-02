#!/usr/bin/env python3
"""Calibrate the steering: load model once, compute each image's dir_vec once, sweep alpha (and layer)
and report CHAIR_s/i/recall + a sample caption per config. dir_vec is layer/alpha-independent (hidden-space)."""
import argparse, importlib.util, json, os
import torch
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration

SYS = ("A chat between a curious user and an artificial intelligence assistant. "
       "The assistant gives helpful, detailed, and polite answers to the user's questions.")
P_IMG = f"{SYS} USER: <image>\nPlease describe this image in detail. ASSISTANT:"
P_TXT = f"{SYS} USER: \nPlease describe this image in detail. ASSISTANT:"

def load_chair_utils():
    p = "/volume/exploration/EvolvingLMMs/lmms-eval/lmms_eval/tasks/coco_cap_chair/utils.py"
    spec = importlib.util.spec_from_file_location("chair_utils", p); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--layer", type=int, default=15)
ap.add_argument("--alphas", default="0,2,4,8,16,32")
ap.add_argument("--topk", type=int, default=10)
ap.add_argument("--max_new_tokens", type=int, default=128)
ap.add_argument("--gpu", default="1")
a = ap.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
alphas = [float(x) for x in a.alphas.split(",")]
cu = load_chair_utils()
proc = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf"); tok = proc.tokenizer
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="cuda:0").eval()
W = model.lm_head.weight; dec = model.model.language_model.layers
ds = load_dataset("tsunghanwu/mscoco_chair", split="test", token=False)

state = {"vec": None, "alpha": 0.0}
def hook(m, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    h = h + state["alpha"] * state["vec"].to(h.dtype)
    return (h,) + out[1:] if isinstance(out, tuple) else h

def dir_vec(image):
    vl = proc(images=image, text=P_IMG, return_tensors="pt").to("cuda:0", torch.float16)
    txt = tok(P_TXT, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        lv = model(**vl).logits[0, -1, :].float()
        lt = model(input_ids=txt.input_ids, attention_mask=txt.attention_mask).logits[0, -1, :].float()
    diff = lv - lt; tv, ti = diff.abs().topk(a.topk)
    dv = torch.zeros_like(diff); dv[ti] = diff[ti]
    v = W.t().float() @ dv; v = v / (v.norm() + 1e-8)
    return v.to(torch.float16), vl

n = min(a.n, len(ds))
res = {al: [] for al in alphas}; caps = {al: None for al in alphas}
hidden_norm_seen = []
for i in range(n):
    image = ds[i]["image"].convert("RGB"); v, vl = dir_vec(image); state["vec"] = v
    for al in alphas:
        state["alpha"] = al
        handle = dec[a.layer].register_forward_hook(hook) if al != 0 else None
        with torch.inference_mode():
            gen = model.generate(**vl, max_new_tokens=a.max_new_tokens, do_sample=False, num_beams=1)
        if handle: handle.remove()
        cap = tok.decode(gen[0, vl.input_ids.shape[1]:], skip_special_tokens=True).strip()
        words, node, _, _ = cu.caption_to_words(cap)
        res[al].append({"answer": list(ds[i]["gt_object"]), "pred": node})
        if i == 0: caps[al] = cap
    if (i + 1) % 10 == 0: print(f"  {i+1}/{n}", flush=True)

# reference layer-15 hidden norm (one forward) for scale intuition
with torch.inference_mode():
    vl = proc(images=ds[0]["image"].convert("RGB"), text=P_IMG, return_tensors="pt").to("cuda:0", torch.float16)
    hs = model(**vl, output_hidden_states=True).hidden_states[a.layer][0]
print(f"\n[scale] layer{a.layer} hidden-state L2 norm: mean={hs.norm(dim=-1).mean():.1f} (alpha is added on a UNIT dir_vec)\n")
print(f"{'alpha':>6}  {'CHAIR_s':>8} {'CHAIR_i':>8} {'recall':>7}   sample caption (img0)")
for al in alphas:
    r = res[al]
    cs = cu.coco_cap_chair_aggregate_results_chair_s(r); ci = cu.coco_cap_chair_aggregate_results_chair_i(r); rc = cu.coco_cap_chair_aggregate_results_recall(r)
    print(f"{al:>6}  {cs:>8.2f} {ci:>8.2f} {rc:>7.2f}   {caps[al][:90]!r}")
print("\n(baseline ref full-500: CHAIR_s=52.20 CHAIR_i=16.18 recall=77.19)")
