#!/usr/bin/env python3
"""
Method idea #1 — VL-vs-Text logit-diff STEERING (training-free, per-image) for LLaVA-1.5 (llava_hf).

Direction (compute_dir_logit_diff), computed once from the prompt's next-token logits:
    logits_diff = logits_vl - logits_txt                      # [vocab]  (image+text)  minus (text-only)
    dir_vocab   = sparse top-k of logits_diff (value = signed diff)
    dir_vec     = normalize( lm_head.weight^T @ dir_vocab )    # [hidden]  vocab-space dir -> hidden space
Inject during generation at one decoder layer:
    h' = h + alpha * dir_vec        (default layer 15, alpha 100)

Generates captions for the CHAIR images (baseline OR steered) and scores CHAIR_s/CHAIR_i/recall using
lmms-eval's own caption_to_words, so numbers are directly comparable to the baseline (52.2 / 16.18 / 77.2).
"""
import argparse, importlib.util, json, os
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration

SYS = ("A chat between a curious user and an artificial intelligence assistant. "
       "The assistant gives helpful, detailed, and polite answers to the user's questions.")
P_IMG = f"{SYS} USER: <image>\nPlease describe this image in detail. ASSISTANT:"
P_TXT = f"{SYS} USER: \nPlease describe this image in detail. ASSISTANT:"

def load_chair_utils():
    p = "/volume/exploration/EvolvingLMMs/lmms-eval/lmms_eval/tasks/coco_cap_chair/utils.py"
    spec = importlib.util.spec_from_file_location("chair_utils", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--mode", choices=["baseline", "steer"], default="steer")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--out", default="method/cap_steer.jsonl")
    return ap.parse_args()

def main():
    a = parse(); os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    cu = load_chair_utils()
    proc = AutoProcessor.from_pretrained(a.model); tok = proc.tokenizer
    model = LlavaForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.float16, device_map="cuda:0").eval()
    W = model.lm_head.weight                       # [vocab, hidden]
    dec_layers = model.model.language_model.layers
    ds = load_dataset("tsunghanwu/mscoco_chair", split="test", token=False)
    print(f"[steer] mode={a.mode} layer={a.layer} alpha={a.alpha} topk={a.topk} n={a.n}", flush=True)

    def dir_vec_for(image):
        vl = proc(images=image, text=P_IMG, return_tensors="pt").to("cuda:0", torch.float16)
        txt = tok(P_TXT, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            lv = model(**vl).logits[0, -1, :].float()
            lt = model(input_ids=txt.input_ids, attention_mask=txt.attention_mask).logits[0, -1, :].float()
        diff = lv - lt
        tv, ti = diff.abs().topk(a.topk)
        dv = torch.zeros_like(diff); dv[ti] = diff[ti]          # signed top-k
        vec = W.t().float() @ dv                                # [hidden]
        vec = vec / (vec.norm() + 1e-8)
        return vec.to(torch.float16), vl

    state = {"vec": None}
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h + a.alpha * state["vec"].to(h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h

    fout = open(a.out, "w"); results = []
    n = min(a.n, len(ds))
    for i in range(n):
        ex = ds[i]; image = ex["image"].convert("RGB")
        vec, vl = dir_vec_for(image)               # NO hook active here
        state["vec"] = vec
        handle = dec_layers[a.layer].register_forward_hook(hook) if a.mode == "steer" else None
        with torch.inference_mode():
            gen = model.generate(**vl, max_new_tokens=a.max_new_tokens, do_sample=False, num_beams=1)
        if handle: handle.remove()                 # hook active ONLY during generate
        cap = tok.decode(gen[0, vl.input_ids.shape[1]:], skip_special_tokens=True).strip()
        words, node_words, _, _ = cu.caption_to_words(cap)
        rec = {"answer": list(ex["gt_object"]), "pred": node_words, "image_id": i}
        results.append(rec)
        fout.write(json.dumps({"image_id": i, "caption": cap, **rec}) + "\n")
        if (i + 1) % 25 == 0:
            print(f"[steer] {i+1}/{n}", flush=True)
    fout.close()
    if handle: handle.remove()

    cs = cu.coco_cap_chair_aggregate_results_chair_s(results)
    ci = cu.coco_cap_chair_aggregate_results_chair_i(results)
    rc = cu.coco_cap_chair_aggregate_results_recall(results)
    print(f"\n[steer] mode={a.mode} n={n}  CHAIR_s={cs:.2f}  CHAIR_i={ci:.2f}  recall={rc:.2f}", flush=True)
    print(f"[steer] (baseline ref: CHAIR_s=52.20 CHAIR_i=16.18 recall=77.19)", flush=True)

if __name__ == "__main__":
    main()
