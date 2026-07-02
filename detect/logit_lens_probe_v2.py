#!/usr/bin/env python3
"""
Detection probe v2 — CONTRASTIVE / object-relative logit-lens grounding (the faithful form of signal #2).

For each visual token v we compute cosine (in unembedding space) to EVERY candidate COCO object, then
score a mentioned object o RELATIVE to the other objects at that token. Intuition (user's): a faithful
object "wins" (is the top / high-margin object) at some visual token that looks at its region; a
hallucinated object never wins at any token.

Per object o, per layer L:
  abs_peak      = max_v cos(v,o)                                  (v1 absolute baseline, for comparison)
  margin_peak   = max_v [ cos(v,o) - max_{o'!=o} cos(v,o') ]      (does o beat all other objects somewhere)
  margin_tkmean = mean of top-k margins
  rank1_frac    = fraction of visual tokens where o is the argmax object  (>0 => o dominates some patch)
  loose_margin  = mean pairwise cosine among the top-k tokens by margin   (signal #1 on the contrastive set)
Label: o not in gt `answer` -> hallucinated (positive=1).
"""
import argparse, json, os
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration

VICUNA_SYS = ("A chat between a curious user and an artificial intelligence assistant. "
              "The assistant gives helpful, detailed, and polite answers to the user's questions.")
PROMPT = f"{VICUNA_SYS} USER: <image>\nPlease describe this image in detail. ASSISTANT:"

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--samples", default="results/llava15_7b_baseline/llava-hf__llava-1.5-7b-hf/20260630_153532_samples_coco_cap_chair.jsonl")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--layers", default="16,20,24,28,31")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--out", default="detect/probe_v2.jsonl")
    return ap.parse_args()

def main():
    a = parse(); os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    layers = [int(x) for x in a.layers.split(",")]
    labels = {}
    cand = set()
    for line in open(a.samples):
        d = json.loads(line); pr = d.get("coco_cap_chair_s", {})
        ans = list(pr.get("answer", [])); pred = list(pr.get("pred", []))
        labels[d["doc_id"]] = {"answer": ans, "pred": pred}
        cand.update(ans); cand.update(pred)
    cand = sorted(cand)                                   # candidate object label space (canonical COCO names present)
    cidx = {o: i for i, o in enumerate(cand)}
    print(f"[v2] {len(labels)} docs, {len(cand)} candidate objects, layers={layers}", flush=True)

    proc = AutoProcessor.from_pretrained(a.model); tok = proc.tokenizer
    model = LlavaForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.float16, device_map="cuda:0").eval()
    img_token_id = model.config.image_token_index
    norm = model.model.language_model.norm; W = model.lm_head.weight

    # candidate object directions in unembedding space (mean subword row, normalized)
    objE = []
    for o in cand:
        ids = tok.encode(" " + o, add_special_tokens=False) or tok.encode(o, add_special_tokens=False)
        objE.append(F.normalize(W[ids].float().mean(0), dim=-1))
    objE = torch.stack(objE)                              # (C, hidden) on cuda

    ds = load_dataset("tsunghanwu/mscoco_chair", split="test", token=False)
    fout = open(a.out, "w"); nrows = 0; n = min(a.n, len(ds))
    for i in range(n):
        lab = labels.get(i)
        if not lab or not lab["pred"]:
            continue
        answer = set(lab["answer"]); pred = list(dict.fromkeys(lab["pred"]))
        inputs = proc(images=ds[i]["image"].convert("RGB"), text=PROMPT, return_tensors="pt").to("cuda:0", torch.float16)
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        vis = (inputs["input_ids"][0] == img_token_id).nonzero(as_tuple=True)[0]
        for o in pred:
            if o not in cidx:
                continue
            j = cidx[o]; label = 1 if o not in answer else 0
            rec = {"image_id": i, "object": o, "hallucinated": label}
            for L in layers:
                hn = F.normalize(norm(out.hidden_states[L][0, vis, :]).float(), dim=-1)   # (576,hidden)
                cos = hn @ objE.T                                                          # (576,C)
                co = cos[:, j]                                                             # (576,) this object
                others = cos.clone(); others[:, j] = -1e9
                margin = co - others.max(dim=1).values                                     # (576,) o vs best-other
                argmax_obj = cos.argmax(dim=1)
                rank1 = (argmax_obj == j).float().mean().item()
                mv, mi = torch.topk(margin, min(a.topk, margin.numel()))
                sub = hn[mi]; sim = sub @ sub.T; k = sim.shape[0]
                loose = ((sim.sum() - k) / (k * (k - 1))).item() if k > 1 else 1.0
                rec[f"abs_peak_L{L}"] = round(co.max().item(), 5)
                rec[f"margin_peak_L{L}"] = round(margin.max().item(), 5)
                rec[f"margin_tk_L{L}"] = round(mv.mean().item(), 5)
                rec[f"rank1_frac_L{L}"] = round(rank1, 5)
                rec[f"loose_margin_L{L}"] = round(loose, 5)
            fout.write(json.dumps(rec) + "\n"); nrows += 1
        if (i + 1) % 50 == 0:
            print(f"[v2] {i+1}/{n}, {nrows} rows", flush=True)
    fout.close(); print(f"[v2] DONE {nrows} rows -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
