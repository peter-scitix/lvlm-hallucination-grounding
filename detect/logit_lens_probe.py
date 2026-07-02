#!/usr/bin/env python3
"""
Per-image logit-lens object-hallucination DETECTION probe for LLaVA-1.5 (llava_hf / transformers 4.57).

For each image we do ONE forward of (vicuna_prompt + image), grab the 576 visual-token hidden
states at several decoder layers, and for each COCO object the model MENTIONED in its caption we
compute two intrinsic, training-free signals (user's hypotheses):

  signal #2  PEAK grounding   = max over visual tokens of the object's logit-lens score
                                 (faithful object -> some token grounds it strongly -> high peak;
                                  hallucinated     -> no token grounds it          -> low peak)
  signal #1  LOOSENESS        = mean pairwise cosine among the top-k retrieved visual tokens
                                 (faithful -> tight cluster -> high pairwise sim;
                                  hallucinated -> scattered  -> low pairwise sim)

Two probe variants for the logit-lens score of visual token v wrt object word o:
  A "prob"   softmax(lm_head(norm(h_v)))[first_subword(o)]          (vocabulary projection prob)
  B "cos"    cosine( norm(h_v), mean_subword lm_head.weight[o] )    (cosine to unembedding row)

Ground truth from the lmms-eval CHAIR samples: object in `answer` -> faithful, else hallucinated.

Outputs a jsonl of per-(image,object) rows with scores at every probed layer -> analyze AUROC separately.
"""
import argparse, json, os, sys
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration

VICUNA_SYS = ("A chat between a curious user and an artificial intelligence assistant. "
              "The assistant gives helpful, detailed, and polite answers to the user's questions.")
PROMPT = f"{VICUNA_SYS} USER: <image>\nPlease describe this image in detail. ASSISTANT:"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--samples", default="results/llava15_7b_baseline/llava-hf__llava-1.5-7b-hf/20260630_153532_samples_coco_cap_chair.jsonl")
    ap.add_argument("--n", type=int, default=20, help="num images (smoke=20, full=500)")
    ap.add_argument("--layers", default="4,8,12,14,16,20,24,31")
    ap.add_argument("--topk", type=int, default=10, help="top-k visual tokens retrieved per object")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out", default="detect/probe_scores.jsonl")
    return ap.parse_args()

def load_labels(path):
    """doc_id -> dict(answer=[gt objs], pred=[mentioned objs])."""
    lab = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            pr = d.get("coco_cap_chair_s", {})
            lab[d["doc_id"]] = {"answer": list(pr.get("answer", [])), "pred": list(pr.get("pred", []))}
    return lab

def main():
    a = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    layers = [int(x) for x in a.layers.split(",")]
    labels = load_labels(a.samples)
    print(f"[probe] loaded labels for {len(labels)} docs; probing layers={layers} topk={a.topk} n={a.n}", flush=True)

    proc = AutoProcessor.from_pretrained(a.model)
    tok = proc.tokenizer
    model = LlavaForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.float16, device_map="cuda:0")
    model.eval()
    img_token_id = model.config.image_token_index
    norm = model.model.language_model.norm
    lm_head = model.lm_head                       # (vocab, 4096), no bias, untied
    W = lm_head.weight                            # (vocab, hidden)
    print(f"[probe] img_token_id={img_token_id} hidden={W.shape[1]} vocab={W.shape[0]} layers_avail={model.config.text_config.num_hidden_layers}", flush=True)

    ds = load_dataset("tsunghanwu/mscoco_chair", split="test", token=False)

    def obj_token_ids(o):
        # mid-sentence form (leading space) is what the model actually emits
        ids = tok.encode(" " + o.strip(), add_special_tokens=False)
        return ids if ids else tok.encode(o.strip(), add_special_tokens=False)

    fout = open(a.out, "w")
    n_rows = 0
    n = min(a.n, len(ds))
    for i in range(n):
        ex = ds[i]
        lab = labels.get(i)
        if lab is None:
            continue
        answer = set(lab["answer"]); pred = list(dict.fromkeys(lab["pred"]))  # dedup, keep order
        if not pred:
            continue
        img = ex["image"].convert("RGB")
        inputs = proc(images=img, text=PROMPT, return_tensors="pt").to("cuda:0", torch.float16)
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        ids = inputs["input_ids"][0]
        vis_pos = (ids == img_token_id).nonzero(as_tuple=True)[0]
        assert vis_pos.numel() == 576, f"img {i}: got {vis_pos.numel()} visual tokens"

        # precompute per-layer normed visual hidden states once
        hn_by_layer = {}
        for L in layers:
            h = out.hidden_states[L][0, vis_pos, :]          # (576, hidden) fp16
            hn = norm(h)                                     # RMSNorm before lens (REQUIRED)
            hn_by_layer[L] = hn
        # logit-lens full-vocab logits only where needed (variant A) -> compute per layer
        logits_by_layer = {L: lm_head(hn_by_layer[L]).float() for L in layers}   # (576, vocab)
        probs_by_layer = {L: logits_by_layer[L].softmax(-1) for L in layers}

        for o in pred:
            oid = obj_token_ids(o)
            if not oid:
                continue
            first = oid[0]
            row = F.normalize(W[oid].float().mean(0, keepdim=True), dim=-1)   # (1,hidden) object direction (mean subword)
            label = 1 if o not in answer else 0                              # 1 = hallucinated (positive)
            rec = {"image_id": i, "object": o, "n_subword": len(oid),
                   "hallucinated": label, "in_answer": (o in answer)}
            for L in layers:
                hn = hn_by_layer[L]                                          # (576,hidden)
                # variant A: prob of object's first subword across visual tokens
                pa = probs_by_layer[L][:, first]                             # (576,)
                # variant B: cosine of normed hidden state to object unembedding row
                hnn = F.normalize(hn.float(), dim=-1)                        # (576,hidden)
                cb = (hnn @ row.T).squeeze(-1)                               # (576,)
                for name, score in (("A", pa), ("B", cb)):
                    peak = score.max().item()
                    tkvals, tk = torch.topk(score, min(a.topk, score.numel()))
                    tkmean = tkvals.mean().item()                           # mean of top-k scores (robust peak)
                    sub = F.normalize(hn[tk].float(), dim=-1)               # (k,hidden)
                    sim = sub @ sub.T                                       # (k,k)
                    k = sim.shape[0]
                    loose = ((sim.sum() - k) / (k * (k - 1))).item() if k > 1 else 1.0  # mean off-diag pairwise cos
                    rec[f"peak{name}_L{L}"] = round(peak, 5)
                    rec[f"tk{name}_L{L}"] = round(tkmean, 5)
                    rec[f"loose{name}_L{L}"] = round(loose, 5)
            fout.write(json.dumps(rec) + "\n"); n_rows += 1
        if (i + 1) % 25 == 0:
            print(f"[probe] {i+1}/{n} images, {n_rows} object-rows", flush=True)
    fout.close()
    print(f"[probe] DONE: {n_rows} object-rows -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
