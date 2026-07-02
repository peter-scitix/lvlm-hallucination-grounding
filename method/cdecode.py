#!/usr/bin/env python3
"""
Contrastive / detection-guided decoding for LLaVA-1.5 (llava_hf, transformers 4.57). Training-free.

Manual greedy loop with optional:
  --vcd W      contrastive decode:  logits = (1+W)*logits_vl - W*logits_txt   (subtract text-only prior)
  --detguide   detection-guided:    penalize first-tokens of LOW-GROUNDING candidate objects by --lam,
               where grounding g(o) = topk-mean cosine of visual tokens (layer L) to o's unembedding row,
               flagged when g(o) < --tau. (uses our AUROC-0.82 grounding detector)
Both can be combined. Scores CHAIR_s/i/recall with lmms-eval's caption_to_words (comparable to baseline).
"""
import argparse, importlib.util, json, os
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration

SYS = ("A chat between a curious user and an artificial intelligence assistant. "
       "The assistant gives helpful, detailed, and polite answers to the user's questions.")
P_IMG = f"{SYS} USER: <image>\nPlease describe this image in detail. ASSISTANT:"
P_TXT = f"{SYS} USER: \nPlease describe this image in detail. ASSISTANT:"

def chair_utils():
    p = "/volume/exploration/EvolvingLMMs/lmms-eval/lmms_eval/tasks/coco_cap_chair/utils.py"
    s = importlib.util.spec_from_file_location("cu", p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="baseline")          # label only
ap.add_argument("--vcd", type=float, default=0.0)      # contrastive weight W (0=off)
ap.add_argument("--detguide", action="store_true")
ap.add_argument("--tau", type=float, default=0.07)     # grounding threshold (flag if g<tau)
ap.add_argument("--lam", type=float, default=8.0)      # logit penalty for flagged objects
ap.add_argument("--glayer", type=int, default=31)
ap.add_argument("--gtopk", type=int, default=10)
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--max_new_tokens", type=int, default=128)
ap.add_argument("--gpu", default="1")
ap.add_argument("--out", default="method/cd_out.jsonl")
a = ap.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
cu = chair_utils()
proc = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf"); tok = proc.tokenizer
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="cuda:0").eval()
W = model.lm_head.weight; norm = model.model.language_model.norm
img_id = model.config.image_token_index; eos = tok.eos_token_id
DEV = "cuda:0"

# candidate objects + unembedding dirs + first-token ids (for detguide)
cand = list(json.load(open("/volume/exploration/EvolvingLMMs/detect/calibration.json"))["per_object_mean_grounding"].keys())
objE, first_tok = [], {}
for o in cand:
    ids = tok.encode(" " + o, add_special_tokens=False) or tok.encode(o, add_special_tokens=False)
    objE.append(F.normalize(W[ids].float().mean(0), dim=-1)); first_tok[o] = ids[0]
objE = torch.stack(objE).to(DEV)                       # (C,hidden)

@torch.inference_mode()
def grounding_penalty(vl):
    out = model(**vl, output_hidden_states=True, use_cache=False)
    vis = (vl.input_ids[0] == img_id).nonzero(as_tuple=True)[0]
    hn = F.normalize(norm(out.hidden_states[a.glayer][0, vis, :]).float(), dim=-1)   # (576,hidden)
    cos = hn @ objE.T                                                                 # (576,C)
    g = cos.topk(min(a.gtopk, cos.shape[0]), dim=0).values.mean(0)                    # (C,) per-object grounding
    pen = torch.zeros(W.shape[0], device=DEV)
    flagged = 0
    for ci, o in enumerate(cand):
        if g[ci].item() < a.tau:
            pen[first_tok[o]] = max(pen[first_tok[o]].item(), a.lam); flagged += 1
    return pen, flagged

@torch.inference_mode()
def generate(image):
    vl = proc(images=image, text=P_IMG, return_tensors="pt").to(DEV, torch.float16)
    pen = None
    if a.detguide:
        pen, _ = grounding_penalty(vl)
    vlo = model(**vl, use_cache=True)
    vpk = vlo.past_key_values; logits_vl = vlo.logits[:, -1, :].float()
    if a.vcd > 0:
        tt = tok(P_TXT, return_tensors="pt").to(DEV)
        to = model(input_ids=tt.input_ids, attention_mask=tt.attention_mask, use_cache=True)
        tpk = to.past_key_values; logits_tx = to.logits[:, -1, :].float()
    gen = []
    for _ in range(a.max_new_tokens):
        logits = (1 + a.vcd) * logits_vl - a.vcd * logits_tx if a.vcd > 0 else logits_vl
        if pen is not None:
            logits = logits - pen
        nxt = int(logits.argmax())
        if nxt == eos:
            break
        gen.append(nxt)
        t = torch.tensor([[nxt]], device=DEV)
        vlo = model(input_ids=t, past_key_values=vpk, use_cache=True)
        vpk = vlo.past_key_values; logits_vl = vlo.logits[:, -1, :].float()
        if a.vcd > 0:
            to = model(input_ids=t, past_key_values=tpk, use_cache=True)
            tpk = to.past_key_values; logits_tx = to.logits[:, -1, :].float()
    return tok.decode(gen, skip_special_tokens=True).strip()

ds = load_dataset("tsunghanwu/mscoco_chair", split="test", token=False)
fout = open(a.out, "w"); results = []; n = min(a.n, len(ds))
print(f"[cd] mode={a.mode} vcd={a.vcd} detguide={a.detguide} tau={a.tau} lam={a.lam} n={n}", flush=True)
for i in range(n):
    cap = generate(ds[i]["image"].convert("RGB"))
    _, node, _, _ = cu.caption_to_words(cap)
    rec = {"answer": list(ds[i]["gt_object"]), "pred": node}
    results.append(rec); fout.write(json.dumps({"image_id": i, "caption": cap, **rec}) + "\n")
    if (i + 1) % 25 == 0: print(f"[cd] {i+1}/{n}", flush=True)
fout.close()
cs = cu.coco_cap_chair_aggregate_results_chair_s(results); ci = cu.coco_cap_chair_aggregate_results_chair_i(results); rc = cu.coco_cap_chair_aggregate_results_recall(results)
avglen = sum(len(r["pred"]) for r in results) / len(results)
print(f"\n[cd] mode={a.mode}  CHAIR_s={cs:.2f}  CHAIR_i={ci:.2f}  recall={rc:.2f}  avg_objs/cap={avglen:.2f}  (base ref 52.20/16.18/77.19)", flush=True)
