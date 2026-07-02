# Training-free Object-Hallucination Detection & Control for LVLMs

Training-free (no fine-tuning) detection and mitigation of object hallucination in large
vision-language models, built on a **logit-lens grounding** signal. All results on **LLaVA-1.5-7B**
(detection also validated on **LLaVA-1.5-13B**), CHAIR (500 COCO val2014, greedy, max_new_tokens=512)
and POPE (standard yes/no protocol).

> This is a research code + notes release for review. Model weights, COCO images, cloned baseline
> repos, and raw run outputs are **not** included. Scripts use absolute paths to the original
> workspace and are provided for reference/reproduction of the method, not turn-key execution.

## 1. Detection (the foundation)
For a mentioned object *o*: project each of the 576 visual-token hidden states (final layer, RMSNorm'd)
onto *o*'s unembedding row, take the top-k-mean cosine, per-object calibrated →
grounding score `gc(o)`. A real object has patches that strongly point to it; a hallucinated one does not.

- **Grounding logit-lens: AUROC 0.82** (500 imgs / ~1681 mentioned objects, 27% hallucinated). Validated as
  genuine per-image grounding (object-prior alone 0.60; per-object demean → 0.82). Cosine works; softmax-prob does not.
- **Self-verification: AUROC 0.89 (7B) / 0.895 (13B).** Directly asking the model "Is there an X?" detects its own
  caption hallucinations *better* than any feature detector — a **generation–verification gap** (the model "knows"
  but emits the object anyway during free generation).

Scripts: `detect/logit_lens_probe*.py`, `detect/analyze_auroc.py`.

## 2. CHAIR (generation) — detection-driven control
Several training-free mechanisms, all driven by the same grounding detector:

| method | mechanism | CHAIR_s | CHAIR_i | recall | notes |
|---|---|---|---|---|---|
| baseline | — | 52.4 | 15.7 | 75.0 | |
| best-of-N + grounding rerank | sample N captions, pick fewest hallucinations | **35.2** | **9.0** | 70.7 | biggest cut; recall −4.3 |
| visual-source neutralization | zero the visual tokens that support a detected hallucination, regenerate | 43.6 | 12.5 | 74.0 | fluent, recall-safe, single mechanism |
| grounding excision (+ base-rate protect, + LM repair) | remove flagged object mentions, repair grammar | ~40 | ~12 | ~74 | fluent (GPT-judge 4.4/5 vs 2.5 for raw deletion) |

Reductions of **−9 to −17 CHAIR_s** — competitive with or beating faithfully-reproduced baselines. Best-of-N's
**oracle upper bound is 23.6 / 5.45 / recall 81**, indicating the sampled captions contain much better outputs and the
bottleneck is the reranker's quality. Scripts: `method/{beston,vtablate,excise,textexcise,repair,gground}.py`.

## 3. POPE (discrimination)
POPE errors are mostly the model being *conservative* (false-negatives), and the model's own yes/no is already strong.
Our grounding signal is largely redundant there, but a **learned soft fusion** of the model's yes/no margin with the
grounding score adds a small, cross-validated gain:

| split | model only (CV AUROC) | + grounding (CV AUROC) | acc gain |
|---|---|---|---|
| adversarial | 0.918 | 0.922 | +0.2% |
| popular | 0.948 | 0.950 | +0.4% |
| random | 0.958 | **0.973** | **+0.8%** |

We also verify that multi-phrasing self-consistency (TACO-style) does **not** help our model, and that grounding cannot
be used as a hard POPE classifier (its ranking is good, AUROC up to 0.94, but its threshold does not transfer).
Scripts: `method/{pope_ground,pope_sc,pope_eval}.py`.

## 4. Key findings (honest)
- **Elimination is detection-precision-bounded.** Surgical excision ≡ soft-suppression ≡ the detector's precision–recall
  frontier; changing the intervention mechanism does not move it (only a better detector does). See `THEORY.md`.
- **Generation–verification gap:** the model self-verifies (0.89) better than feature detectors (0.82); cross-model.
- **Faithful competitor reproduction (`COMPETITOR_ANALYSIS.md`):** the ICLR-Oral opponent's headline is *not
  reproducible* from its release (missing module; the runnable part is a no-op); PAI genuinely reduces CHAIR (−8) but at
  a recall cost, and is a no-op on standard POPE. On standard POPE, no training-free decoder we tested reliably improves
  the baseline.

## 5. Limitations
Single model family (LLaVA-1.5-7B; detection cross-checked on 13B), CHAIR-centered; POPE gains are small and need a
2-parameter calibration; best-of-N costs N× generation. Broader models (InstructBLIP/Qwen-VL) and benchmarks (AMBER)
are future work.

## Repo layout
- `METHOD_SUMMARY.md` — concise method overview.
- `WRITEUP.md` — detailed method + results.
- `THEORY.md` — the detection-bounded frontier (selective-prediction framing).
- `COMPETITOR_ANALYSIS.md` — faithful reproduction of PAI and the ICLR-Oral opponent.
- `detect/` — grounding detector + AUROC analysis.
- `method/` — CHAIR control + POPE fusion + baselines/ablations.
