# Training-free Object-Hallucination Detection & Control for LVLMs

Training-free (no fine-tuning) detection and mitigation of object hallucination in large
vision-language models, built on a **logit-lens grounding** signal. All results on **LLaVA-1.5-7B**
(detection also validated on **LLaVA-1.5-13B**), CHAIR (500 COCO val2014, greedy, max_new_tokens=512)
and POPE (standard yes/no protocol).

> This is a research code + notes release for review. Model weights, COCO images, cloned baseline
> repos, and raw run outputs are **not** included. Scripts use absolute paths to the original
> workspace and are provided for reference/reproduction of the method, not turn-key execution.

## Results at a glance (honest)
| task | baseline | ours | read |
|---|---|---|---|
| **CHAIR** (generation) | 52.4 CHAIR_s | **30.8** (−21.6), CHAIR_i 15.7→8.8 | strong; beats faithfully-reproduced PAI at matched recall; the ICLR-Oral opponent's headline is unreproducible |
| **POPE** (discrimination) | 0.84 / 0.87 / 0.87 acc | +0.2 / +0.4 / **+0.8%** | small — baseline is near-saturated; *no* training-free method reliably improves standard POPE, and competitors' claimed POPE gains do not reproduce |

The split is intrinsic: our approach detects hallucinated objects and intervenes on *generation*, which is exactly
what CHAIR measures; POPE is *discrimination*, where the model's own yes/no is already near-optimal. Detection itself
is strong on both (grounding 0.82 / self-verification 0.89).

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
| **best-of-N + grounding⊕self-verify rerank** | sample N captions, pick the one both detectors rank least-hallucinated | **30.8** | **8.8** | 71.0 | strongest; fluent (a real sample); beats PAI (31.2@69.6) at higher recall |
| best-of-N, recall-preserving rerank | reward object coverage in the rerank | 45.2 | 12.8 | **76.8** | −7 CHAIR_s with recall *fully preserved* (elimination can't do this) |
| visual-source neutralization | zero the visual tokens that support a detected hallucination, regenerate | 43.6 | 12.5 | 74.0 | fluent, recall-safe, single mechanism |
| grounding excision (+ base-rate protect, + LM repair) | remove flagged object mentions, repair grammar | ~40 | ~12 | ~74 | fluent (GPT-judge 4.4/5 vs 2.5 for raw deletion) |

Reductions of up to **−21.6 CHAIR_s** (CHAIR_i nearly halved), recall-safe, and **fluent** (best-of-N selects a real
sampled caption). Because it *selects* rather than *removes*, best-of-N can cut hallucinations without the recall cost
of suppression — the recall-preserving variant even keeps recall at 76.8. Its **oracle upper bound is 22.0 / 5.58 /
recall 80.5**, so the sampled captions contain much better outputs and the remaining gap is reranker quality. Scripts:
`method/{beston,vtablate,excise,textexcise,repair,gground}.py`.

## 3. POPE (discrimination)
The model's **POPE baseline is already near-saturated**: acc **0.840 / 0.868 / 0.870** (adv/pop/rand), F1
0.829/0.855/0.857 — so there is little headroom, and POPE errors are mostly the model being *conservative*
(false-negatives). Our grounding signal has real presence-discrimination power (AUROC up to 0.94 on random) but is
largely **redundant** with the model's own yes/no. A **learned soft fusion** of the model's yes/no margin with the
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
- **Hallucination is norm-driven, not direction-driven** (`MECHANISM.md`). Perturbing a hallucination's supporting
  visual tokens: shrinking their **norm** suppresses it monotonically (→0% = −6.8 CHAIR_s, →30% = −3.2, →60% = −0.4),
  but removing the object's **semantic direction** while preserving norm does *nothing* (≈0). The object is bound to
  the patches' magnitude, not a removable direction — a mechanistic explanation for why visual-feature zeroing works.
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
- `MECHANISM.md` — causal probe: hallucination is norm-driven, not direction-driven.
- `RESPONSE.md` — point-by-point answers to review questions (looseness, attention-vs-semantic, geometry experiment).
- `COMPETITOR_ANALYSIS.md` — faithful reproduction of PAI and the ICLR-Oral opponent.
- `detect/` — grounding detector + AUROC analysis.
- `method/` — CHAIR control + POPE fusion + baselines/ablations.
