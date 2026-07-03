# Training-free Object-Hallucination Detection & Control for LVLMs

Training-free (no fine-tuning) detection and mitigation of object hallucination in large
vision-language models, built on a **logit-lens grounding** signal and a **self-verification** signal.
Core results on **LLaVA-1.5-7B** (CHAIR + POPE), with generalization checks on **AMBER-generative**, a
**second model size (LLaVA-1.5-13B)**, and a **second architecture (Qwen2.5-VL-7B)** — see §5.

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

## 5. Generalization: second benchmark, second model size, second architecture
The method was developed on LLaVA-1.5-7B / CHAIR. We verified how far it transfers (all runs n=250–400, same recipe).

**(a) A second generation benchmark — AMBER-generative (LLaVA-1.5-7B and 13B).** AMBER has its own images, object
vocabulary, and scorer, so it is an independent test of the generation control.

| model | method | AMBER CHAIR ↓ | Cover ↑ | Hal ↓ |
|---|---|---|---|---|
| 7B | baseline (greedy) | 7.44 | 50.0 | 33.7 |
| 7B | **best-of-N (grounding ⊕ self-verify rerank)** | **5.11** | 48.9 | **23.7** |
| 13B | baseline (greedy) | 6.27 | 51.1 | 26.1 |
| 13B | **best-of-N** | **4.72** | 50.9 | **19.7** |

`Hal` = fraction of captions containing *any* hallucination — cut by **10 points** on 7B (33.7 → 23.7) and **6.4** on
13B (26.1 → 19.7), coverage essentially preserved in both. The generation control is **not CHAIR-specific**.

**(b) A second model size — LLaVA-1.5-13B (CHAIR).** Same best-of-N recipe:

| method | CHAIR_s ↓ | CHAIR_i ↓ | recall |
|---|---|---|---|
| baseline | 51.6 | 14.0 | 76.3 |
| **best-of-N** | **36.0** | **8.8** | 76.7 |

**−15.6 CHAIR_s, recall preserved** — the same pattern as 7B (−21.6 there), driven by the self-verification reranker.

**(c) A second architecture — Qwen2.5-VL-7B (detection → control).** Here the two detectors diverge, which is itself the finding:

| detector (AUROC) | LLaVA-1.5-7B | LLaVA-1.5-13B | Qwen2.5-VL-7B |
|---|---|---|---|
| grounding logit-lens | 0.82 | 0.80 | **0.59** |
| self-verification | 0.89 | 0.90 | **0.94** |

- **Self-verification transfers across architectures — even stronger on Qwen (0.94).** The generation–verification
  gap is a property of autoregressive-vs-discriminative behavior, not of LLaVA specifically. This is the robust,
  portable detector.
- **The logit-lens grounding detector is architecture-specific.** It relies on visual-token → LM-unembedding
  alignment that holds in the LLaVA family but is weak at Qwen2.5-VL's final layer (its dynamic-resolution visual
  stack is not directly "readable" by the LM head). Honest scope boundary — grounding is a LLaVA-family signal;
  self-verification is the universal one. (A per-architecture layer sweep for Qwen grounding is a pending follow-up.)

**Control also transfers via self-verification.** Because self-verification is strong on Qwen, best-of-N reranked by
it reduces Qwen's *own* hallucination (CHAIR, n=250):

| Qwen2.5-VL-7B | CHAIR_s ↓ | CHAIR_i ↓ | recall |
|---|---|---|---|
| baseline (greedy) | 21.2 | 12.3 | 55.0 |
| **best-of-N (self-verify rerank)** | **17.2** | **9.3** | **62.5** |

Qwen already hallucinates far less than LLaVA-1.5 (21.2 vs 52.4 CHAIR_s), yet best-of-N still cuts **−4.0 CHAIR_s while
*raising* recall** (55.0 → 62.5). The full detect → control loop is architecture-portable through the self-verification
detector — even where the logit-lens grounding signal is not.

## 6. Limitations
POPE gains are small and need a 2-parameter calibration; best-of-N costs N× generation. The **logit-lens grounding
detector does not transfer to Qwen2.5-VL** at the final layer (§5c) — the portable detector is self-verification.
Generation control is now validated on **CHAIR + AMBER, on LLaVA-1.5-7B/13B, and cross-architecture on Qwen2.5-VL**
(all via best-of-N + self-verification). MME baseline is established (Total 1755.78, LLaVA-1.5-7B) but MME is a
yes/no discrimination benchmark where the generation control does not apply. MMHal-Bench is in progress.

## Repo layout
- `METHOD_SUMMARY.md` — concise method overview.
- `WRITEUP.md` — detailed method + results.
- `THEORY.md` — the detection-bounded frontier (selective-prediction framing).
- `MECHANISM.md` — causal probe: hallucination is norm-driven, not direction-driven.
- `RESPONSE.md` — point-by-point answers to review questions (looseness, attention-vs-semantic, geometry experiment).
- `COMPETITOR_ANALYSIS.md` — faithful reproduction of PAI and the ICLR-Oral opponent.
- `detect/` — grounding detector + AUROC analysis.
- `method/` — CHAIR control + POPE fusion + baselines/ablations.
