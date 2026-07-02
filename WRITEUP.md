# Training-free object-hallucination control for LVLMs: a detection-bounded frontier and a base-rate-aware fix

**Model:** LLaVA-1.5-7B (llava-hf). **Benchmarks:** CHAIR (500 COCO val2014, greedy, max_new_tokens=512,
prompt "Please describe this image in detail."), POPE (adv/pop/random, standard yes/no protocol). All methods
**training-free** (no fine-tuning). Splits: cal = even image idx, test = odd (calibration quantities — protect
list, conformal γ — are derived on cal and evaluated on test).

## 0. TL;DR (honest)
1. A **training-free per-image grounding detector** (logit-lens) detects hallucinated mentioned-objects at **AUROC 0.82**.
2. We show **elimination is fundamentally detection-precision-bound**: surgical excision ≡ soft-suppression ≡ the
   detector's precision–recall frontier. Changing the *realization* buys nothing; an orthogonal causal signal adds
   +0.003 AUROC. The recall-safe frontier for detector-driven suppression tops out at **CHAIR_s −7**.
3. We **diagnose the binding constraint**: a single global threshold across objects with heterogeneous
   grounding-scale and base-rate. One class — **person (5.8% hallucination rate) — causes 65% of the false-positive
   suppressions** (real people get low grounding cosine → flagged → wrongly removed → recall collapse).
4. **A stronger, training-free detector: self-verification (AUROC 0.89).** Asking the model "Is there a {o}?" for
   each object *it itself mentioned* catches its own hallucinations at **0.89** — beating our grounding (0.82) and the
   supervised competitor (0.88). The model "knows" but doesn't act during generation (a generation–verification gap).
5. **Fix = base-rate-aware removal + a decoupled fluent rewrite.** Protect low-hallucination-rate classes from removal;
   excise the rest; then a light LM **grammar-repair pass** (ban only the removed objects, "fix grammar, add nothing")
   restores fluency. **all-500: 50.4/15.4/76.9 → 40.8/12.0/74.5 (CHAIR_s −9.6, recall −2.4)**, and **GPT-judge blind
   fluency 4.4/5** (vs raw-excise 2.5, baseline 4.9). Beats prior soft-suppression (−7) AND is fluent AND POPE-neutral.
6. Mechanistic findings: (a) detector-driven **removal must be post-hoc** — decode-time suppression couples through the
   autoregressive trajectory; (b) **regeneration drifts** (loses real objects, adds new hallucinations) so it cannot
   match excision's surgical precision — hence *decouple* removal (excise, optimal) from fluency (repair, object-set-preserving).
6. **POPE is a dead end for all training-free decoders** (ours, PAI, the ICLR-Oral opponent all = baseline);
   discrimination-time, the model's own yes/no judgment already dominates any external grounding signal.

## 1. Detector (training-free, per-image)
For object word *o* with unembedding rows W_U[T(o)], visual-token hidden states h_vis ∈ R^{576×d} at layer L=31,
LN = final RMSNorm:
  g(o) = mean_topk_{p∈576} cos( LN(h_vis[p]) , mean_{t∈T(o)} W_U[t] ),   gc(o) = g(o) − ĝ_o  (per-object mean).
Detects hallucinated-vs-real mentioned objects at **AUROC 0.82**, validated as genuine per-image grounding
(object-identity prior alone 0.60; per-object demean raises to 0.82; freq–hallucination corr ≈ 0). The **cosine**
logit-lens works; the softmax-prob logit-lens does not (0.52). Scripts: detect/logit_lens_probe*.py, analyze_auroc.py.

**Self-verification detector (AUROC 0.89) — stronger, still training-free.** For each object *o* the model mentioned in
its own caption, ask it "Is there a {o} in the image?" and read the yes/no logit gap. This catches the model's own
hallucinations at **AUROC 0.8949** (n=1681; hard "model-says-no" precision 0.70), beating grounding (0.82) and the
supervised competitor (0.88). The model *knows* an object is unsupported when asked directly, but emits it anyway
during free generation — a **generation–verification gap**. Caveat: this higher AUROC does *not* translate into a
better CHAIR excision frontier (self-check+protect 43.4 vs grounding+protect 40.4 @ recall 74.7) — CHAIR_s is
caption-level and grounding's per-object calibration fits it better — so self-verification's value is as a
detector/flagger, not a better excision score. Script: method/selfverify.py, detect/selfverify.json.

## 2. Elimination is detection-precision-bound (4× verified)
- **Excision ≡ soft.** method/excise.py: removing flagged objects (gc<τ) from the count traces exactly the same
  CHAIR–recall curve as soft-suppression z'(o)=z(o)−β·[τ−gc(o)]₊ (both ≈ 43/13/74.5 at recall 74.5). The frontier IS
  the detector's PR curve; the realization is irrelevant.
- **Orthogonal signal doesn't help.** method/probe_cfg.py: a CFG image-effect signal Δ(o)=logprob_full(o)−logprob_textonly(o)
  (how much the image raised the object's emission logit) adds only **+0.003** AUROC over grounding — it is redundant,
  not orthogonal. Detection caps at ~0.82.
- **Idea sweep (adversarially verified).** truncate-and-regenerate = recall-crash/length-confound; gated contrastive
  decoding = no-op and already claimed (ECD/CATCH); causal severing = EAZY's headline mechanism; selective abstention
  (method/abstain.py) backfires (hedge words encourage enumeration → more objects) or degenerates to cheap-baseline.

## 3. Root cause: a global threshold on heterogeneous objects
At the operating point that removes half the hallucinations (recall 0.50, gc<−0.05), detector precision is only
**0.447** — for 139 true hallucinations removed, **172 real objects are wrongly removed**. That collateral is
dominated by **one class**:

| object | mentions | hallucination rate | share of false-positive removals |
|---|---|---|---|
| **person** | 292 | **0.058** | **65%** |
| bottle | 40 | 0.575 | — |
| handbag | 31 | 0.613 | — |

Real people receive low grounding-cosine (person is visually diverse; the "person" unembedding row aligns weakly with
any specific person's tokens), so a global threshold flags and removes them. **Excluding person raises detector AUROC
0.766→0.824; within-person AUROC is 0.857** (person is detectable — it is just on a different score scale). The cap is
a thresholding artifact, not a detection-power limit.

## 4. Fix: base-rate-aware post-hoc text-excision
Protect classes whose calibration-split hallucination rate is low (cal-derived, rate<0.12, n≥10:
`person, tennis racket, surfboard, sports ball, elephant, umbrella`) from removal; excise the remaining flagged
object mentions from the generated caption post-hoc (method/textexcise.py). CHAIR (500, greedy, mnt=512):

| method (training-free) | CHAIR_s | CHAIR_i | recall | note |
|---|---|---|---|---|
| baseline | 50.4 | 15.4 | 76.9 | — |
| soft-suppression (β=12), decode-time | 43.4 | 13.2 | 74.5 | −7 (fluent) |
| **base-rate-aware excision (τ=−0.06)** | **40.4** | **11.6** | **74.7** | **−10, recall held** |
| base-rate-aware excision (τ=−0.05) | 34.0 | 9.8 | 72.5 | −16.4, recall −4.4 |

Held-out (protect list from cal/even, evaluated on test/odd, n=250): real text-excision **41.2/12.2/72.6** vs
no-protect ≈43.4 at matched recall — the base-rate-aware protection adds **~3 CHAIR_s at matched recall on unseen data**.

**The gain is post-hoc only.** Decode-time soft/hard + the same protection (method/gground.py --protect), even at an
aggressive operating point (τ=0, β=20), does *not* beat plain soft (44.4 vs 43.6): decode interventions couple through
the autoregressive trajectory (not penalizing person → longer captions → more downstream hallucination), whereas
post-hoc excision is surgical and per-object independent. This is why the recall-safe frontier only moves post-hoc.

**Making it fluent — decoupled grammar repair (method/repair.py).** Regex deletion leaves grammatical seams
("Additionally, is located near…"); GPT-judge blind fluency of raw-excise is only **2.5/5**. We restore fluency with a
light LM repair pass: feed the broken excised text, **ban only the removed objects' tokens**, instruct "fix grammar,
add no new objects/details". This preserves the object set (CHAIR/recall ≈ excise) while restoring readability:
all-500 **40.8/12.0/74.5** (−9.6 CHAIR_s, recall −2.4), **GPT-judge blind fluency 4.4/5** (sonnet 4.55, haiku 4.33;
n=80) — near baseline (4.9), far above raw-excise (2.5). This decouples *what to remove* (excise, optimal precision)
from *fluency* (repair, object-set-preserving). Whole-caption **regeneration** was tried and drifts (instruction-rewrite
−4.4, ban-rewrite −6.4, both recall<74: regen loses real objects and adds new hallucinations). Over-constraining the
repair to force CHAIR≡excise exactly (**faithful** mode, ban all absent COCO objects) **crashes recall 74.7→68.0** — the
model needs rephrasing freedom; the non-faithful repair (ban only removed objects, accept +0.4 CHAIR) is the recall-safe
optimum. **POPE-neutral by construction** (post-hoc caption editing does not touch the yes/no discrimination task).

## 5. POPE — a dead end for all training-free decoders (standard protocol, n=500/split)
method/pope_eval.py (yes/no-constrained prompt, short decode):

| method | adv | pop | rand |
|---|---|---|---|
| baseline | 0.840 | 0.868 | 0.870 |
| ours (soft) | = baseline | = baseline | = baseline | (soft inactive on yes/no) |
| PAI (attn α0.2/0.5, cfg γ1.1/2) | 0.840 → 0.834 | ≈baseline | ≈baseline | (no-op / slightly negative) |
| opponent SGRS (their code) | ≈0.81 | — | — | (no-op) |

method/pope_ground.py (use grounding to decide yes/no): pure-grounding acc 0.707 << model baseline 0.837; the optimal
weight on grounding when combined with the model is **λ=0.00** — grounding adds *zero* over the model's own judgment.
**Elimination helps generation (CHAIR), not discrimination (POPE):** when asked "is there an X", the model invokes its
internal judgment, which already beats any external grounding signal. This is intrinsic, not a tuning failure — no
training-free decoder (PAI, the opponent) moves standard POPE either.

## 6. Optional: conformal certificate (theoretical add-on)
Conformal Risk Control (Learn-Then-Test) can calibrate the excision threshold τ to give a distribution-free
finite-sample guarantee E[CHAIR_i] ≤ α over exchangeable images. This is a certified operating point, not a better
number. NB: the earlier method/conformal_calib.py pools static per-object labels on baseline captions — the
exchangeable unit for a decode/edit-time knob must be the image + its corrected caption (fix pending).

## 7. Positioning / novelty (honest)
- **Detector:** training-free logit-lens per-object grounding (0.82) — vs supervised 2604.04863 (GPT-4o labels + XGBoost, 0.88)
  and EAZY (attention-based selection). Unsupervised, different signal.
- **Elimination:** base-rate-aware post-hoc excision. The novel ingredient is the **base-rate-aware protection**
  (a Bayesian prior over per-class hallucination rate that fixes the global-threshold collateral) — moving the
  recall-safe frontier −7→−10. Post-hoc self-correction is Woodpecker/LURE-adjacent, but those need GPT / a trained
  reviser; ours is training-free and uses its own grounding detector.
- **Structural result:** "training-free detector-driven hallucination elimination is precision-bounded, and the
  precision bottleneck is object-class heterogeneity" is, to our knowledge, a new characterization.
- **Honest limits:** raw-CHAIR magnitude is bounded by detection precision; the number is −10 recall-safe, not a
  SOTA crush; and the deployable realization is post-hoc text-editing with a grammar-cleanup dependency.

## 8. Artifacts
detect/ (probes, calibration, AUROC); method/ (excise.py frontier + --protect, textexcise.py deployable method,
probe_cfg.py orthogonal-signal test, pope_eval.py / pope_pai.py / pope_ground.py POPE, gground.py soft/hard/protect,
abstain.py, conformal_calib.py; excise_gc.json cache, protect_list.txt); COMPETITOR_ANALYSIS.md; GOAL.md; memory
[[eazy-project-findings]], [[chair-method-landscape]].
