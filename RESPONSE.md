# Responses to review questions

Point-by-point answers to the questions raised in review, with the supporting numbers
(LLaVA-1.5-7B; CHAIR 500 COCO / test split n=250; POPE standard protocol).

## Q1. Did you test the *internal* similarity (looseness) of the supporting visual tokens? Is there a stronger internal signal?
Yes, quantified. The pairwise similarity among a hallucinated object's top-k supporting visual tokens
("looseness") is a **real but weak** signal — AUROC **0.66–0.67** at mid layers (L20–24), confirming that a
hallucination's supporting tokens are looser. But it is **weaker than grounding** (cosine logit-lens, 0.79 @L31)
and **redundant** with it: adding looseness to grounding in 5-fold CV logistic regression leaves AUROC at 0.788
(vs 0.788 for grounding alone). The softmax-probability variant is near-chance. **No internal-similarity signal we
found beats grounding; the semantic logit-lens is the workhorse.**

## Q2. The fine-grained token-grounding detector (CVPR'26, arXiv:2604.04863) is very close to ours.
Agreed it is close, but on a **different axis**: that work is **attention-based**; ours is **semantic** — we
project each visual token onto the vocabulary/unembedding ("what does this patch *say* it is") via a logit-lens,
never using attention weights. It is also detection-only, whereas we couple detection with control and a mechanism
analysis. Honest caveat: as a pure *detector* it is a strong competitor (supervised, 0.88); our grounding is 0.82
training-free, and our self-verification detector reaches 0.89.

## Q3. These competitors are attention-level; we should tell the story at the semantic level.
We already are, on two axes: (i) **detection** = logit-lens semantic read-out (not attention); (ii) the **control
mechanism** is characterized *geometrically/semantically* — hallucination is driven by the visual tokens' **norm**
(magnitude in representation space), not by attention weights (see Q4 / `MECHANISM.md`).

## Q4. For elimination, look at the geometry papers; run experiments — is there a similar phenomenon (perturb the hallucination's visual tokens so the VLM stops generating it)?
Yes — the phenomenon exists and we characterized it (`MECHANISM.md`, n=250, 72 flagged instances). Perturbing a
detected hallucination's supporting visual tokens:
- **Shrinking their norm suppresses the hallucination monotonically:** keep 0% / 30% / 60% of the norm → CHAIR_s
  −6.8 / −3.2 / −0.4.
- **Editing the semantic direction (project off the object's readout direction, norm preserved) does nothing** (≈0).
- **Conclusion:** the hallucination is bound to the supporting patches' **magnitude/presence**, not a removable
  semantic direction. This is thematically consistent with the radial/norm structure of CLIP embeddings
  (Double-Ellipsoid CLIP, arXiv:2411.14517 — thematic link only, not their result), and it gives the *mechanism*
  behind why EAZY-style visual-feature zeroing works.
- (FlashTrace is a text-LLM token-attribution tool — no visual tokens / no embedding edits — not directly relevant.)

## Q5. (SAGE / OpenReview VPOFg0VBFx) intra-image vs dataset-level.
We use the *intra-image* per-object calibration idea (subtract each object's own mean grounding), which lifts
detection AUROC 0.79 → 0.82 out-of-sample. We do **not** do dataset-level steering (which would collapse to a
global direction).

## Honest gaps to flag proactively
1. The CVPR attention detector is a genuine near-neighbor; our moat is *semantic detection + control + mechanism*, not raw detection AUROC.
2. POPE gains are small (+0.8%): the model's POPE baseline (0.84–0.87) is near-saturated. We show the field's
   *claimed* POPE gains do not reproduce (PAI is a no-op on standard POPE; the ICLR-Oral headline is unreproducible).
3. Results are LLaVA-1.5-7B / CHAIR-centered so far (detection cross-checked on 13B). Multi-model + AMBER are next.
