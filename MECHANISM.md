# Mechanism: object hallucination is norm-driven, not direction-driven

A causal probe of *where* an object hallucination lives in the visual tokens. We take a detected
hallucinated object *o*, retrieve (via the grounding logit-lens) the top-k visual tokens that most support
it, perturb only those tokens (at the `multi_modal_projector` output, before the LM), regenerate the caption,
and measure the change in CHAIR. LLaVA-1.5-7B, CHAIR test split, n=250, 72 flagged instances, identical across rows.

| perturbation of the supporting visual tokens | CHAIR_s | ΔCHAIR_s | CHAIR_i |
|---|---|---|---|
| baseline (no edit) | 52.4 | — | 15.7 |
| **scale norm → 0%** (zero out) | 45.6 | **−6.8** | 13.0 |
| scale norm → 30% | 49.2 | −3.2 | 15.1 |
| scale norm → 60% | 52.0 | −0.4 | 15.0 |
| **project off the object's semantic direction** (norm preserved) | 52.0 | **−0.4 ≈ 0** | 15.6 |

## Findings
1. **Monotone dose–response in norm.** Shrinking the supporting tokens' norm suppresses the hallucination
   monotonically (keep 0% → −6.8, 30% → −3.2, 60% → −0.4). Magnitude is a causal lever.
2. **Direction is not a lever.** Removing the component of the visual tokens along the object's
   unembedding (logit-lens readout) direction — while preserving their norm — has **no effect** (−0.4, same as
   the norm-preserving scale-0.6). You cannot surgically excise the "object-ness" direction from a patch.
3. **Targeting is partially specific (matched control).** Selecting the hallucination's *own* top-k support tokens
   vs. the *same number* of random visual tokens, both at matched threshold (τ=−0.03, 82 flagged instances, scale→0):
   targeted → **−8.0 CHAIR_s** (52.4→44.4), random → **−4.0** (52.4→48.4). So targeting the grounding-identified
   support tokens is **~2× more effective** than perturbing random patches — but random is *not* null. Part of the
   effect is generic (shrinking any visual-token norms mildly degrades over-generation), part is specific to the
   grounding-selected tokens. Honest read: the grounding localization helps (2×), but the norm lever is not perfectly
   surgical.
4. **Interpretation.** The hallucinated object is not localized to a removable semantic direction inside its
   supporting visual tokens; it is bound to those tokens' overall **presence/magnitude (radial component)**. This
   is thematically consistent with work showing the *radial / norm* structure of CLIP-style embeddings is
   non-trivial and information-bearing (e.g. Levi & Gilboa, "The Double-Ellipsoid Geometry of CLIP," ICML 2025 /
   arXiv:2411.14517, where image/text embeddings lie on separable, off-origin ellipsoid *shells* tied to
   uncertainty) — though that work is about CLIP contrastive geometry, not LVLM decoding, so we claim only a
   thematic link, not their result. Our finding also explains why EAZY-style visual-feature *zeroing* (= our
   norm→0) works, and adds the mechanism EAZY does not report: suppression scales **monotonically with norm
   reduction**, and is **null for a norm-preserving semantic-direction edit**.

## Why this matters for positioning
- **Detection** here is *semantic* (project visual tokens onto the vocabulary/unembedding = "what does this
  patch say it is"), in contrast to attention-based hallucination detectors.
- **The control mechanism** is characterized *geometrically* (norm-causal), not as an attention edit — a
  different axis of explanation from the attention-centric literature.

Script: `method/vtablate.py` (`--mode zero|scale|project`, `--scale`). This is an analysis/mechanism probe;
the deployable control methods are in the main README (best-of-N rerank, visual-token neutralization, excision).
