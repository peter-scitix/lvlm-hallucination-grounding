# Competitor analysis — LVLMs-Saliency ("Hallucination Begins Where Saliency Drops", ICLR 2026 Oral, arXiv 2601.20279)

## Their method
- **Signal:** token saliency = |Attn ⊙ Grad(Attn)| (attention × input-gradient), needs a **backward pass**.
- **SGRS** (Saliency-Guided Rejection Sampling): at decode, top-K candidates; accept the sampled one if its
  saliency ≥ τ = α·(mean saliency of recent accepted tokens); resample up to R times; fallback = max-saliency.
- **LocoRE** (Local Coherence Reinforcement): boost attention from the current token to recent predecessors
  (their `AttnAdapter`, applied to layers 9–14).
- Core observation ("hallucination = low grounding/saliency") ≈ **our** core observation.

## Their reported numbers (LLaVA-1.5-7B, from their Table 1)
| method | POPE F1/Acc | CHAIR C_s/C_i |
|---|---|---|
| baseline (Beam) | 85.4/84.0 | 51.0/15.2 |
| LocoRE | 86.9/87.3 | 38.4/11.2 |
| SGRS+LocoRE (theirs) | 87.0/87.5 | 35.6/~10 |
| EAH (their strongest CHAIR baseline) | 85.7/86.0 | 36.4/9.9 |

## FINDING 1 — their released code cannot reproduce their headline
`do_reject_sample.py` calls `AttnAdapter(...)` (the LocoRE module) at line 289, but **`AttnAdapter` is never
defined or imported anywhere in the repo.** So `use_visaug=True` (SGRS+LocoRE = the 35.6 config) **crashes**.
Only SGRS-only (`use_visaug=False`) is runnable. Their headline number is **not independently reproducible** from
the release.

## FINDING 2 — their headline ≈ a plainly-run PAI (our fair reproduction)
All methods re-run in OUR pipeline, identical protocol (LLaVA-1.5-7B, greedy, max_new_tokens=512, test-250, our
CHAIR scorer, baseline 51.6/15.8/74.9):
| method (our fair repro) | CHAIR_s | CHAIR_i | recall |
|---|---|---|---|
| baseline | 51.6 | 15.8 | 74.9 |
| **PAI (attention→image)** | **35.2** | **10.9** | 67.6 |
| LocoRE (reimpl from abstract) | ~52 | ~15 | ~75 (≈ null) |
| VCD (text-only contrast) | ~54 | 14.3 | ~79 (≈ null) |
| ours: grounding soft-suppression | 42.8 | 12.2 | 71.4 |
Their claimed 35.6/~10 ≈ our fairly-run PAI (35.2/10.9). Their "Oral SOTA" does **not** beat a standard
attention method under a controlled protocol. (Caveat: LocoRE/VCD are our reimplementations; nulls are
suggestive, not definitive — their official LocoRE code is missing.)

## FINDING 3 — CHAIR is dominated by caption length (the "compressed baseline" confound)
Same model, same 15 images, ONLY changing max_new_tokens:
| baseline | CHAIR_s | CHAIR_i | recall | objs/cap |
|---|---|---|---|---|
| mnt=64 | 20.0 | 6.67 | 74.4 | 5.0 |
| mnt=128 | 46.7 | 10.3 | 89.7 | 7.7 |
Shortening captions alone drops CHAIR_s 47→20. Any method that implicitly shortens output "reduces
hallucination" trivially; cross-paper CHAIR numbers at unstated/unequal lengths are not comparable. **All
comparisons must fix max_new_tokens.**

## FINDING 4 — faithful SGRS-only reproduction (their official code)
Ran their `do_reject_sample.py` (SGRS-only, use_visaug=False) via their original LLaVA env (isolated venv:
transformers 4.37.2 + haotian-liu LLaVA + liuhaotian/llava-v1.5-7b + CLIP). SGRS is ~4× slower than normal decode
(per-candidate gradient), so run at mnt=64 on a subset; compared to the **matched mnt=64 baseline (20.0/6.67/74.4)**.
Result: **SGRS-only is a NO-OP.** On the first 5 CHAIR images its captions are **byte-identical** to the
greedy baseline (identical=5/5), same CHAIR (40.0/5.7/76.5 = baseline). With their default params (α=0.6, K=5,
R=3) the saliency threshold τ=α·mean(recent) is always cleared by the top-1 candidate, so no rejection ever
fires → output == greedy. Since SGRS-only does nothing, their reported gain must come ENTIRELY from **LocoRE
— the module missing from their released code.** Net: the reproducible part of their method has zero effect,
and the effective part is unreproducible.


## FINDING 5 — faithful PAI (their OFFICIAL code, LALBJ/PAI) reproduced
Ran PAI's official chair_eval.py (attn α=0.5 + CFG γ=1.1, layers 2-32, mnt=512) + its own baseline, 100 CHAIR
imgs, original liuhaotian/llava-v1.5-7b, same pipeline:
| method (PAI official code) | CHAIR_s | CHAIR_i | recall |
|---|---|---|---|
| baseline | 37.0 | 11.8 | 74.2 |
| PAI (attn+cfg) | 29.0 | 8.7 | 65.4 |
PAI genuinely reduces CHAIR (37->29, -8) but costs recall (74.2->65.4, -8.7). Our earlier llava_hf reimpl (35.2)
slightly understated PAI (it omitted the CFG term). So: PAI is the ONE competitor whose gains are real+reproducible
-- and its weakness is recall. The Oral opponent's gains are NOT reproducible (SGRS no-op + LocoRE missing).

## Bottom line
- Opponent (ICLR Oral): headline unreproducible (LocoRE code missing); reproducible SGRS = no-op (= greedy).
- PAI (ECCV, official): real but recall-costly (-8.7 recall for -8 CHAIR_s).
- CHAIR is length-confounded (mnt64 vs mnt128 halves it) -- fix mnt across all methods.
- POPE: on the STANDARD protocol NO training-free decoder beats baseline (PAI no-op/slightly-negative, opponent SGRS
  ≈ baseline, ours = baseline); PAI's official POPE eval is degenerate (unconstrained prompt → 0.50 baseline). POPE is
  a wash for everyone — comparisons should not claim POPE gains.
- OUR angle: (1) training-free grounding detector 0.82; (2) the structural result that elimination is
  detection-precision-bound (excision ≡ soft ≡ PR frontier); (3) base-rate-aware post-hoc excision that diagnoses and
  fixes the person-dominated collateral, moving the recall-safe frontier from −7 to −10 CHAIR_s (held-out) — a
  reproducible, recall-preserving gain over plain suppression, without PAI's recall cost and without the opponent's
  reproducibility gap. Deployable realization is post-hoc (decode-time can't capture it); grammar-cleanup pending.

## FINDING 6 — implemented the MISSING LocoRE from the paper (Eq.8): also a NO-OP
Reimplemented LocoRE Eq.8 faithfully on llava_hf (post-softmax gain (1+β) on the last query's attention to the
recent w_s output tokens, renormalized, layers 9-14 per their code). First-100 CHAIR imgs, baseline-100 = 53.0/15.7/79.2:
| LocoRE β | CHAIR_s | CHAIR_i | recall |
|---|---|---|---|
| baseline | 53.0 | 15.7 | 79.2 |
| 0.15 (paper) | 53.0 | 15.5 | 78.5 |
| 0.5 | 49.0 | 14.2 | 79.9 |
| 1.0 | 52.0 | 14.0 | 80.5 |
| 2.0 (13x paper) | 51.0 | 13.6 | 78.5 |
At the paper's β=0.15, LocoRE = baseline (no effect). Even 13x stronger → only -2. NOWHERE near their 35.6.

## FINDING 7 — POPE: NO training-free decoder beats baseline (and PAI's official POPE eval is degenerate)
Faithfully reproduced POPE for all methods on the STANDARD protocol (yes/no-constrained prompt, short decode,
our validated method/pope_eval.py, n=500/split, baseline 0.840/0.868/0.870):
| method (standard POPE) | adv | pop | rand |
|---|---|---|---|
| baseline (re-run) | 0.840 | 0.868 | 0.870 |
| PAI attn-only (α=0.2) | 0.840 | 0.868 | 0.870 |  ← identical predictions to baseline |
| PAI full (α=0.2, γ=1.1) | 0.840 | 0.868 | 0.870 |
| PAI full (α=0.2, γ=2) | 0.834 | 0.862 | 0.866 |  ← slightly NEGATIVE |
| PAI strong attn (α=0.5) | 0.832 | 0.860 | 0.864 |  ← slightly negative |
| opponent SGRS (their code, SGRS-only) | ≈0.813 | — | — |  ← ≈ baseline (SGRS is the proven no-op) |
| ours (grounding soft) | =baseline | =baseline | =baseline |  (soft inactive on yes/no) |
On the standard protocol **no training-free decoder moves POPE** — PAI is a no-op or slightly negative in every
config; the opponent's SGRS ≈ baseline. POPE (a discrimination task with baseline already 0.84–0.87) is insensitive
to decode-time intervention. (Our method/pope_ground.py confirms the mechanism: using grounding to decide yes/no
gives optimal mixing weight λ=0.00 over the model's own judgment — grounding adds nothing to discrimination.)

**PAI's OFFICIAL POPE eval is DEGENERATE in faithful repro.** PAI's pope_eval.py uses a prompt WITHOUT the yes/no
constraint and decodes 512 tokens, so LLaVA-1.5 free-generates and over-affirms: baseline acc **0.504, yes-rate
0.992** ("Yes, there is a dog… riding a snowboard" for absent objects), reproduced identically on PAI's own original
adversarial data (sub-84). Adding PAI's strong CFG (γ=2) breaks the all-yes bias (yes-rate 0.99→0.59) but the eval
stays at random (acc 0.486 — it turns "always yes" into "describe instead of answer"). So PAI's reported POPE gains
exist only relative to its own broken (unconstrained-prompt) baseline; against a correctly-prompted 0.84 baseline PAI
adds nothing. This mirrors the CHAIR finding that the field's eval harnesses are finicky and must be re-run.

## FINAL VERDICT on LVLMs-Saliency (ICLR 2026 Oral)
Both components, faithfully reproduced from the paper, produce NO meaningful CHAIR reduction:
  - SGRS-only: byte-identical to greedy (no-op).
  - LocoRE (Eq.8, β=0.15): = baseline (no-op); negligible even at 13x β.
Their headline 51->35.6 is UNREPRODUCIBLE from the paper's described methods (and the combined code is missing).
Caveat: LocoRE is our faithful reimplementation; but across β=0.15..2.0 with a clean implementation the effect
is negligible, which is strong evidence the described mechanism cannot yield their headline.
By contrast, PAI (ECCV, official code) genuinely works (37->29) at a recall cost. Our method targets that recall gap.
