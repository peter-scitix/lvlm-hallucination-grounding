# Hallucination elimination is a selective-prediction problem: a risk–coverage bound

**Thesis.** Training-free, per-object, detection-driven object-hallucination *elimination* in LVLMs is exactly
**selective prediction over object mentions**. Its achievable (CHAIR_i, recall) frontier is a reparameterization of
the detector score's ROC curve, *independent of the intervention mechanism*. Hence the whole training-free family
(VCD / PAI / OPERA / EAZY / suppression / excision) is bounded by one quantity — the detector's ROC — and the only way
to move the frontier is a better score. The strongest **available** score is the model's own verification
("Is there an X?"), which realizes a **generation–verification gap** (AUROC 0.89 vs feature detectors 0.82) and
provably lowers the frontier.

## 1. Setup
For image x, the LVLM caption mentions a deduplicated set of object classes M(x). Each mention o∈M(x) has a latent
label h(o)∈{0,1} (1 = hallucinated: o∉objects(x)). Over the population of mentions, let π = P(h=1) (base rate).
A **detector** assigns a score s(o) with higher = more likely hallucinated. An **elimination method** outputs a kept
set K⊆M (removed = M∖K); we score the corrected caption by:
- **CHAIR_i** = P(h=1 | o∈K) = fraction of *kept* mentions that are hallucinated (per-instance hallucination rate);
- **recall** = fraction of *ground-truth* objects still mentioned. Since mentioned-grounded ⊆ GT, removing a grounded
  mention lowers recall; keeping all grounded mentions preserves the baseline recall.

Treat "remove o" as the classifier predicting positive (hallucinated). At threshold t (remove iff s(o)≥t):
- **TPR(t)** = P(s≥t | h=1) — fraction of hallucinations removed;
- **FPR(t)** = P(s≥t | h=0) — fraction of grounded mentions wrongly removed.

## 2. Theorem 1 — the frontier is the ROC curve (risk–coverage), with achievability
Fix a detector score s. For threshold t (remove iff s≥t),
  CHAIR_i(t) = P(h=1 | s<t) = [ π·(1−TPR(t)) ] / [ π·(1−TPR(t)) + (1−π)·(1−FPR(t)) ],
  recall(t)  = recall_base · P(s<t | h=0) = recall_base · (1 − FPR(t)),
both functions of (TPR(t), FPR(t)) — a point on the ROC of s.

**(≤, optimality via Neyman–Pearson).** Among *all* post-hoc removal rules δ (keep/remove per mention, measurable
w.r.t. the information in s), fix the grounded-removal rate FPR = 1 − recall/recall_base. Minimizing CHAIR_i at fixed
FPR ⟺ maximizing the removed-hallucination rate TPR at fixed FPR. By the Neyman–Pearson lemma the maximizer is the
likelihood-ratio threshold test on s — i.e., a threshold rule. Hence **no rule using s beats the ROC(s) frontier.**

**(=, achievability by construction).** The excision rule "remove o iff s(o) ≥ t" is itself a valid post-hoc
per-object rule and realizes exactly (recall(t), CHAIR_i(t)); randomizing between two thresholds realizes every point
on the concave hull. Therefore **every** point of ROC(s)'s hull is attained ⇒ frontier(s) = ROC(s) with equality, not
merely ≤. *(This achievability half is exactly what separates us from lower-bound-only results like Kalai et al.
2509.04664, which prove err_gen ≥ 2·err_iiv for single-pass generation but do not characterize the attainable
removal frontier.)*

**Corollary (strict ROC-dominance ⇒ strictly better frontier).** If ROC_A(f) ≥ ROC_B(f) for all FPR f with strict
inequality somewhere and *no crossing*, then A attains strictly lower CHAIR_i at every recall. **Bayes floor:** over
all detectors using available features φ, the frontier is minimized by the ROC of the Bayes score P(h=1|φ).

*(This is the selective-classification risk–coverage principle (Chow 1970; El-Yaniv & Wiener 2010; Franc et al. 2021)
specialized to object mentions; the CHAIR_i/recall closed form, the Neyman–Pearson optimality among elimination rules,
and the achievability=ROC statement in the removal regime are the contribution.)*

**Empirical confirmation (LLaVA-1.5-7B, 500 CHAIR imgs, n=1428 mentions, π=0.195):** ROC-AUC grounding 0.767 vs
self-verification 0.893, and the two ROCs **do not cross** (self-verify TPR ≥ grounding TPR at all 25 FPR grid points
in [0.02,0.5]: e.g. FPR 0.1 → 0.65 vs 0.40, FPR 0.3 → 0.93 vs 0.70). Hence self-verification dominates on the
CHAIR_i–recall frontier at *every* recall: recall 72 → C_i 11.09 vs 13.11; recall 66 → 9.50 vs 12.30; recall 62 →
8.68 vs 11.99. Exactly the Corollary (strict, no-crossing).

## 3. Theorem 2 — mechanism invariance
Let an elimination mechanism remove/suppress mention o with "strength" a nondecreasing function of s(o) (hard excision:
1[s≥t]; soft-suppression: remove iff β·[s−τ]_+ flips the token argmax, a monotone threshold in s; grammar-repair on an
excised set: identity on K). Any such **monotone per-object** rule induces a threshold on s (possibly randomized), so its
(recall, CHAIR_i) point lies on the ROC-derived frontier of Theorem 1. **No monotone per-object intervention mechanism
can beat the frontier; only the score s can.** *(Empirically excision ≡ soft-suppression trace the same curve.)*

## 4. When the bound is violated (and why not beaten): decode-time coupling
Decode-time interventions (attention edits, logit CFG, decode-time suppression) change the autoregressive trajectory, so
the kept set is **not** a subset of the original M(x): new mentions (grounded or hallucinated) appear and per-object
labels become coupled through generation. Theorem 2's hypothesis (monotone per-object rule on a fixed M) fails, so no
frontier guarantee holds — empirically these methods move *off* the curve, usually **worse** (protecting person at
decode time → longer captions → more downstream hallucination). This explains why the base-rate/detector gains only
realize *post-hoc*: post-hoc excision keeps M fixed and per-object independent; decode-time does not.

## 5. The Bayes floor and the generation–verification gap
Over all detectors using features φ(o,x), the best frontier is the ROC of the Bayes classifier P(h=1 | φ). The relevant
question is which φ is available training-free. We find the model's **own verification** — querying "Is there an X?" and
reading the yes/no margin — has ROC 0.89, strictly above every feature detector we built (grounding 0.82, attention,
CFG-image-effect +0.003). I.e., **the model is a better detector of its own hallucinations than any external feature**:
it "knows" an object is unsupported when asked, yet emits it during free generation — a **generation–verification gap**.
By Theorem 1's corollary this self-verification score realizes the lowest achievable frontier among training-free
options. Whether it is Bayes-optimal (headroom above 0.89) is open.

## 6. Consequences
1. **Unification / bound (honest scope).** *Post-hoc* per-object methods (excision / hard-ban / rejection over a fixed
   mention set) are provably one point under one ROC (§2–3); cross-method "wins" at unequal recall are frontier-sliding.
   **Empirically (self-verification post-hoc frontier, 500 imgs):** baseline (76.9,15.4), soft (74.5,13.2),
   PAI-α0.4 (76.1,13.5), PAI-α0.5 (69.6,10.2), ours soft+PAI (72.8,11.9) all land ON or INSIDE the frontier — i.e., even
   *decode-time* methods (PAI/soft) sit at or above the best-detector post-hoc bound. **One exception:** a strong
   decode-time combo (soft+PAI, τ0.45) dips 0.56 *below* it (71.3, C_i 10.23 vs frontier 10.79) — consistent with §4:
   decode-time interventions escape the static assumption and can access an *online* signal (attention re-weighting)
   equivalent to a different, possibly richer score s′. So the clean bound is for post-hoc methods; decode-time methods
   are **approximately** bounded and, when they exceed it, do so by effectively improving the score — which is exactly
   the thesis's escape condition, not a counterexample to it.
2. **Where progress must come from:** not new decode tricks (mechanism-invariant) but better *scores* — and the best
   training-free score is the model's own verification (close the generation–verification gap).
3. **Practical corollary:** self-verification-scored post-hoc excision + grammar-repair gives the best training-free
   recall-safe *and fluent* elimination (CHAIR_i −4 to −6 at held recall; GPT-judge fluency 4.4/5), beating
   suppression — because it uses the best score, not a better mechanism.

## 7. Assumptions / honesty
- Theorems assume **removal-only** interventions on a **fixed mention set** with **per-object** decisions. Decode-time
  and free-regeneration violate this (Section 4) — a feature, not a bug: it delimits exactly when the bound holds.
- CHAIR_i (per-instance) is the ROC-governed quantity; **CHAIR_s (caption-level) is not** (all-or-nothing per caption),
  which is why a higher-ROC detector need not lower CHAIR_s — reconciled in §2.
- The Bayes floor is stated relative to *available* features; we do not claim self-verification is globally optimal.
- Empirical scope so far: LLaVA-1.5-7B / CHAIR. Cross-model (13B, InstructBLIP) + AMBER validation pending (required).
