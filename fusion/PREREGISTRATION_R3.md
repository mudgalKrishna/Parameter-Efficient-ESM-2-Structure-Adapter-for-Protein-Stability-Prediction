# Pre-registration — round 3 (written before any round-3 result was read)

Date: 2026-10-24. Rounds 1 and 2 have each evaluated the official Tsuboyama/ThermoMPNN test split
once (0.8416 and 0.8349 respectively, against retrained ThermoMPNN 0.7685). This file fixes the
round-3 protocol in advance.

## Why round 3 exists

Round 2's own training logs identify the binding constraint. Across all 91 runs, validation peaked
at a **mean epoch of 8.4 (max 22) against a 40-epoch cap** — every run early-stopped and then
overfit. Combined with two other round-2 results (15B scored *below* 3B; +13% more variants from
train+val retraining changed nothing, Δ −0.0007), the diagnosis is that the model is
**protein-limited, not capacity- or compute-limited**: there are 215,731 training variants but only
239 training *proteins*, and variants within a protein are strongly correlated, so the effective
sample size is the protein count.

Two candidate improvements from round 2's list are therefore **abandoned before testing**, and this
is recorded here rather than quietly dropped:

- *"Train longer / more epochs"* — refuted by the best-epoch distribution above. Not attempted.
- *"Reverse-mutation augmentation for antisymmetry"* — a no-op for this architecture. ΔΔG is read
  off a single wild-type forward pass as `head[mut] − head[wt]`, so ΔΔG(wt→mut) = −ΔΔG(mut→wt)
  holds identically by construction. Not attempted.

## What round 3 tests

1. **More proteins (primary).** FireProtDB adds training proteins from a different assay.
   Only FireProt's `homologue-free` split is used (89 proteins; the other 11 are homologous to
   megascale proteins per ThermoMPNN's own `megascale_vs_fireprot_homologues.m8`), so no FireProt
   protein is homologous to *any* megascale protein. This makes the added data safe for every CV
   fold and for the official test split simultaneously, with no per-fold exclusion. 88 of 89 were
   successfully converted (2,577 of 2,578 variants). **FireProt proteins enter the fit set only —
   never val or test — so every reported score remains measured on Tsuboyama proteins alone and
   stays directly comparable to rounds 1 and 2.**
2. **Regularisation (secondary).** LoRA dropout and head dropout, since the epoch-8 overfit says
   regularisation is the lever that more steps is not.
3. **Soft-rank Spearman surrogate (secondary).** Optimise a differentiable approximation of the
   evaluation metric instead of the pairwise hinge.

Two label corrections are applied to FireProt and were verified against the data *before* training:
its ddG uses the opposite sign convention (positive = destabilizing; median +0.68 and
corr(ddG, ASA) = −0.34, versus Tsuboyama's negative = destabilizing), and its spread is wider
(std 1.88 vs 1.00). Labels are negated and rescaled to the Tsuboyama training std.

## Rules

1. All round-3 decisions use **fold validation only**. The arm to report is chosen by **mean
   validation median Spearman across folds**, never by any test number.
2. The fold *test* sets are scored once per arm, after the arm is chosen on fold validation.
3. **The official test split is not re-evaluated in round 3.** It has been scored twice; a third
   look would weaken the locked guarantee that is rounds 1–2's main credibility claim. Round 3 is
   reported on the CV folds only, and the round-2 official-test number stands as the headline.
4. The round-2 baseline is **re-run from scratch in this round** with identical code and seeds
   (arm `S0_base` / `F0_base`), so the A/B is paired and not a comparison against numbers produced
   by different code on a different machine.
5. A cheap 650M screen runs first. If FireProt does not help at 650M, the 3B stage is skipped and
   the negative result is reported.

## Success criterion

FireProt is reported as an improvement only if it raises **mean fold validation** across folds and
seeds, and the gain survives on fold test. If validation improves but fold test does not, both are
published and the headline does not change.

If no arm beats the round-2 baseline, round 3 is published as a **negative result** — specifically,
as evidence that the protein-limited diagnosis is correct but that cross-assay data does not lift
the ceiling — and the round-2 numbers remain the repository's headline. Nothing in round 3 may
revise the round-2 claims downward or upward on the official test split.
