# Protein Stability Prediction: ESM-2 + ProteinMPNN Fusion

Predicting how a single amino-acid mutation changes a protein's stability (ΔΔG), evaluated on the
Tsuboyama et al. mega-scale benchmark using ThermoMPNN's own protein splits.

**Metric: median per-protein Spearman ρ.** Higher is better.

![results](fusion/results/results.png)

## Headline result

| Evaluation | This work | ThermoMPNN | Δ (95% CI) | Proteins won | p |
|---|---|---|---|---|---|
| **5 cross-validation folds, identical protocol** | **0.848** avg | 0.787 avg (retrained) | **+0.061** | **259 / 298** | 4.6e-4 … 6.5e-11 |
| Official test split | **0.835** | 0.769 (retrained) | +0.066 [0.036, 0.090] | 21 / 28 | 2.7e-4 |
| Official test split | **0.835** | 0.772 (released weights) | +0.063 [0.029, 0.085] | 21 / 28 | 5.8e-4 |

Won **5 of 5** cross-validation folds. Every fold's bootstrap confidence interval excludes zero.

Per fold:

| Fold | This work | ThermoMPNN retrained | Δ | 95% CI | Proteins won | p |
|---|---|---|---|---|---|---|
| CV0 | 0.849 | 0.781 | +0.068 | [0.040, 0.084] | 44 / 51 | 1.0e-6 |
| CV1 | 0.811 | 0.773 | +0.038 | [0.022, 0.055] | 46 / 59 | 4.6e-4 |
| CV2 | 0.849 | 0.778 | +0.071 | [0.051, 0.092] | 55 / 63 | 2.2e-9 |
| CV3 | 0.860 | 0.802 | +0.059 | [0.045, 0.088] | 56 / 63 | 3.8e-9 |
| CV4 | 0.869 | 0.801 | +0.068 | [0.051, 0.083] | 58 / 62 | 6.5e-11 |

## The methodological finding

**A naive comparison against ThermoMPNN's released weights is invalid on its own CV folds**, and the
data shows exactly why. Those weights were trained on the mega-split training proteins. The CV folds
repartition all 298 proteins, so most fold-*test* proteins sit inside the released model's *training*
set:

| ThermoMPNN, released weights | Median ρ |
|---|---|
| Official test split (genuinely unseen) | 0.772 |
| CV folds (largely seen during its training) | 0.847 – 0.881 |

That **+0.076 average inflation is memorisation, not skill.** Scored against the released weights,
this model appears to *lose* all five folds. Scored against the *same architecture retrained on each
fold's own training proteins*, it wins all five. The retrained model is the only fair baseline.

**The retrained baseline is faithful, not a weakened straw man.** Trained on the official split it
reaches ρ = 0.773, versus the released weights' 0.772 — reproducing the published-level result to
within 0.001. `fusion/code/thermo_train.py` re-implements ThermoMPNN's `TransferModel` using their
own `config.yaml` (hidden dims [64, 32], 2 final layers, light attention, frozen ProteinMPNN).

## 1. Why predict protein stability?

Proteins are chains of amino acids that fold into particular three-dimensional shapes. Their shape
and stability affect how they work. A small change in the sequence can make a protein more stable,
less stable, or have little effect.

These effects can be measured experimentally, but testing every possible mutation is slow and
expensive. A prediction model helps prioritise which mutations are worth investigating — it does not
replace laboratory testing.

## 2. What is predicted

**ΔΔG**, the change in folding free energy caused by a mutation:

`ΔΔG = ΔG_mutant − ΔG_wild-type`

Under this benchmark's convention a **negative value means a destabilizing mutation**.

## 3. How the model works

Two pretrained models are fused, each contributing a different kind of information.

**Sequence stream — ESM-2 with LoRA**
- ESM-2 (650M / 3B / 15B) with LoRA adapters on the q/k/v/dense projections, so the backbone stays
  frozen and only a small set of parameters trains.
- A softmax-weighted mixture of *all* hidden layers rather than just the last — different depths
  carry different signal for stability.
- A learned α·(masked-LM log-likelihood ratio) term, giving the model direct access to ESM-2's own
  zero-shot opinion about the substitution.

**Structure stream — official ProteinMPNN**
- The official `v_48_020` checkpoint, loaded with a **full strict `state_dict` load**, all residues
  visible, wild-type sequence supplied. Decoder hidden states and sequence embeddings are consumed
  exactly the way ThermoMPNN consumes them.
- This replaced a *re-implemented* encoder used in the earlier phase of this project, which only
  partially matched the public checkpoint (465 vs 416 edge-embedding inputs, different dense-layer
  naming, different message scaling) — so most of its supposedly "pretrained" weights were silently
  random. Fixing this was worth more than any hyperparameter change.

**Fusion and head**
- A 2-layer transformer fuses both streams; a per-residue 20-way head predicts over amino acids, and
  ΔΔG is read off as `head[mutant] − head[wild-type]`.
- Loss = MSE + (1 − per-protein Pearson) + a **pairwise ranking loss**, because the evaluation metric
  is rank-based and a pure regression loss optimises the wrong thing.

## 4. Data and evaluation setup

The **Tsuboyama ΔΔG mega-scale dataset**, using ThermoMPNN's published splits so the comparison is
like-for-like.

| Split | Proteins | Variants | Purpose |
| --- | ---: | ---: | --- |
| Train | 239 | 215,731 | Train model parameters |
| Validation | 31 | 27,328 | Compare configurations, select checkpoints |
| Test | 28 | 28,172 | Held back for the locked final evaluation |

Plus ThermoMPNN's **5 cross-validation folds**, which repartition all 298 proteins — 271,231 variants
in total. Splits are at the **protein level**: no protein appears in two splits, because variants of
the same protein are correlated and mixing them would inflate the result.

**Protocol**
- Model selection uses validation data only.
- The *ensembling strategy* was chosen on the CV folds, then locked before the test split was touched
  (`fusion/results/selection_locked_before_test_v2.json`).
- `fusion/PREREGISTRATION.md` was written **before any model in this phase was trained** and fixes
  the rules, including the commitment to report negative results.

## 5. Reading the metric

**Median per-protein Spearman ρ.** For each test protein, the predicted ranking of its variants is
compared with the measured ranking; the median is taken across proteins.

- Closer to **1** — the model orders mutations much as the experiment does.
- Around **0** — no rank agreement.
- Negative — the ordering is reversed.

The median *per-protein* score is the headline because variants from one protein are not independent
observations, so a pooled correlation would be dominated by whichever proteins happen to have the
most variants.

## 6. Ablations

Validation ρ, 3 seeds each — full records in `fusion/results/round2_runs.csv`.

| Variant | Mean val ρ | |
|---|---|---|
| ProteinMPNN only | 0.797 | structure alone |
| ESM-2 650M only | 0.848 | sequence alone |
| Fusion, 650M | 0.857 | **both streams beat either one** |
| Fusion, 3B | 0.868 | |
| Fusion, 3B + ranking loss | **0.871** | best single configuration |
| Fusion, 15B | 0.864 | no gain over 3B |
| + cross-attention fusion | 0.858 | rejected |
| + auxiliary burial-prediction task | 0.863 | rejected |

The fusion result is the point: **0.857 beats both 0.797 (structure alone) and 0.848 (sequence
alone)**, so the two streams carry complementary information. Scaling to 3B helps; 15B does not.

## 7. Negative results

Reported because the pre-registration committed to reporting them.

- **Retraining on train+validation does not help.** Mean Δ = −0.0007 across the 5 folds in a clean
  A/B scored on untouched fold-test proteins — despite roughly 13% more training data. The final
  model therefore does *not* use it.
- **The ensembling strategy barely matters.** Single-best, top-2/3/4, all-average and greedy
  validation selection all land within 0.0025 of each other on the folds — far inside the 0.022
  fold-to-fold spread. Reporting "we used a clever ensemble" would overstate its contribution.
- **Cross-attention fusion and an auxiliary burial task both hurt** (see ablations) and were dropped.

## 8. Round 3: a pre-registered attempt to improve on this, which failed

Round 2's training logs point at a specific constraint. Across all 91 runs, validation peaked at a
**mean epoch of 8.4 (max 22) against a 40-epoch cap** — every run early-stopped and then overfit.
Together with two other round-2 results (15B scored *below* 3B; train+val retraining gave
Δ −0.0007 despite +13% more variants), the diagnosis is that the model is **protein-limited, not
capacity- or compute-limited**: 215,731 training variants but only 239 training *proteins*, and
variants within a protein are strongly correlated, so the effective sample size is the protein count.

Two ideas were therefore **dropped before testing**, and the reasons are recorded rather than the
ideas quietly forgotten:

- *Train longer* — directly refuted by the best-epoch distribution above.
- *Reverse-mutation augmentation to enforce antisymmetry* — a no-op for this architecture. ΔΔG is
  read off one wild-type forward pass as `head[mut] − head[wt]`, so ΔΔG(wt→mut) = −ΔΔG(mut→wt)
  holds identically by construction.

Three ideas were tested, under `PREREGISTRATION_R3.md` (written before any round-3 result was read).
**All three failed.** Paired on (fold, seed), 650M, 3 folds × 2 seeds per arm:

| Arm | Δ val | val wins | Δ fold test | test wins | p (test) |
|---|---|---|---|---|---|
| **+ FireProtDB, 88 proteins** (full weight) | −0.0082 | 1/6 | −0.0111 | **0/6** | 0.031 |
| + FireProtDB, down-weighted to 0.3 | −0.0061 | 1/6 | −0.0138 | **0/6** | 0.031 |
| + LoRA & head dropout | −0.0020 | 3/6 | −0.0068 | 1/6 | 0.094 |
| + soft-rank Spearman surrogate | +0.0003 | 4/6 | −0.0033 | 1/6 | 0.22 |

The pre-registered success criterion was a rise in **mean fold validation**. Only the soft-rank arm
rose at all, by +0.0003 (p = 0.56) — noise — and it lost on fold test. Per the pre-registration the
3B stage was skipped and this is published as a negative result. **The round-2 numbers stand
unchanged, and the official test split was not scored a third time.**

### Why the FireProt result is informative rather than a bug

This was the arm with real expected upside: it raises training proteins from 239 to 327 (+37%) on
precisely the axis the diagnosis identifies. It made the model *worse*, in every fold and every seed.
Three checks say that is a property of the data, not of the plumbing:

- **Labels are correctly converted.** FireProt uses the opposite sign convention (positive =
  destabilizing; median +0.68, and corr(ddG, ASA) = −0.34 confirms buried positions score high)
  and a wider spread (std 1.88 vs 1.00). After negating and rescaling, the converted labels match
  Tsuboyama to three decimals: median −0.473 vs −0.497, std 1.004 vs 1.002, fraction destabilizing
  0.77 vs 0.77. Feeding it raw would have been the obvious bug; it was checked for and fixed first.
- **No leakage.** Only FireProt's `homologue-free` split is used, so no added protein is homologous
  to *any* megascale protein, per ThermoMPNN's own mmseqs map. A plain k-mer screen catches 7 of the
  11 contaminated proteins; their map catches all 11, including remote homologues at 29–59% identity
  that share no 8-mers. 88 of 89 proteins converted (2,577 of 2,578 variants).
- **It is not a loss-weighting artifact.** FireProt proteins carry a median of ~9 variants against
  Tsuboyama's ~900, so each contributes an equally-weighted but much noisier per-protein correlation
  term. Down-weighting them to 0.3 did not help — it was slightly worse (0/6 on test either way).

The conclusion is that **cross-assay stability data does not transfer here**: FireProt aggregates
many labs, methods and conditions, and the label noise it brings outweighs the structural diversity
its extra proteins add. The protein-limited diagnosis appears to be right about the *constraint* and
wrong about the *remedy* — more proteins help only if they come from a comparably uniform assay.

Practically, 0.848 looks close to the ceiling for this approach: Tsuboyama's own experimental
replicate reproducibility bounds per-protein ρ near 0.88–0.90, and none of the obvious levers moved
the result. Reported in full because a pre-registration that only publishes successes is worthless.

## 9. Disclosure

The official test split was scored **twice in total**, once per project phase. The earlier phase used
a validation-selected ensemble (test ρ = 0.842); this phase locked its ensemble using the CV-chosen
strategy (test ρ = 0.835). The 0.007 difference sits well inside the ±0.03 CI width at n = 28. Both
numbers are reported; neither was selected by looking at test.

The phase reported above is the one to cite, because its margin is backed by 5 independent folds
against a fairly-trained baseline rather than a single split.

## 10. Earlier phase of this project

The project's first phase used a re-implemented structure encoder and reached test ρ = **0.7547**,
against ThermoMPNN's 0.7720 — a paired difference of −0.0173 (95% CI −0.068 to +0.038, p = 0.465),
i.e. no statistically clear difference in either direction. Those experiments established the
groundwork and are kept in the repository:

| Earlier configuration | Test median ρ | 95% CI |
| --- | ---: | ---: |
| ESM-2 650M + LoRA + ProteinMPNN + structure adapter | 0.7547 | 0.7118–0.8137 |
| ESM-2 650M, full fine-tuning, sequence only | 0.7503 | 0.6900–0.8071 |
| ESM-2 150M + LoRA + ProteinMPNN + structure adapter | 0.7271 | 0.6610–0.7899 |
| Ridge regression, 29 hand-crafted features | 0.6289 | 0.5860–0.6569 |

Three changes account for most of the gap between 0.7547 and 0.835:

1. **The structure encoder was actually loading its pretrained weights.** The earlier encoder only
   partially matched the official checkpoint, so much of it was random.
2. **A ranking loss**, matching the rank-based evaluation metric (+0.004 on validation).
3. **A larger backbone** — 3B rather than 650M (+0.011 on validation).

In the earlier phase the structure branch added only +0.0044 over a sequence-only model, with a CI
including zero. With the checkpoint correctly loaded, the structure stream contributes a clear
+0.009 on validation (0.848 → 0.857) and the fused model beats either stream alone.

## 11. Repository layout

```
fusion/
  README.md                   detailed write-up of the headline phase
  PREREGISTRATION.md          written before any model in the headline phase was trained
  PREREGISTRATION_R3.md       written before any round-3 result was read
  code/
    stab2.py                  the fusion model (ESM-2 LoRA + ProteinMPNN + transformer)
    thermo_train.py           retrains ThermoMPNN's TransferModel = the fair baseline
    ens2.py ens3.py final.py  ensembling, strategy selection, final locked evaluation
    gen_thermo_all.py         released-weight predictions, all 271k variants
    launch2.py run_thermo.sh  job queues
    figure.py                 the figure at the top of this file
    stab3.py                  round 3: FireProt, regularisation, soft-rank loss
    fireprot_build.py         converts FireProtDB to the training tensor format
    ab.py runq.py             round-3 paired A/B analysis and job runner
  results/
    final_results.json        headline numbers
    final_cv_fair.csv         per-fold fair comparison
    round2_runs.csv           91 training runs, headline phase
    thermompnn_retrained_runs.csv   30 baseline runs
    runs2/ runs_thermo/       per-run JSON records
    round3_runs.csv           30 round-3 runs (the negative result)
    round3_ab.json            round-3 paired comparisons
    results.png               the figure

train_esm2_spurs_fixed.py     earlier-phase pipeline (marimo notebook)
run_structure_track.py        earlier-phase runner
analyze_650_local.py          earlier-phase analysis
```

## 12. Reproduce

```bash
git clone https://github.com/dauparas/ProteinMPNN    ext/ProteinMPNN
git clone https://github.com/Kuhlman-Lab/ThermoMPNN  ext/ThermoMPNN

python fusion/code/stab2.py --tag R1_rank --seed 0 \
       --esm facebook/esm2_t36_3B_UR50D \
       --lora_rank 32 --lora_lr 2e-4 --head_lr 5e-4 --rank_w 0.5   # a single model
python fusion/code/launch2.py 2        # the full sweep, from results/queue2.txt
bash   fusion/code/run_thermo.sh       # retrain the baseline, 5 seeds x 6 splits
python fusion/code/gen_thermo_all.py   # released-weight predictions, all 271k variants
python fusion/code/ens3.py             # choose the ensembling strategy on the CV folds
python fusion/code/ens3.py --lock      # lock the ensemble before touching test
python fusion/code/final.py            # fair comparison + the single locked test evaluation
```

Round 3 (the negative result) adds:

```bash
python fusion/code/fireprot_build.py                   # FireProtDB -> training tensors
python fusion/code/runq.py queue_screen.txt 2           # the paired A/B arms
python fusion/code/ab.py S0_base S1_fp S2_fp03 S3_reg S4_sr
```

Runtimes on one RTX PRO 6000 (96 GB): 650M ≈ 2 min, 3B ≈ 3 min, 15B ≈ 6 min, and the ThermoMPNN
baseline ≈ 10 s once its frozen features are cached. The full study is **209 training runs**: 58
(earlier phase) + 91 (headline phase) + 30 (retrained ThermoMPNN baseline) + 30 (round 3).

## 13. Limitations

- The official test split has only **28 proteins**; the 5-fold result across 298 proteins is the more
  reliable evidence, and the two agree (+0.066 and +0.061).
- A strong rank correlation does not mean individual ΔΔG values are numerically accurate.
- All results are on one dataset. Tsuboyama et al. is dominated by small, well-behaved domains, so
  performance on large multi-domain or membrane proteins may differ.
- Both models here consume experimental or predicted backbone structures; errors in those inputs
  propagate.
- Predictions are for research prioritisation and are not a substitute for experimental measurement.
- This approach looks close to its ceiling. Round 3 tested four further ideas under
  pre-registration and none improved on 0.848 (section 8), and Tsuboyama's own replicate
  reproducibility bounds per-protein ρ near 0.88–0.90, so the remaining gap to 1.0 is largely
  experimental noise rather than model error.

## Summary

Fusing a LoRA-adapted ESM-2 with the official ProteinMPNN encoder reaches **0.835** median
per-protein Spearman on the held-out Tsuboyama test split and **0.848** averaged across ThermoMPNN's
5 cross-validation folds, beating a faithfully retrained ThermoMPNN by **+0.066** and **+0.061**
respectively — winning **5 of 5 folds** and **259 of 298 proteins**, with every confidence interval
excluding zero.

Establishing that margin required noticing that ThermoMPNN's *released* weights cannot legitimately
be scored on its own CV folds, because those folds redistribute the proteins it trained on — an
inflation of +0.076 that reverses the apparent conclusion.
