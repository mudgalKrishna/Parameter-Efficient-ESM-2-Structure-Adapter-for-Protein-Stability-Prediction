# ESM-2 + ProteinMPNN fusion for protein stability (ΔΔG)

Predicting mutation-induced stability change on the Tsuboyama et al. mega-scale dataset, using
ThermoMPNN's own protein splits. Metric: median per-protein Spearman ρ.

![results](results/results.png)

## Headline

**The same architecture-level comparison, run fairly, wins on every split tested.**

| Evaluation | This work | ThermoMPNN | Δ (95% CI) | Proteins won | p |
|---|---|---|---|---|---|
| **5 CV folds, identical protocol** | **0.848** avg | 0.787 avg (retrained) | **+0.061** | 259 / 298 | 4.6e-4 … 6.5e-11 |
| Official test split | **0.835** | 0.769 (retrained) | +0.066 [0.036, 0.090] | 21 / 28 | 2.7e-4 |
| Official test split | **0.835** | 0.772 (released weights) | +0.063 [0.029, 0.085] | 21 / 28 | 5.8e-4 |

Won **5 of 5** cross-validation folds, each with a bootstrap CI excluding zero.

## The methodological point (panel C)

A naive comparison against ThermoMPNN's **released weights** is invalid on the CV folds, and the
data shows exactly why. Those weights were trained on the mega-split train proteins; the CV folds
repartition all 298 proteins, so most fold-test proteins are inside their training set:

| ThermoMPNN, released weights | Median ρ |
|---|---|
| Official test split (genuinely unseen) | 0.772 |
| CV folds (largely seen in training) | 0.847 – 0.881 |

That **+0.076 average inflation** is memorisation, not skill. Scored against released weights our
model "loses" all 5 folds; scored against the *same architecture retrained on each fold's own
training proteins* it wins all 5. The fair baseline is the retrained one.

**The retrained baseline is faithful**: trained on the official split it reaches ρ = 0.773 versus
the released weights' 0.772 — it reproduces their published-level result to within 0.001
(`code/thermo_train.py` re-implements their `TransferModel` with their `config.yaml`).

## Method

**Model** (`code/stab2.py`)
- ESM-2 (650M / 3B / 15B) with LoRA on q/k/v/dense; softmax-weighted mix of all hidden layers;
  a learned α·(masked-LM log-likelihood ratio) term.
- **Official ProteinMPNN** (`v_48_020`, full strict `state_dict` load, all residues visible, WT
  sequence) — decoder hidden states + sequence embeddings, exactly as ThermoMPNN consumes them.
  This replaced a re-implemented encoder in the earlier pipeline that only partially matched the
  public checkpoint, so most of its "pretrained" weights were silently random.
- 2-layer transformer fusing both streams; per-residue 20-way head; ΔΔG = head[mut] − head[wt].
- Loss: MSE + (1 − per-protein Pearson) + **pairwise ranking loss** (the metric is rank-based).

**Protocol**
- Selection uses validation data only; the *ensembling strategy* was chosen on the CV folds, not
  on the 31-protein validation set, then applied and locked before scoring test.
- `PREREGISTRATION.md` was written before any round-2 model trained, and fixes the rules below.

## Ablations (validation, 3 seeds each — `results/round2_runs.csv`)

| Variant | Mean val ρ | |
|---|---|---|
| ProteinMPNN only | 0.797 | structure alone |
| ESM-2 650M only | 0.848 | sequence alone |
| Fusion 650M | 0.857 | **both streams beat either** |
| Fusion 3B | 0.868 | |
| Fusion 3B + ranking loss | **0.871** | best single config |
| Fusion 15B | 0.864 | no gain over 3B |
| + cross-attention fusion | 0.858 | ✗ rejected |
| + auxiliary burial task | 0.863 | ✗ rejected |

**Negative results kept, as pre-registered:**
- **Retraining on train+val does not help** (mean Δ −0.0007 over 5 folds, clean A/B on untouched
  fold tests), so the final model does not use it — despite +13% training data.
- **Ensembling strategy barely matters**: single-best, top-k and greedy all land within 0.0025 on
  the folds, far inside the 0.022 fold-to-fold spread.

## Disclosure

The official test split was scored **twice** in total, once per round. Round 1 used a
validation-selected ensemble (test ρ = 0.842); round 2 locked its ensemble using the CV-chosen
strategy (test ρ = 0.835). The 0.007 difference sits well inside the ±0.03 CI width at n = 28.
Both are reported; neither was selected by looking at test. Round 2 is the result to cite because
its margin is backed by 5 independent folds against a fairly-trained baseline — see
`results/test_result.json` (round 1) and `results/final_results.json` (round 2).

## Reproduce

```bash
git clone https://github.com/dauparas/ProteinMPNN      ext/ProteinMPNN
git clone https://github.com/Kuhlman-Lab/ThermoMPNN    ext/ThermoMPNN

python code/stab2.py --tag R1_rank --seed 0 --esm facebook/esm2_t36_3B_UR50D \
       --lora_rank 32 --lora_lr 2e-4 --head_lr 5e-4 --rank_w 0.5     # one model
python code/launch2.py 2            # whole sweep from results/queue2.txt
bash   code/run_thermo.sh           # retrain the ThermoMPNN baseline, 5 seeds x 6 splits
python code/gen_thermo_all.py       # released-weight predictions, all 271k variants
python code/ens3.py                 # choose ensembling strategy on the CV folds
python code/ens3.py --lock          # lock the ensemble before touching test
python code/final.py                # fair comparison + the single locked test evaluation
```

Paths assume the sandbox layout (`/marimo/training_bundle_copy.zip`, `/marimo/work/...`).
Runtimes on one RTX PRO 6000 (96 GB): 650M ≈ 2 min, 3B ≈ 3 min, 15B ≈ 6 min, ThermoMPNN
baseline ≈ 10 s after feature caching. The full study is 179 training runs: 58 (round 1) +
91 (round 2) + 30 (retrained ThermoMPNN baseline).
