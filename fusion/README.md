# fusion/ — code and results

**All results, method and discussion live in the [repository README](../README.md).** This file is
just an index of the folder, so there is only one place where numbers are written down.

## Pre-registrations

| File | Written before |
|---|---|
| `PREREGISTRATION.md` | any model of the headline phase was trained |
| `PREREGISTRATION_R3.md` | any round-3 result was read |

## code/

| File | What it does |
|---|---|
| `paths.py` | every filesystem path, resolved from `ESM2_*` environment variables |
| `setup_env.py` | fetches ProteinMPNN, ThermoMPNN, the splits, FireProtDB and S669 |
| `stab2.py` | the fusion model — ESM-2 + LoRA, official ProteinMPNN, transformer fusion |
| `stab3.py` | round 3: FireProt, regularisation, soft-rank loss, and the S669 split |
| `thermo_train.py` | retrains ThermoMPNN's `TransferModel` per split — **the fair baseline** |
| `ens2.py` | per-protein z-scored prediction loading and mixing |
| `ens3.py` | chooses the ensembling strategy on the CV folds (`--lock` fixes it before test) |
| `final.py` | the fair per-fold comparison and the single locked official-test evaluation |
| `gen_thermo_all.py` | released-weight ThermoMPNN predictions for all 271k variants |
| `fireprot_build.py` | converts FireProtDB to the training tensor format (sign + scale corrected) |
| `s669_build.py` | converts the S669 benchmark to the tensor format |
| `s669_thermo.py` | retrains ThermoMPNN on `train_s669` — the transfer baseline |
| `s669_eval.py` | pooled S669 scoring |
| `s669_compare.py` | matched S669 head-to-head, restricted to variants both models place |
| `launch2.py` `runq.py` `run_thermo.sh` | job queues |
| `ab.py` | round-3 paired A/B analysis |
| `figure.py` | the 3-panel results figure |

## results/

| File | Contents |
|---|---|
| `final_results.json` | the headline numbers |
| `final_cv_fair.csv` | per-fold fair comparison vs the retrained baseline |
| `final_official_per_protein.csv` | per-protein scores on the official test split |
| `s669_summary.json` | the transfer result — full set, matched comparison, and by length |
| `s669_by_length.csv` | transfer performance vs chain length |
| `s669_comparison.json` `s669_results.json` | matched head-to-head; per-model pooled scores |
| `round3_runs.csv` `round3_ab.json` | round 3 — the negative result |
| `round2_runs.csv` | 91 runs of the headline phase |
| `thermompnn_retrained_runs.csv` | 30 retrained-baseline runs |
| `selection_locked_before_test_v2.json` | the ensemble, locked before test was touched |
| `strategy_choice.json` `cv_strategy_comparison.csv` | ensembling strategy selection |
| `cv_vs_thermompnn.csv` | folds vs *released* weights (shows the memorisation gap) |
| `runs2/` `runs_thermo/` `runs_thermo_s669/` | per-run JSON records |
| `results.png` | the figure |

Paths come from `paths.py`, which reads `ESM2_WORK`, `ESM2_BUNDLE`, `ESM2_EXT` and `ESM2_DATA`
and defaults to `./work`. Run `python setup_env.py --check` to verify the layout. See the
repository README for setup and reproduce steps.
