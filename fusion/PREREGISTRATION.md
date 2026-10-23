# Pre-registration — round 2 (written before any round-2 model was trained)

Date: 2026-10-23. Round 1 already evaluated the official Tsuboyama/ThermoMPNN test split once
(locked ensemble 0.8416 vs released ThermoMPNN 0.7720). This file fixes the round-2 protocol in
advance so the final number is measured, not selected.

## Rules
1. **All** round-2 architecture, loss, hyperparameter and ensemble decisions use the validation
   split (31 proteins) only, or the cross-validation folds — never the official test split.
2. Generalization is verified on ThermoMPNN's own 5 CV folds (`cv_{train,val,test}_{0..4}` in
   `dataset_splits/mega_splits.pkl`). Each fold: select on that fold's val, score that fold's test once.
3. The official test split is evaluated **exactly once more**, at the end, on one ensemble locked
   beforehand in `selection_locked_before_test_v2.json`. Both the round-1 and round-2 test numbers
   are reported, and the README discloses that the test split has been scored twice in total.
4. Final-model retraining on train+val is allowed (standard practice) but the number of epochs must
   be taken from the validation-selected runs, never tuned against test.
5. Caveat to state in the write-up: released ThermoMPNN was trained on the mega-split train set, so on
   CV folds some CV-test proteins are inside its training data. CV-fold comparisons against released
   ThermoMPNN therefore favour ThermoMPNN; our CV numbers are reported as a conservative check, and
   the headline paired comparison stays on the official test split where the released weights are valid.

## Success criterion
Round 2 is reported as an improvement only if it beats round 1 **on validation and on the CV folds**.
If it beats round 1 on validation but loses on the final test evaluation, both numbers are published.
