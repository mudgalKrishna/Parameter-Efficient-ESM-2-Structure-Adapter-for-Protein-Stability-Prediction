"""Choose the ensembling strategy on ThermoMPNN's CV folds, then apply it to the official split.

  python ens3.py            compare strategies across the 5 CV folds -> strategy_choice.json
  python ens3.py --lock     apply the CV-chosen strategy to the mega split -> selection_locked_before_test_v2.json

Nothing here reads the official test split; --lock only uses mega validation to rank configs,
with the *strategy* (how many members, how combined) already fixed by the folds.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/marimo/work")
from ens2 import TRUTH, load, mix, per_prot, thermompnn_rho, paired  # noqa: E402

W = Path("/marimo/work")
TRAINVAL_TAG = "R1_rank_trainval"


def rank_configs(cfgs, part="val"):
    """Configs ordered by their score on the given part (fold validation)."""
    s = {t: per_prot(d[part]).median() for t, d in cfgs.items()}
    return [t for t, _ in sorted(s.items(), key=lambda kv: -kv[1])], s


def greedy(cfgs, steps=6):
    chosen, best = [], -1
    for _ in range(steps):
        cand = None
        for t in cfgs:
            v = per_prot(mix(chosen + [t], cfgs, "val")).mean()
            if v > best + 1e-4:
                best, cand = v, t
        if cand is None:
            break
        chosen.append(cand)
    return chosen


def strategies(cfgs):
    """name -> member list, decided using fold validation only."""
    order, _ = rank_configs(cfgs)
    out = {"single_best": order[:1], "top2_avg": order[:2], "top3_avg": order[:3],
           "top4_avg": order[:4], "all_avg": order, "greedy_val": greedy(cfgs)}
    return {k: v for k, v in out.items() if v}


if __name__ == "__main__":
    if "--lock" in sys.argv:
        choice = json.loads((W / "strategy_choice.json").read_text())
        name = choice["chosen_strategy"]
        cfgs = {t: d for t, d in load("mega").items() if t != TRAINVAL_TAG}
        members = strategies(cfgs)[name]
        sel = {"strategy": name, "strategy_chosen_on": "5 CV folds (see strategy_choice.json)",
               "members": members, "n_configs_available": len(cfgs),
               "val_median_spearman": float(per_prot(mix(members, cfgs, "val")).median()),
               "val_mean_spearman": float(per_prot(mix(members, cfgs, "val")).mean()),
               "single_best_config": strategies(cfgs)["single_best"][0],
               "retrain_on_trainval": choice.get("trainval_helps", False)}
        (W / "selection_locked_before_test_v2.json").write_text(json.dumps(sel, indent=1))
        print(json.dumps(sel, indent=1))
    else:
        rows, tv_rows = [], []
        for k in range(5):
            allc = load(f"cv{k}")
            cfgs = {t: d for t, d in allc.items() if t != TRAINVAL_TAG}
            if len(cfgs) < 2:
                print(f"cv{k}: only {len(cfgs)} configs, skipping")
                continue
            for name, mem in strategies(cfgs).items():
                te = per_prot(mix(mem, cfgs, "test"))
                rows.append({"fold": f"cv{k}", "strategy": name, "n_members": len(mem),
                             "fold_test_median": te.median(), "fold_test_mean": te.mean()})
            # clean A/B: train-only vs retrained on fold train+val, both scored on untouched fold test
            if TRAINVAL_TAG in allc and "R1_rank" in cfgs:
                a = per_prot(mix(["R1_rank"], cfgs, "test")).median()
                b = per_prot(mix([TRAINVAL_TAG], allc, "test")).median()
                tv_rows.append({"fold": f"cv{k}", "train_only": a, "trainval": b, "delta": b - a})

        if not rows:
            print("no CV-fold runs with >=2 configs yet; nothing to compare")
            raise SystemExit(0)
        d = pd.DataFrame(rows)
        piv = d.groupby("strategy")[["fold_test_median", "fold_test_mean"]].agg(["mean", "std"]).round(4)
        print("\n=== strategy comparison on held-out CV fold test sets ===")
        print(piv.to_string())
        best = d.groupby("strategy")["fold_test_median"].mean().idxmax()
        print("\nCV-chosen strategy:", best)

        tv_helps = False
        if tv_rows:
            tv = pd.DataFrame(tv_rows)
            print("\n=== retrain on train+val (fold test, clean A/B) ===")
            print(tv.round(4).to_string(index=False))
            tv_helps = bool(tv["delta"].mean() > 0)
            print("mean delta:", round(tv["delta"].mean(), 4), "-> trainval helps:", tv_helps)

        # conservative sanity check: fold ensembles vs released ThermoMPNN (favours ThermoMPNN, see pre-registration)
        th_rows = []
        for k in range(5):
            allc = load(f"cv{k}")
            cfgs = {t: d2 for t, d2 in allc.items() if t != TRAINVAL_TAG}
            if len(cfgs) < 2:
                continue
            ours = per_prot(mix(strategies(cfgs)[best], cfgs, "test"))
            th = thermompnn_rho(list(ours.index))
            if len(th) >= 5:
                th_rows.append(paired(ours, th, f"cv{k}"))
        if th_rows:
            t = pd.DataFrame(th_rows)
            print("\n=== CV folds vs released ThermoMPNN (conservative: it trained on some of these) ===")
            print(t[["comparison", "n_proteins", "model_median", "ref_median", "delta_median", "wins"]].to_string(index=False))
            print("folds won:", int((t["delta_median"] > 0).sum()), "/", len(t),
                  "| mean delta:", round(t["delta_median"].mean(), 4))
            t.to_csv(W / "cv_vs_thermompnn.csv", index=False)

        d.to_csv(W / "cv_strategy_comparison.csv", index=False)
        (W / "strategy_choice.json").write_text(json.dumps(
            {"chosen_strategy": best, "trainval_helps": tv_helps,
             "per_strategy_fold_test_median": d.groupby("strategy")["fold_test_median"].mean().round(4).to_dict(),
             "n_folds": int(d["fold"].nunique())}, indent=1))
        print("\nwrote strategy_choice.json")
