"""Final fair comparison: this model vs ThermoMPNN retrained under an identical protocol.

Both sides see exactly the same fold-train proteins, select on the same fold-validation, and are
scored once on the same held-out fold-test proteins. This removes the confound that the *released*
ThermoMPNN was trained on the mega-split train set and therefore has seen most CV-fold test proteins.

  python final.py           per-fold + official-split comparison -> final_results.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, "/marimo/work")
from ens2 import TRUTH, load, mix, per_prot, thermompnn_rho  # noqa: E402
from ens3 import strategies  # noqa: E402

W = Path("/marimo/work")
TRAINVAL_TAG = "R1_rank_trainval"


def our_fold(split, strategy):
    cfgs = {t: d for t, d in load(split).items() if t != TRAINVAL_TAG}
    if len(cfgs) < 2:
        return None, None
    mem = strategies(cfgs)[strategy]
    return per_prot(mix(mem, cfgs, "test")), mem


def thermo_retrained(split):
    """Per-protein Spearman of the retrained ThermoMPNN, averaged over seeds."""
    acc, n = {}, 0
    for f in sorted((W / "runs_thermo").glob(f"thermo_{split}_seed*.json")):
        j = json.loads(f.read_text())
        pp = j.get("test_per_protein") or {}
        if not pp:
            continue
        n += 1
        for k, v in pp.items():
            acc[k] = acc.get(k, 0.0) + v
    if not n:
        return pd.Series(dtype=float), 0
    return pd.Series({k: v / n for k, v in acc.items()}), n


def paired(a, b, name):
    j = pd.concat([a, b], axis=1, keys=["m", "r"]).dropna()
    if len(j) < 5:
        return None
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(j), (10000, len(j)))
    dm = np.median(j["m"].values[idx], 1) - np.median(j["r"].values[idx], 1)
    lo, hi = np.quantile(dm, [0.025, 0.975])
    return {"comparison": name, "n_proteins": int(len(j)),
            "ours_median": round(float(j["m"].median()), 4), "ref_median": round(float(j["r"].median()), 4),
            "delta_median": round(float(j["m"].median() - j["r"].median()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "mean_delta": round(float((j["m"] - j["r"]).mean()), 4),
            "wins": int((j["m"] > j["r"]).sum()),
            "wilcoxon_p": float(wilcoxon(j["m"], j["r"]).pvalue)}


strategy = json.loads((W / "strategy_choice.json").read_text())["chosen_strategy"]
print("CV-chosen ensembling strategy:", strategy, "\n")

rows, detail = [], {}
for k in range(5):
    sp = f"cv{k}"
    ours, mem = our_fold(sp, strategy)
    if ours is None:
        continue
    th, nseed = thermo_retrained(sp)
    r = paired(ours, th, f"{sp}: ours vs ThermoMPNN retrained (same fold-train)")
    if r:
        r["thermo_seeds"] = nseed
        r["our_members"] = len(mem)
        rows.append(r)
        print(json.dumps(r), flush=True)
    detail[sp] = {"ours": ours.to_dict(), "thermo_retrained": th.to_dict()}

summary = {}
if rows:
    d = pd.DataFrame(rows)
    d.to_csv(W / "final_cv_fair.csv", index=False)
    summary["cv_folds"] = {
        "n_folds": len(d), "folds_won": int((d["delta_median"] > 0).sum()),
        "mean_delta_median": round(float(d["delta_median"].mean()), 4),
        "ours_mean_median": round(float(d["ours_median"].mean()), 4),
        "thermo_mean_median": round(float(d["ref_median"].mean()), 4),
        "per_fold": rows,
    }
    print("\n=== FAIR CV SUMMARY ===")
    print(f"folds won: {summary['cv_folds']['folds_won']}/{len(d)} | "
          f"ours {summary['cv_folds']['ours_mean_median']} vs retrained ThermoMPNN "
          f"{summary['cv_folds']['thermo_mean_median']} | mean delta {summary['cv_folds']['mean_delta_median']}")

# official split: the locked ensemble vs released AND retrained ThermoMPNN
lock = W / "selection_locked_before_test_v2.json"
if lock.exists():
    sel = json.loads(lock.read_text())
    cfgs = {t: d2 for t, d2 in load("mega").items() if t != TRAINVAL_TAG}
    ours = per_prot(mix(sel["members"], cfgs, "test"))
    rel = thermompnn_rho(list(ours.index))
    ret, nseed = thermo_retrained("mega")
    out = [r for r in (paired(ours, rel, "official test: ours vs ThermoMPNN RELEASED weights"),
                       paired(ours, ret, "official test: ours vs ThermoMPNN RETRAINED (same protocol)")) if r]
    for r in out:
        print(json.dumps(r))
    summary["official_test"] = {"locked_members": sel["members"], "thermo_retrain_seeds": nseed, "results": out}
    pd.DataFrame({"ours": ours, "thermompnn_released": rel, "thermompnn_retrained": ret}).to_csv(
        W / "final_official_per_protein.csv")

(W / "final_results.json").write_text(json.dumps(summary, indent=1))
(W / "final_cv_per_protein.json").write_text(json.dumps(detail, indent=1))
print("\nwrote final_results.json")
