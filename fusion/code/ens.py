"""Validation-only ensemble selection; optional single locked test evaluation vs released ThermoMPNN.

python ens.py            -> validation report + greedy ensemble chosen on val (writes selection.json)
python ens.py --test     -> scores the LOCKED selection.json on test once, paired vs ThermoMPNN
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

import paths

W = paths.WORK
z = zipfile.ZipFile(paths.BUNDLE)
D = {}
for sp in ("val", "test"):
    t = np.load(io.BytesIO(z.read(f"tensors/{sp}_tensors.npz")))
    D[sp] = {k: t[k] for k in t.files}


def per_prot(pred, sp):
    d = D[sp]
    out = {}
    for i, pid in enumerate(d["protein_ids"]):
        m = d["label_mask"][i]
        if m.sum() >= 3:
            out[str(pid)] = spearmanr(pred[i][m], d["labels_raw"][i][m]).correlation
    return pd.Series(out)


def zs(pred, sp):
    """Per-protein z-score so members on different scales average fairly (rank metric is unaffected)."""
    d = D[sp]
    p = pred.copy()
    for i in range(len(p)):
        L = int(d["lengths"][i])
        v = p[i, :L]
        p[i, :L] = (v - v.mean()) / (v.std() + 1e-8)
    return p


runs = {}
for f in sorted(W.glob("runs/*/seed*.npz")):
    tag = f.parent.name
    if tag == "smoke":
        continue
    a = np.load(f)
    runs.setdefault(tag, []).append((f.stem, zs(a["val"], "val"), a["test"]))

rows = []
cfg_val = {}
for tag, lst in runs.items():
    seeds_med = [per_prot(v, "val").median() for _, v, _ in lst]
    ens = np.mean([v for _, v, _ in lst], 0)
    cfg_val[tag] = ens
    rows.append({"config": tag, "n_seeds": len(lst), "seed_mean_val": np.mean(seeds_med), "seed_std_val": np.std(seeds_med),
                 "seed_ens_val": per_prot(ens, "val").median()})
rep = pd.DataFrame(rows).sort_values("seed_ens_val", ascending=False)
print(rep.round(4).to_string(index=False))

if "--test" not in sys.argv:
    # greedy forward selection (with replacement) of config-level seed ensembles, on val only.
    # Criterion = mean per-protein Spearman (less noisy than the median over 31 proteins).
    chosen, cur = [], None
    best = -1
    for _ in range(10):
        cand = None
        for tag, v in cfg_val.items():
            trial = v if cur is None else (cur * len(chosen) + v) / (len(chosen) + 1)
            s = per_prot(trial, "val").mean()
            if s > best + 1e-4:
                best, cand = s, tag
        if cand is None:
            break
        chosen.append(cand)
        cur = np.mean([cfg_val[t] for t in chosen], 0)
        print("greedy add", cand, "-> val mean", round(best, 4), "median", round(per_prot(cur, "val").median(), 4))
    sel = {"members": chosen, "val_mean_spearman": best, "val_median_spearman": per_prot(cur, "val").median(),
           "single_best_config": rep.iloc[0]["config"]}
    (W / "selection.json").write_text(json.dumps(sel, indent=1))
    print("LOCKED selection:", sel)
else:
    sel = json.loads((W / "selection.json").read_text())
    print("evaluating locked selection on TEST once:", sel)

    def cfg_test(tag):
        return np.mean([zs(t, "test") for _, _, t in runs[tag]], 0)

    ens_test = np.mean([cfg_test(t) for t in sel["members"]], 0)
    single_test = cfg_test(sel["single_best_config"])
    ours = per_prot(ens_test, "test")
    single = per_prot(single_test, "test")

    th = pd.read_parquet(W / "thermompnn_all_preds.parquet")
    th = th[th["split"] == "test"]
    tp = np.full_like(D["test"]["labels_raw"], np.nan)
    pid_i = {str(p): i for i, p in enumerate(D["test"]["protein_ids"])}
    AA = "ACDEFGHIKLMNPQRSTVWY"
    for r in th.itertuples():
        # released ThermoMPNN uses positive = destabilizing (train-overlap r = -0.84 vs ddG_ML): flip once
        tp[pid_i[r.protein_id], r.pos - 1, AA.index(r.mut_aa)] = -r.ddG_pred
    d = D["test"]
    thermo = {}
    for i, pid in enumerate(d["protein_ids"]):
        m = d["label_mask"][i] & np.isfinite(tp[i])
        assert m.sum() == d["label_mask"][i].sum(), pid  # identical variants
        thermo[str(pid)] = spearmanr(tp[i][m], d["labels_raw"][i][m]).correlation
    thermo = pd.Series(thermo)

    def paired(a, b, name):
        j = pd.concat([a, b], axis=1, keys=["m", "r"]).dropna()
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(j), (10000, len(j)))
        dm = np.median(j["m"].values[idx], 1) - np.median(j["r"].values[idx], 1)
        lo, hi = np.quantile(dm, [0.025, 0.975])
        return {"comparison": name, "n_proteins": len(j), "model_median": round(j["m"].median(), 4), "thermompnn_median": round(j["r"].median(), 4),
                "delta_median": round(j["m"].median() - j["r"].median(), 4), "ci95": [round(lo, 4), round(hi, 4)],
                "mean_delta": round((j["m"] - j["r"]).mean(), 4), "wins": int((j["m"] > j["r"]).sum()),
                "wilcoxon_p": float(wilcoxon(j["m"], j["r"]).pvalue)}

    res = [paired(ours, thermo, "locked ensemble vs ThermoMPNN"), paired(single, thermo, "single best config (seed-ens) vs ThermoMPNN")]
    for r in res:
        print(json.dumps(r))
    pd.DataFrame({"ours_ensemble": ours, "ours_single": single, "thermompnn": thermo}).to_csv(W / "test_per_protein.csv")
    (W / "test_result.json").write_text(json.dumps({"selection": sel, "results": res}, indent=1))
