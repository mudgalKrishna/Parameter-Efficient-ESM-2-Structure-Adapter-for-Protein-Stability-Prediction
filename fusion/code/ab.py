"""Paired A/B of round-3 arms on ThermoMPNN's CV folds.

  python ab.py <tagA> <tagB> [...]       compare arms; first tag is the baseline

Per (fold, seed) a run yields a per-protein Spearman on that fold's val and test proteins.
Arms are compared **paired on (fold, seed)**, so the comparison is not confounded by which folds
or seeds happened to run. Selection is reported on val; test is shown alongside but, per
`PREREGISTRATION_R3.md`, the arm to report is chosen on val only.
"""
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

import paths

W = paths.WORK
BUNDLE = paths.BUNDLE

# ---- truth, keyed by protein id
z = zipfile.ZipFile(BUNDLE)
TRUTH = {}
for sp in ("train", "val", "test"):
    t = np.load(io.BytesIO(z.read(f"tensors/{sp}_tensors.npz")))
    for i, pid in enumerate(t["protein_ids"]):
        TRUTH[str(pid)] = (np.nan_to_num(t["labels_raw"][i]), t["label_mask"][i])


def per_protein(pids, arr):
    """{pid: spearman} for one run's prediction block."""
    out = {}
    for i, pid in enumerate(pids):
        pid = str(pid)
        y, m = TRUTH[pid]
        # each bundle split file has its own pad width, and a prediction block is padded to the
        # longest protein in its own batch list; both exceed the protein's length, so clip to the
        # shorter of the two
        Wd = min(arr.shape[1], m.shape[0])
        m = m[:Wd]
        y = y[:Wd]
        if m.sum() < 3:
            continue
        r = spearmanr(arr[i][:Wd][m], y[m]).correlation
        if np.isfinite(r):
            out[pid] = float(r)
    return out


def load_arm(tag):
    """{(split, seed): {'val': {...}, 'test': {...}}}"""
    runs = {}
    d = W / "runs3" / tag
    if not d.is_dir():
        return runs
    for f in sorted(d.glob("*.npz")):
        sp, sd = f.stem.rsplit("_seed", 1)
        z2 = np.load(f, allow_pickle=True)
        runs[(sp, int(sd))] = {
            "val": per_protein(z2["val_pids"], z2["val"]),
            "test": per_protein(z2["test_pids"], z2["test"]),
        }
    return runs


tags = sys.argv[1:]
if not tags:
    tags = sorted(p.name for p in (W / "runs3").iterdir() if p.is_dir())
arms = {t: load_arm(t) for t in tags}
for t, r in arms.items():
    print(f"{t:14s} {len(r)} runs: {sorted(r.keys())}")

rows = []
for t, runs in arms.items():
    for (sp, sd), d in runs.items():
        rows.append({"arm": t, "split": sp, "seed": sd,
                     "val_median": float(np.median(list(d["val"].values()))),
                     "test_median": float(np.median(list(d["test"].values()))),
                     "n_val": len(d["val"]), "n_test": len(d["test"])})
df = pd.DataFrame(rows)
if df.empty:
    print("no runs found")
    raise SystemExit(0)

print("\n=== per arm, mean over (fold, seed) ===")
agg = df.groupby("arm")[["val_median", "test_median"]].agg(["mean", "std", "count"]).round(4)
print(agg.to_string())

base = tags[0]
print(f"\n=== paired vs baseline '{base}' (matched on fold+seed) ===")
out = {}
for t in tags[1:]:
    j = df[df.arm == base].merge(df[df.arm == t], on=["split", "seed"], suffixes=("_b", "_t"))
    if j.empty:
        print(f"{t}: no matching (fold, seed) pairs")
        continue
    dv = j["val_median_t"] - j["val_median_b"]
    dt = j["test_median_t"] - j["test_median_b"]
    rec = {"arm": t, "n_pairs": len(j),
           "val_delta_mean": round(float(dv.mean()), 4), "val_wins": int((dv > 0).sum()),
           "test_delta_mean": round(float(dt.mean()), 4), "test_wins": int((dt > 0).sum()),
           "base_val": round(float(j["val_median_b"].mean()), 4),
           "arm_val": round(float(j["val_median_t"].mean()), 4),
           "base_test": round(float(j["test_median_b"].mean()), 4),
           "arm_test": round(float(j["test_median_t"].mean()), 4)}
    if len(j) >= 6:
        rec["val_wilcoxon_p"] = float(wilcoxon(j["val_median_t"], j["val_median_b"]).pvalue)
        rec["test_wilcoxon_p"] = float(wilcoxon(j["test_median_t"], j["test_median_b"]).pvalue)
    out[t] = rec
    print(json.dumps(rec))

# per-protein paired view, pooled across seeds within a fold (the metric the README reports)
print("\n=== per-fold, per-protein (seed-averaged) ===")
fold_rows = []
for t, runs in arms.items():
    byfold = defaultdict(lambda: defaultdict(list))
    for (sp, sd), d in runs.items():
        for pid, r in d["test"].items():
            byfold[sp][pid].append(r)
    for sp, pp in byfold.items():
        med = np.median([np.mean(v) for v in pp.values()])
        fold_rows.append({"arm": t, "split": sp, "n_proteins": len(pp), "test_median": round(float(med), 4)})
fd = pd.DataFrame(fold_rows)
if not fd.empty:
    print(fd.pivot_table(index="split", columns="arm", values="test_median").round(4).to_string())

df.to_csv(W / "round3_runs.csv", index=False)
(W / "round3_ab.json").write_text(json.dumps(
    {"baseline": base, "comparisons": out,
     "arm_means": {t: {"val": round(float(g["val_median"].mean()), 4),
                       "test": round(float(g["test_median"].mean()), 4),
                       "n": int(len(g))} for t, g in df.groupby("arm")}}, indent=1))
print("\nwrote round3_runs.csv and round3_ab.json")
