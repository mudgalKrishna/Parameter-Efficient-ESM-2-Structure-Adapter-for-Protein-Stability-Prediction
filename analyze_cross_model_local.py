"""Cross-model (150M vs 650M) paired protein-level analysis on identical test sets.

Local variant: reads the downloaded 150M artifacts and the newly downloaded 650M
artifacts, writes results next to them.
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(os.environ.get("TRACK_BASE", "xfer650"))
A = Path(os.environ.get("DIR_150", "xfer150/esm2_structure_track_150M"))
B = Path(os.environ.get("DIR_650", str(BASE / "esm2_structure_track_650M")))
OUT_CSV = BASE / "cross_model_paired_bootstrap.csv"
OUT_REP = BASE / "structure_track_final_reports.csv"
OUT_SUM = BASE / "structure_track_cross_model_summary.json"
N_BOOT = 10_000
SEED = 42


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    pa = pickle.load(open(A / "per_prot_test_tables.pkl", "rb"))
    pb = pickle.load(open(B / "per_prot_test_tables.pkl", "rb"))

    rows = []
    rng = np.random.default_rng(SEED)
    for step in sorted(set(pa) & set(pb)):
        da, db = pa[step], pb[step]
        cols = ["protein_id", "spearman_rho"]
        m = da[cols].merge(db[cols], on="protein_id", suffixes=("_150", "_650")).dropna()
        if len(m) < 5:
            continue
        d = (m["spearman_rho_650"] - m["spearman_rho_150"]).to_numpy()
        idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
        boot = d[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
        rows.append({
            "step_id": step,
            "n_proteins": int(len(m)),
            "median_rho_150M": float(m["spearman_rho_150"].median()),
            "median_rho_650M": float(m["spearman_rho_650"].median()),
            "mean_diff_650_minus_150": float(d.mean()),
            "diff_ci95_lo": float(lo),
            "diff_ci95_hi": float(hi),
            "p_boot_two_sided": float(p),
            "frac_proteins_650_better": float((d > 0).mean()),
        })

    df = pd.DataFrame(rows).sort_values("step_id")
    df.to_csv(OUT_CSV, index=False)

    rep = []
    for model, d in (("150M", A), ("650M", B)):
        f = d / "test_once_final_report.csv"
        if f.exists():
            t = pd.read_csv(f)
            t.insert(0, "model", model)
            rep.append(t)
    if rep:
        pd.concat(rep, ignore_index=True).to_csv(OUT_REP, index=False)

    summary = {
        "models": {
            m: {
                "out_dir": str(d),
                "selected": {
                    k: (v.get("step_id") if isinstance(v, dict) else None)
                    for k, v in json.loads((d / "selected_model_before_test.json").read_text()).items()
                    if k.startswith("selected_")
                },
                "run_summary": json.loads((d / "run_summary.json").read_text())
                if (d / "run_summary.json").exists()
                else {},
            }
            for m, d in (("150M", A), ("650M", B))
        },
        "cross_model_steps": int(len(df)),
    }
    OUT_SUM.write_text(json.dumps(summary, indent=2))
    print(df.to_string(index=False))
    print("wrote", OUT_CSV, OUT_REP, OUT_SUM)


if __name__ == "__main__":
    main()
