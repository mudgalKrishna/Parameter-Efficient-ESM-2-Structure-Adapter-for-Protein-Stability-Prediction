"""Paired protein-level bootstrap: 650M vs ThermoMPNN vs 150M (28 test proteins)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

A = Path(os.environ.get("DIR_150", "xfer150/esm2_structure_track_150M"))
B = Path(os.environ.get("DIR_650", "xfer650/esm2_structure_track_650M"))
OUT = Path(os.environ.get("OUT_CSV", "xfer650/paired_bootstrap_cross_model.csv"))

MODELS = {
    "650M LoRA+ProteinMPNN+SPURS": B / "per_protein__HC06_ESM2_Lora_Plus_ProteinMPNN_SPURSAdapter.csv",
    "650M full-FT seq-only": B / "per_protein__HC04_ESM2_Full_Finetune_SeqOnly.csv",
    "150M LoRA+ProteinMPNN+SPURS": A / "per_protein__HC09_ESM2_Lora_Plus_ProteinMPNN_SPURSAdapter.csv",
    "150M full-FT seq-only": A / "per_protein__HC06_ESM2_Full_Finetune_SeqOnly.csv",
    "ThermoMPNN (released)": B / "per_protein__BASELINE_ThermoMPNN_Released.csv",
    "Zero-shot masked-marginal": B / "per_protein__HC01_Baseline_ZeroShot_MaskedMarginal.csv",
    "Ridge 3D": B / "per_protein__HC03_Baseline_Ridge_HandCrafted3D.csv",
}


def load(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)[["protein_id", "n_variants", "spearman_rho"]]
    return df.set_index("protein_id")


def boot(m: pd.DataFrame, r: pd.DataFrame, n_boot: int = 10000, seed: int = 42):
    j = m.join(r, how="inner", lsuffix="_m", rsuffix="_r").dropna()
    vm = j["spearman_rho_m"].to_numpy(float)
    vr = j["spearman_rho_r"].to_numpy(float)
    d = vm - vr
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    dm = np.median(vm[idx], axis=1) - np.median(vr[idx], axis=1)
    try:
        from scipy.stats import wilcoxon
        p = 1.0 if np.allclose(d, 0) else float(wilcoxon(vm, vr, zero_method="zsplit").pvalue)
    except Exception:
        p = float("nan")
    lo, hi = np.quantile(dm, [0.025, 0.975])
    return dict(
        n_proteins=int(len(d)),
        n_variants=int(j["n_variants_m"].sum()),
        model_median=round(float(np.median(vm)), 4),
        ref_median=round(float(np.median(vr)), 4),
        delta_median=round(float(np.median(vm) - np.median(vr)), 4),
        delta_median_ci_low=round(float(lo), 4),
        delta_median_ci_high=round(float(hi), 4),
        delta_mean=round(float(d.mean()), 4),
        wilcoxon_p=round(p, 5),
        resolvable_95ci=bool(lo > 0 or hi < 0),
    )


def main():
    data = {k: load(v) for k, v in MODELS.items() if v.exists()}
    missing = [k for k, v in MODELS.items() if not v.exists()]
    if missing:
        print("MISSING", missing)

    pairs = [
        ("650M LoRA+ProteinMPNN+SPURS", "ThermoMPNN (released)"),
        ("650M full-FT seq-only", "ThermoMPNN (released)"),
        ("650M LoRA+ProteinMPNN+SPURS", "150M LoRA+ProteinMPNN+SPURS"),
        ("650M full-FT seq-only", "150M full-FT seq-only"),
        ("150M LoRA+ProteinMPNN+SPURS", "ThermoMPNN (released)"),
        ("650M LoRA+ProteinMPNN+SPURS", "650M full-FT seq-only"),
        ("650M LoRA+ProteinMPNN+SPURS", "Ridge 3D"),
    ]
    rows = []
    for m, r in pairs:
        if m in data and r in data:
            row = {"model": m, "reference": r, **boot(data[m], data[r])}
            rows.append(row)
            print(json.dumps(row), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print("\nSaved", OUT)


if __name__ == "__main__":
    main()
