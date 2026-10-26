"""Round-2 selection and evaluation.

  python ens2.py              validation report + greedy ensemble (mega split) -> selection_v2.json
  python ens2.py --cv         score each CV fold's test once, vs released ThermoMPNN (conservative)
  python ens2.py --test       single locked evaluation on the official test split
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

W = Path("/marimo/work")
AA = "ACDEFGHIKLMNPQRSTVWY"
z = zipfile.ZipFile("/marimo/training_bundle_copy.zip")

# ---- truth, keyed by protein_id (splits pooled; padded to a common length)
raw = {sp: dict(np.load(io.BytesIO(z.read(f"tensors/{sp}_tensors.npz")))) for sp in ("train", "val", "test")}
LMAX = max(d["seq_indices"].shape[1] for d in raw.values())


def pad(a, val=0):
    if a.shape[1] == LMAX:
        return a
    w = [(0, 0), (0, LMAX - a.shape[1])] + [(0, 0)] * (a.ndim - 2)
    return np.pad(a, w, constant_values=val)


TRUTH = {}
for sp, d in raw.items():
    y, m, L = pad(d["labels_raw"]), pad(d["label_mask"], False), d["lengths"]
    for i, pid in enumerate(d["protein_ids"]):
        TRUTH[str(pid)] = {"y": np.nan_to_num(y[i]), "m": m[i], "L": int(L[i])}


def rho(pred_row, pid):
    m = TRUTH[pid]["m"]
    if m.sum() < 3:
        return np.nan
    return spearmanr(pred_row[m], TRUTH[pid]["y"][m]).correlation


def per_prot(pred_by_pid):
    return pd.Series({p: rho(v, p) for p, v in pred_by_pid.items()}).dropna()


def zs(row, pid):
    """Per-protein standardisation so members on different scales average fairly (rank metric unaffected)."""
    L = TRUTH[pid]["L"]
    out = row.copy()
    v = out[:L]
    out[:L] = (v - v.mean()) / (v.std() + 1e-8)
    return out


def load(split):
    """-> {config: {"val": {pid: row}, "test": {pid: row}} } averaged over seeds, z-scored."""
    cfgs = {}
    for f in sorted(W.glob(f"runs2/*/{split}_seed*.npz")):
        tag = f.parent.name
        if tag.startswith("smoke"):
            continue
        a = np.load(f, allow_pickle=False)
        d = cfgs.setdefault(tag, {"val": {}, "test": {}, "n": 0})
        d["n"] += 1
        for part in ("val", "test"):
            for pid, row in zip(a[f"{part}_pids"], a[part]):
                pid = str(pid)
                d[part][pid] = d[part].get(pid, 0) + zs(row, pid)
    for d in cfgs.values():
        for part in ("val", "test"):
            d[part] = {p: v / d["n"] for p, v in d[part].items()}
    return cfgs


def mix(members, cfgs, part):
    pids = list(cfgs[members[0]][part])
    return {p: np.mean([cfgs[t][part][p] for t in members], 0) for p in pids}


def thermompnn_rho(pids, require_full=True):
    """Released-ThermoMPNN per-protein Spearman. Prefers the dense all-variant file so the
    CV folds (whose proteins sit in the official train/val splits) get full coverage."""
    dense = W / "thermompnn_all_variants.parquet"
    if dense.exists():
        th = pd.read_parquet(dense)
    else:
        th = pd.read_parquet(W / "thermompnn_all_preds.parquet")
        th = th[th["split"] == "test"]
    out = {}
    byp = {p: g for p, g in th.groupby("protein_id")}
    for pid in pids:
        if pid not in byp:
            continue
        g = byp[pid]
        pred = np.full((LMAX, 20), np.nan)
        for r in g.itertuples():
            pred[r.pos - 1, AA.index(r.mut_aa)] = -r.ddG_pred  # released model: positive = destabilising
        truth_m = TRUTH[pid]["m"]
        m = truth_m & np.isfinite(pred)
        # only compare where ThermoMPNN covers (essentially) the same variants scored here
        if require_full and truth_m.sum() and m.sum() / truth_m.sum() < 0.99:
            continue
        if m.sum() >= 3:
            out[pid] = spearmanr(pred[m], TRUTH[pid]["y"][m]).correlation
    return pd.Series(out)


def paired(a, b, name):
    j = pd.concat([a, b], axis=1, keys=["m", "r"]).dropna()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(j), (10000, len(j)))
    dm = np.median(j["m"].values[idx], 1) - np.median(j["r"].values[idx], 1)
    lo, hi = np.quantile(dm, [0.025, 0.975])
    return {"comparison": name, "n_proteins": int(len(j)),
            "model_median": round(float(j["m"].median()), 4), "ref_median": round(float(j["r"].median()), 4),
            "delta_median": round(float(j["m"].median() - j["r"].median()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "mean_delta": round(float((j["m"] - j["r"]).mean()), 4),
            "wins": int((j["m"] > j["r"]).sum()),
            "wilcoxon_p": float(wilcoxon(j["m"], j["r"]).pvalue)}


if __name__ == "__main__":
    if "--cv" in sys.argv:
        sel = json.loads((W / "selection_v2.json").read_text())
        rows, per_fold = [], {}
        for k in range(5):
            cfgs = load(f"cv{k}")
            have = [t for t in sel["members"] if t in cfgs]
            if not have:
                print(f"cv{k}: no runs yet")
                continue
            ours = per_prot(mix(have, cfgs, "test"))
            th = thermompnn_rho(list(ours.index))
            r = paired(ours, th, f"cv{k} ensemble vs released ThermoMPNN")
            r["members_available"] = len(have)
            rows.append(r)
            per_fold[f"cv{k}"] = {"ours": ours.to_dict(), "thermompnn": th.to_dict()}
            print(json.dumps(r), flush=True)
        if rows:
            d = pd.DataFrame(rows)
            d.to_csv(W / "cv_results.csv", index=False)
            print("\nmean delta over folds:", round(d["delta_median"].mean(), 4),
                  "| folds won:", int((d["delta_median"] > 0).sum()), "/", len(d))
            (W / "cv_per_protein.json").write_text(json.dumps(per_fold, indent=1))

    elif "--test" in sys.argv:
        sel = json.loads((W / "selection_locked_before_test_v2.json").read_text())
        print("LOCKED round-2 selection:", sel["members"], flush=True)
        cfgs = load("mega")
        ours = per_prot(mix(sel["members"], cfgs, "test"))
        th = thermompnn_rho(list(ours.index))
        res = [paired(ours, th, "round-2 locked ensemble vs released ThermoMPNN")]
        best1 = sel.get("single_best_config")
        if best1 in cfgs:
            res.append(paired(per_prot(mix([best1], cfgs, "test")), th, f"best single config {best1} vs released ThermoMPNN"))
        for r in res:
            print(json.dumps(r))
        pd.DataFrame({"ours_v2": ours, "thermompnn": th}).to_csv(W / "test_per_protein_v2.csv")
        (W / "test_result_v2.json").write_text(json.dumps({"selection": sel, "results": res}, indent=1))

    else:
        cfgs = load("mega")
        rows = []
        for tag, d in cfgs.items():
            rows.append({"config": tag, "n_seeds": d["n"], "val_median": per_prot(d["val"]).median(),
                         "val_mean": per_prot(d["val"]).mean()})
        rep = pd.DataFrame(rows).sort_values("val_median", ascending=False)
        print(rep.round(4).to_string(index=False))
        chosen, best = [], -1
        for _ in range(10):
            cand = None
            for tag in cfgs:
                s = per_prot(mix(chosen + [tag], cfgs, "val")).mean()
                if s > best + 1e-4:
                    best, cand = s, tag
            if cand is None:
                break
            chosen.append(cand)
            print("greedy add", cand, "-> val mean", round(best, 4),
                  "median", round(per_prot(mix(chosen, cfgs, "val")).median(), 4))
        sel = {"members": chosen, "val_mean_spearman": best,
               "val_median_spearman": float(per_prot(mix(chosen, cfgs, "val")).median()),
               "single_best_config": rep.iloc[0]["config"]}
        (W / "selection_v2.json").write_text(json.dumps(sel, indent=1))
        print("selection_v2:", sel["members"], "val median", round(sel["val_median_spearman"], 4))
