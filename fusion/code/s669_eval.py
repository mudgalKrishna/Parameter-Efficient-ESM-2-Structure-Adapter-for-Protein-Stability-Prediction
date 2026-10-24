"""Score the S669 transfer test.

  python s669_eval.py T_3B T_650

S669 carries a median of 3 variants per chain, so a median per-protein Spearman is meaningless
here. The literature scores it **pooled over all variants**, and that is the headline below
(Pearson and Spearman, plus RMSE). Per-chain numbers are reported only for the chains that have
enough variants to support one.

Seeds are averaged at the prediction level before scoring, matching how the CV-fold ensembles
were built.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

W = Path("/marimo/work")
S6 = np.load(W / "s669_tensors.npz", allow_pickle=True)
PIDS = [str(p) for p in S6["protein_ids"]]
Y, M = S6["labels_raw"], S6["label_mask"]
TRUTH = {p: (Y[i], M[i]) for i, p in enumerate(PIDS)}


def load(tag):
    """Seed-averaged predictions per chain, from runs tagged `tag` on the s669 split."""
    acc, n = {}, 0
    for f in sorted((W / "runs3" / tag).glob("s669_seed*.npz")):
        z = np.load(f, allow_pickle=True)
        n += 1
        for i, p in enumerate(z["test_pids"]):
            p = str(p)
            a = z["test"][i]
            acc[p] = a if p not in acc else acc[p] + a
    return {p: v / n for p, v in acc.items()}, n


def pooled(pred):
    """Flatten every labelled (chain, position, mutant AA) into one vector pair."""
    ps, ts = [], []
    for p, a in pred.items():
        y, m = TRUTH[p]
        Wd = min(a.shape[0], m.shape[0])
        mm = m[:Wd]
        ps.append(a[:Wd][mm])
        ts.append(y[:Wd][mm])
    return np.concatenate(ps), np.concatenate(ts)


out = {}
for tag in sys.argv[1:]:
    pred, nseed = load(tag)
    if not pred:
        print(f"{tag}: no runs")
        continue
    a, b = pooled(pred)
    sp = spearmanr(a, b).correlation
    pe = pearsonr(a, b)[0]
    # per-chain, only where there are enough variants to mean anything
    per = []
    for p, q in pred.items():
        y, m = TRUTH[p]
        Wd = min(q.shape[0], m.shape[0])
        mm = m[:Wd]
        if mm.sum() >= 5:
            r = spearmanr(q[:Wd][mm], y[:Wd][mm]).correlation
            if np.isfinite(r):
                per.append(r)
    # bootstrap CI on the pooled Spearman
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(a), (4000, len(a)))
    bs = np.array([spearmanr(a[i], b[i]).correlation for i in idx[:800]])
    lo, hi = np.nanquantile(bs, [0.025, 0.975])
    rec = {"tag": tag, "seeds": nseed, "chains": len(pred), "variants": int(len(a)),
           "pooled_spearman": round(float(sp), 4), "pooled_pearson": round(float(pe), 4),
           "ci95_spearman": [round(float(lo), 4), round(float(hi), 4)],
           "rmse": round(float(np.sqrt(((a - b) ** 2).mean())), 4),
           "n_chains_ge5": len(per),
           "median_per_chain_spearman": round(float(np.median(per)), 4) if per else None}
    out[tag] = rec
    print(json.dumps(rec))

if out:
    (W / "s669_results.json").write_text(json.dumps(out, indent=1))
    print("\nwrote s669_results.json")
    print("\nNote: the only baseline this repo compares against is a ThermoMPNN retrained on the")
    print("same train_s669 proteins and scored on the same converted S669 tensors. Published S669")
    print("numbers from other papers are NOT comparable here -- they use their own structure")
    print("preparation, variant filtering and sign conventions.")
