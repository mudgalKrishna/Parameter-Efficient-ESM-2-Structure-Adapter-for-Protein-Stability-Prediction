"""Fair S669 transfer comparison: fusion model vs ThermoMPNN retrained on the same proteins.

  python s669_compare.py T_3B

Both sides train on mega `train_s669` (every S669 homologue removed), select on mega val, and
score the external S669 benchmark. The two PDB-parsing paths place slightly different variant
subsets, so **the comparison is restricted to the variants both sides scored** -- otherwise the
two pooled Spearmans are computed on different data and are not comparable.

Significance is a paired bootstrap over variants on the difference in pooled Spearman.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

W = Path("/marimo/work")
AA = "ACDEFGHIKLMNPQRSTVWY"
TAG = sys.argv[1] if len(sys.argv) > 1 else "T_3B"

S6 = np.load(W / "s669_tensors.npz", allow_pickle=True)
PIDS = [str(p) for p in S6["protein_ids"]]
Y, M = S6["labels_raw"], S6["label_mask"]

# ---- ours: seed-averaged per-chain [L,20] blocks -> {(chain, idx, mut): pred}
acc, nseed = {}, 0
for f in sorted((W / "runs3" / TAG).glob("s669_seed*.npz")):
    z = np.load(f, allow_pickle=True)
    nseed += 1
    for i, p in enumerate(z["test_pids"]):
        p = str(p)
        acc[p] = z["test"][i] if p not in acc else acc[p] + z["test"][i]
ours, truth = {}, {}
for i, p in enumerate(PIDS):
    if p not in acc:
        continue
    a = acc[p] / nseed
    Wd = min(a.shape[0], M.shape[1])
    chain = p.replace("S669_", "")
    for r, c in zip(*np.nonzero(M[i][:Wd])):
        ours[(chain, int(r), AA[c])] = float(a[r, c])
        truth[(chain, int(r), AA[c])] = float(Y[i][r, c])
print(f"ours ({TAG}, {nseed} seeds): {len(ours)} variants")

# ---- ThermoMPNN: seed-averaged, keyed the same way
tacc, tn = {}, 0
for f in sorted((W / "runs_thermo").glob("thermo_s669_seed*.npz")):
    z = np.load(f, allow_pickle=True)
    tn += 1
    for t, ix, mu, pr in zip(z["tags"], z["idx"], z["mut"], z["pred"]):
        k = (str(t), int(ix), str(mu))
        tacc[k] = float(pr) if k not in tacc else tacc[k] + float(pr)
thermo = {k: v / tn for k, v in tacc.items()}
print(f"thermompnn retrained ({tn} seeds): {len(thermo)} variants")

keys = sorted(set(ours) & set(thermo))
print(f"intersection scored by both: {len(keys)} variants, "
      f"{len({k[0] for k in keys})} chains")
if len(keys) < 100:
    print("too few shared variants to compare")
    raise SystemExit(1)

a = np.array([ours[k] for k in keys])
b = np.array([thermo[k] for k in keys])
y = np.array([truth[k] for k in keys])

sa, sb = spearmanr(a, y).correlation, spearmanr(b, y).correlation
pa, pb = pearsonr(a, y)[0], pearsonr(b, y)[0]

rng = np.random.default_rng(0)
idx = rng.integers(0, len(y), (2000, len(y)))
d = np.array([spearmanr(a[i], y[i]).correlation - spearmanr(b[i], y[i]).correlation for i in idx])
lo, hi = np.nanquantile(d, [0.025, 0.975])

res = {
    "benchmark": "S669 (external transfer test)",
    "n_variants_compared": int(len(keys)),
    "n_chains": len({k[0] for k in keys}),
    "ours_tag": TAG, "ours_seeds": nseed, "thermo_seeds": tn,
    "ours_pooled_spearman": round(float(sa), 4),
    "thermompnn_retrained_pooled_spearman": round(float(sb), 4),
    "delta_spearman": round(float(sa - sb), 4),
    "delta_ci95": [round(float(lo), 4), round(float(hi), 4)],
    "frac_bootstrap_favouring_ours": round(float((d > 0).mean()), 4),
    "ours_pooled_pearson": round(float(pa), 4),
    "thermompnn_retrained_pooled_pearson": round(float(pb), 4),
    "note": ("both sides trained on mega train_s669 (232 proteins, all S669 homologues removed), "
             "selected on mega val, scored on the same converted S669 variants"),
}
print(json.dumps(res, indent=1))
(W / "s669_comparison.json").write_text(json.dumps(res, indent=1))
print("\nwrote s669_comparison.json")
