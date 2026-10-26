"""Results figure for the README.

Three panels: (A) paired per-protein comparison on the official test split,
(B) the fair per-fold comparison, (C) why the released weights cannot be used on the folds.
"""
import json
from pathlib import Path

import matplotlib

import paths
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

W = paths.WORK
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8a84"
OURS = "#2a78d6"      # categorical slot 1
THERMO = "#eb6834"    # categorical slot 2

off = pd.read_csv(W / "final_official_per_protein.csv", index_col=0)
fair = pd.read_csv(W / "final_cv_fair.csv")
rel = pd.read_csv(W / "cv_vs_thermompnn.csv")
res = json.loads((W / "final_results.json").read_text())

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), facecolor=SURFACE)
for a in axes:
    a.set_facecolor(SURFACE)

# ---- Panel A: paired per-protein scatter (official test split)
ax = axes[0]
x, y = off["thermompnn_retrained"], off["ours"]
lo, hi = 0.40, 1.0
ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", zorder=1)
ax.scatter(x, y, s=58, color=OURS, edgecolor=SURFACE, linewidth=2, zorder=3)
wins = int((y > x).sum())
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
ax.set_xlabel("ThermoMPNN retrained — per-protein Spearman $\\rho$")
ax.set_ylabel("This work — per-protein Spearman $\\rho$")
ax.set_title("A · Official test split, paired by protein", loc="left", fontsize=11, color=INK, pad=10)
ax.grid(True, color="#e8e7e3", lw=0.8, zorder=0)
ax.set_axisbelow(True)
ax.text(0.045, 0.955, f"{wins} of {len(off)} proteins above the line\n"
                      f"median {y.median():.3f} vs {x.median():.3f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=9.5, color=INK2,
        bbox=dict(facecolor=SURFACE, edgecolor="#e0dfda", boxstyle="round,pad=0.45"))
ax.annotate("points above the line =\nthis work scores higher", xy=(0.97, 0.06),
            xycoords="axes fraction", fontsize=8.5, color=MUTED, ha="right", va="bottom")

# ---- Panel B: fair per-fold comparison
ax = axes[1]
folds = [c.split(":")[0] for c in fair["comparison"]]
xs = range(len(folds))
w = 0.38
ax.bar([i - w / 2 - 0.012 for i in xs], fair["ours_median"], w, color=OURS, label="This work", zorder=3)
ax.bar([i + w / 2 + 0.012 for i in xs], fair["ref_median"], w, color=THERMO, label="ThermoMPNN retrained", zorder=3)
for i, (o, r) in enumerate(zip(fair["ours_median"], fair["ref_median"])):
    ax.text(i - w / 2 - 0.012, o + 0.008, f"{o:.3f}", ha="center", fontsize=8.5, color=INK2)
    ax.text(i + w / 2 + 0.012, r + 0.008, f"{r:.3f}", ha="center", fontsize=8.5, color=INK2)
ax.set_xticks(list(xs)); ax.set_xticklabels([f.upper() for f in folds])
ax.set_ylim(0, 1.0)
ax.set_ylabel("Median per-protein Spearman $\\rho$")
ax.set_title("B · Identical protocol, 5 held-out folds", loc="left", fontsize=11, color=INK, pad=10)
ax.legend(frameon=False, loc="upper right", fontsize=9, ncol=1)
ax.grid(True, axis="y", color="#e8e7e3", lw=0.8, zorder=0)
ax.set_axisbelow(True)
cv = res["cv_folds"]
ax.text(0.02, 0.955, f"won {cv['folds_won']}/{cv['n_folds']} folds · mean $\\Delta$ +{cv['mean_delta_median']:.3f}",
        transform=ax.transAxes, va="top", fontsize=9.5, color=INK2,
        bbox=dict(facecolor=SURFACE, edgecolor="#e0dfda", boxstyle="round,pad=0.45"))

# ---- Panel C: why released weights can't be scored on the folds
ax = axes[2]
rel_by = dict(zip([c for c in rel["comparison"]], rel["ref_median"]))
labels = ["Official\ntest"] + [f.upper() for f in folds]
released = [off["thermompnn_released"].median()] + [rel_by.get(f, float("nan")) for f in folds]
retrained = [off["thermompnn_retrained"].median()] + list(fair["ref_median"])
xs = range(len(labels))
ax.bar([i - w / 2 - 0.012 for i in xs], released, w, color=THERMO, hatch="////",
       edgecolor=SURFACE, linewidth=1.4, label="Released weights", zorder=3)
ax.bar([i + w / 2 + 0.012 for i in xs], retrained, w, color=THERMO, label="Retrained on fold-train", zorder=3)
ax.set_xticks(list(xs)); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 1.0)
ax.set_ylabel("ThermoMPNN median $\\rho$")
ax.set_title("C · The released weights saw the fold proteins", loc="left", fontsize=11, color=INK, pad=10)
ax.legend(frameon=False, loc="upper right", fontsize=9, ncol=1)
ax.grid(True, axis="y", color="#e8e7e3", lw=0.8, zorder=0)
ax.set_axisbelow(True)
gap = sum(released[1:]) / 5 - sum(retrained[1:]) / 5
ax.text(0.02, 0.955, f"unseen test: parity\nfolds: +{gap:.3f} inflated",
        transform=ax.transAxes, va="top", fontsize=9.5, color=INK2,
        bbox=dict(facecolor=SURFACE, edgecolor="#e0dfda", boxstyle="round,pad=0.45"))
ax.axvline(0.5, color=MUTED, lw=1, ls=":", ymax=0.78)

fig.tight_layout(pad=1.6)
fig.savefig(paths.FIGURE_PNG, dpi=170, facecolor=SURFACE, bbox_inches="tight")
print("saved", paths.FIGURE_PNG)
print("panelA wins", wins, "| panelB folds", folds, "| memorisation gap", round(gap, 4))
