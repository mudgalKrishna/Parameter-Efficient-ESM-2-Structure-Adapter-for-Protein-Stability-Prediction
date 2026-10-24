"""Convert the S669 benchmark into the training bundle's tensor format, for a transfer test.

S669 is an external benchmark: different proteins, different labs, different assays. Nothing here
is trained on. It answers the one question the CV-fold results cannot — does the model generalise
off the Tsuboyama dataset at all?

Sign: ThermoMPNN negates BOTH megascale `ddG_ML` (datasets.py:161) and S669 `DDG`
(datasets.py:313) by the same factor, so S669's raw convention already matches Tsuboyama's raw
convention that this model trains on. Labels are therefore used **as published, with no flip**.

Positions use author residue numbering (`MUT` = e.g. "S11A" -> resSeq "11"), matching
ThermoMPNN's `resn_list.index(pos)`, not a sequence offset.

Output: /marimo/work/s669_tensors.npz, same keys as tensors/*_tensors.npz in the bundle.
"""
from pathlib import Path

import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
AA3 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", "HIS": "H",
    "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q",
    "ARG": "R", "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "MSE": "M", "HSD": "H", "HSE": "H", "HSP": "H", "SEC": "C", "PYL": "K",
}
BB = ("N", "CA", "C", "O")

D = Path("/marimo/data_s669")
OUT = Path("/marimo/work/s669_tensors.npz")


def parse_chain(pdb_path, chain):
    """(seq, coords[L,4,3], valid[L], resn[L]) for one chain, in file order.

    resn holds the author residue number as a string (no insertion code), which is what the
    S669 MUT field indexes.
    """
    res, order = {}, []
    for line in open(pdb_path):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if line[17:20].strip() not in AA3:
            continue
        if line[21] != chain:
            continue
        if line[16] not in (" ", "A"):        # keep only altloc blank or A
            continue
        key = line[22:27]
        if key not in res:
            res[key] = {"aa": AA3[line[17:20].strip()], "xyz": {}, "resn": line[22:26].strip()}
            order.append(key)
        nm = line[12:16].strip()
        if nm in BB and nm not in res[key]["xyz"]:
            res[key]["xyz"][nm] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    seq, co, vm, resn = [], [], [], []
    for k in order:
        r = res[k]
        seq.append(r["aa"])
        co.append([r["xyz"][a] if a in r["xyz"] else (0.0, 0.0, 0.0) for a in BB])
        vm.append(all(a in r["xyz"] for a in BB))
        resn.append(r["resn"])
    return "".join(seq), np.asarray(co, np.float32), np.asarray(vm, bool), resn


def main():
    df = pd.read_csv(D / "s669_clean_dir.csv")
    print(f"S669: {len(df)} variants, {df.PDB.nunique()} chains")

    recs, dropped = [], []
    for tag, g in df.groupby("PDB"):
        pid, chain = tag[:-1], tag[-1]
        path = D / "pdbs" / f"{pid}.pdb"
        if not path.exists():
            dropped.append((tag, "no pdb", len(g)))
            continue
        seq, co, vm, resn = parse_chain(path, chain)
        if len(seq) == 0:
            dropped.append((tag, "empty chain", len(g)))
            continue
        L = len(seq)
        y = np.zeros((L, 20), np.float32)
        m = np.zeros((L, 20), bool)
        placed = miss = 0
        for r in g.itertuples():
            wt, mut, pos = r.MUT[0], r.MUT[-1], r.MUT[1:-1]
            try:
                i = resn.index(pos)
            except ValueError:
                miss += 1
                continue
            if seq[i] != wt or not vm[i]:
                miss += 1
                continue
            a = AA.find(mut)
            if a < 0:
                miss += 1
                continue
            y[i, a] = float(r.DDG)          # published convention, no flip (see module docstring)
            m[i, a] = True
            placed += 1
        if placed == 0:
            dropped.append((tag, f"0 of {len(g)} placed", len(g)))
            continue
        if miss:
            dropped.append((tag, f"{miss} of {len(g)} variants unplaced", miss))
        recs.append(dict(pid=f"S669_{tag}", seq=seq, L=L, X=co, vm=vm, y=y, m=m, n=placed))

    tot = sum(r["n"] for r in recs)
    print(f"built {len(recs)} chains, {tot} of {len(df)} variants placed ({100 * tot / len(df):.1f}%)")
    for t in dropped:
        print("   ", t)

    LMAX = max(r["L"] for r in recs)
    n = len(recs)
    out = dict(
        protein_ids=np.array([r["pid"] for r in recs]),
        lengths=np.array([r["L"] for r in recs], np.int64),
        seq_indices=np.full((n, LMAX), -1, np.int64),
        backbone_coords=np.zeros((n, LMAX, 4, 3), np.float32),
        residue_valid_mask=np.zeros((n, LMAX), bool),
        labels_raw=np.zeros((n, LMAX, 20), np.float32),
        label_mask=np.zeros((n, LMAX, 20), bool),
    )
    for i, r in enumerate(recs):
        L = r["L"]
        out["seq_indices"][i, :L] = [AA.find(c) for c in r["seq"]]
        out["backbone_coords"][i, :L] = r["X"]
        out["residue_valid_mask"][i, :L] = r["vm"]
        out["labels_raw"][i, :L] = r["y"]
        out["label_mask"][i, :L] = r["m"]

    np.savez_compressed(OUT, **out)
    v = out["labels_raw"][out["label_mask"]]
    print(f"wrote {OUT}  LMAX={LMAX}  n={n}")
    print(f"labels: median {np.median(v):+.3f} mean {v.mean():+.3f} std {v.std():.3f} "
          f"frac<0 {(v < 0).mean():.2f}   (Tsuboyama: -0.497 / -0.743 / 1.002 / 0.77)")
    print(f"variants per chain: min {min(r['n'] for r in recs)} "
          f"median {int(np.median([r['n'] for r in recs]))} max {max(r['n'] for r in recs)}")


if __name__ == "__main__":
    main()
