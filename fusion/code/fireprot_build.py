"""Convert FireProtDB into the same tensor format as the Tsuboyama training bundle.

Only the 89 proteins in FireProt's `homologue-free` split are used: ThermoMPNN's own mmseqs
map (dataset_splits/megascale_vs_fireprot_homologues.m8) shows the other 11 are homologous to
megascale proteins, so excluding them keeps this data safe for every CV fold *and* the official
test split, with no per-fold bookkeeping.

Two corrections are applied, both verified against the data:
  * sign — FireProt reports ddG with POSITIVE = destabilizing (median +0.68, corr(ddG, ASA)
    = -0.34); Tsuboyama's ddG_ML has NEGATIVE = destabilizing. FireProt ddG is negated.
  * scale — FireProt is a different assay with a wider spread, so after negation its labels are
    rescaled to the Tsuboyama training std (mean is left alone; it carries real signal).

Output: $ESM2_WORK/fireprot_tensors.npz, same keys as tensors/*_tensors.npz in the bundle.
"""
import io
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import paths

AA = "ACDEFGHIKLMNPQRSTVWY"
AA3 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", "HIS": "H",
    "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q",
    "ARG": "R", "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    # common modified residues seen in these PDBs, mapped to their parent
    "MSE": "M", "HSD": "H", "HSE": "H", "HSP": "H", "SEC": "C", "PYL": "K",
}
BB = ("N", "CA", "C", "O")

FP = paths.FIREPROT_DIR
BUNDLE = paths.BUNDLE
OUT = paths.FP_TENSORS


def all_chains(pdb_path):
    """Chain IDs that carry standard-residue ATOM records, in file order."""
    seen = []
    for line in open(pdb_path):
        if line.startswith(("ATOM", "HETATM")) and line[17:20].strip() in AA3:
            c = line[21]
            if c not in seen:
                seen.append(c)
    return seen


def parse_backbone(pdb_path, want_chain=None):
    """Return (seq, coords[L,4,3], valid[L]) for the first (or named) chain of a PDB.

    Residues are taken in file order. A residue missing any backbone atom is kept as a
    position but flagged invalid, so numbering stays aligned with the sequence.
    """
    res = {}
    order = []
    chain_seen = []
    for line in open(pdb_path):
        if not line.startswith("ATOM") and not line.startswith("HETATM"):
            continue
        name = line[12:16].strip()
        resn = line[17:20].strip()
        chain = line[21]
        if resn not in AA3:
            continue
        if chain not in chain_seen:
            chain_seen.append(chain)
        if want_chain is not None and chain != want_chain:
            continue
        if want_chain is None and chain != chain_seen[0]:
            continue
        key = (chain, line[22:27])  # resSeq + insertion code
        if key not in res:
            res[key] = {"aa": AA3[resn], "xyz": {}}
            order.append(key)
        if name in BB and name not in res[key]["xyz"]:
            res[key]["xyz"][name] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    seq, co, vm = [], [], []
    for key in order:
        r = res[key]
        seq.append(r["aa"])
        have = all(a in r["xyz"] for a in BB)
        co.append([r["xyz"][a] if a in r["xyz"] else (0.0, 0.0, 0.0) for a in BB])
        vm.append(have)
    return "".join(seq), np.asarray(co, dtype=np.float32), np.asarray(vm, dtype=bool)


def ref_to_pdb_map(pdb_seq, ref_seq):
    """Map each ref_seq index to a pdb_seq index, tolerating gaps on either side.

    FireProt's `pdb_sequence` is the full construct while the deposited structure often has
    disordered termini or internal loops missing, so a single constant offset is not enough
    (it drops ~9 proteins). difflib's matching blocks give a gapped correspondence instead.
    """
    import difflib

    sm = difflib.SequenceMatcher(a=ref_seq, b=pdb_seq, autojunk=False)
    mapping = {}
    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            mapping[i + k] = j + k
    if len(mapping) < 0.6 * len(ref_seq):
        return None
    return mapping


def main():
    splits = pickle.load(open(FP / "csvs" / "fireprot_splits.pkl", "rb"))
    keep = {str(x) for x in splits["homologue-free"]}

    df = pd.read_csv(FP / "csvs" / "4_fireprotDB_bestpH.csv").dropna(subset=["ddG"])
    df = df[df.pdb_id_corrected.isin(keep)]
    print(f"homologue-free: {df.pdb_id_corrected.nunique()} proteins, {len(df)} variants")

    # Tsuboyama training std, for the scale match
    z = zipfile.ZipFile(BUNDLE)
    t = np.load(io.BytesIO(z.read("tensors/train_tensors.npz")))
    ts = t["labels_raw"][t["label_mask"]]
    ts_std = float(ts.std())
    fp_std = float(df.ddG.std())
    scale = ts_std / fp_std
    print(f"label std: tsuboyama {ts_std:.3f} | fireprot {fp_std:.3f} -> scale x{scale:.3f}")

    recs, skipped = [], []
    for pid, g in df.groupby("pdb_id_corrected"):
        ref = g.pdb_sequence.iloc[0]
        chain = str(g.chain.iloc[0]) if "chain" in g.columns and pd.notna(g.chain.iloc[0]) else None
        path = FP / "pdbs" / f"{pid}.pdb"
        if not path.exists():
            skipped.append((pid, "no pdb"))
            continue
        # try the annotated chain, then every chain present, keeping the best match to `ref`
        cands = [chain] if chain and len(chain) == 1 else []
        cands += [None] + [c for c in all_chains(path) if c not in cands]
        best = None
        for c in cands:
            seq, co, vm = parse_backbone(path, c)
            if len(seq) == 0:
                continue
            mp = {i: i for i in range(len(ref))} if seq == ref else ref_to_pdb_map(seq, ref)
            if mp is None:
                continue
            score = sum(1 for i, j in mp.items() if j < len(seq) and seq[j] == ref[i])
            if best is None or score > best[0]:
                best = (score, seq, co, vm, mp)
        if best is None:
            skipped.append((pid, f"no usable chain/alignment (ref {len(ref)})"))
            continue
        _, seq, co, vm, mp = best

        L = len(seq)
        y = np.zeros((L, 20), dtype=np.float32)
        m = np.zeros((L, 20), dtype=bool)
        placed = 0
        for r in g.itertuples():
            i = mp.get(int(r.pdb_position))
            if i is None or not (0 <= i < L):
                continue
            if seq[i] != r.wild_type:
                continue                      # wild-type disagreement -> drop the variant
            a = AA.find(r.mutation)
            if a < 0 or not vm[i]:
                continue
            y[i, a] = -float(r.ddG) * scale   # negate to Tsuboyama's convention, then rescale
            m[i, a] = True
            placed += 1
        if placed < 1:
            skipped.append((pid, "no variants placed"))
            continue
        recs.append(dict(pid=f"FP_{pid}", seq=seq, L=L, X=co, vm=vm, y=y, m=m, n=placed))

    print(f"built {len(recs)} proteins, {sum(r['n'] for r in recs)} variants placed")
    if skipped:
        print(f"skipped {len(skipped)}:")
        for s in skipped:
            print("   ", s)

    LMAX = max(r["L"] for r in recs)
    n = len(recs)
    out = dict(
        protein_ids=np.array([r["pid"] for r in recs]),
        lengths=np.array([r["L"] for r in recs], dtype=np.int64),
        seq_indices=np.full((n, LMAX), -1, dtype=np.int64),
        backbone_coords=np.zeros((n, LMAX, 4, 3), dtype=np.float32),
        residue_valid_mask=np.zeros((n, LMAX), dtype=bool),
        labels_raw=np.zeros((n, LMAX, 20), dtype=np.float32),
        label_mask=np.zeros((n, LMAX, 20), dtype=bool),
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
    print(f"label check: median {np.median(v):+.3f} mean {v.mean():+.3f} std {v.std():.3f} "
          f"| frac<0 {(v < 0).mean():.2f}  (should match Tsuboyama: negative = destabilizing)")


if __name__ == "__main__":
    main()
