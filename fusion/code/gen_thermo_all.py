"""Released ThermoMPNN predictions for EVERY variant of all 298 bundle proteins.

The original gen_thermompnn_preds.py covered the official test split densely but only
25 sampled train variants per protein, which is too sparse to compare on the CV folds.
"""
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

THERMO = Path(os.environ.get("THERMOMPNN_DIR", "/marimo/work/ext/ThermoMPNN"))
BUNDLE = Path(os.environ.get("ESM2_BUNDLE", "/marimo/training_bundle_copy.zip"))
PDB_DIR = Path(os.environ.get("ESM2_PDB_DIR", "/marimo/work/pdbs"))
OUT = Path(os.environ.get("OUT", "/marimo/work/thermompnn_all_variants.parquet"))

sys.path.insert(0, str(THERMO))
from protein_mpnn_utils import parse_PDB, tied_featurize  # noqa: E402
from transfer_model import TransferModel  # noqa: E402

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_IDX = {a: i for i, a in enumerate(ALPHABET)}


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def load_model(device):
    cfg = AttrDict({
        "model": AttrDict({"hidden_dims": [64, 32], "subtract_mut": True, "num_final_layers": 2,
                           "freeze_weights": True, "load_pretrained": True, "lightattn": True}),
        "platform": AttrDict({"thermompnn_dir": str(THERMO)}),
    })
    m = TransferModel(cfg)
    ck = torch.load(THERMO / "models" / "thermoMPNN_default.pt", map_location="cpu", weights_only=False)
    m.load_state_dict({k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}, strict=True)
    return m.to(device).eval()


def predict_protein(model, device, pdb_path, wt_seq, df):
    pdb = parse_PDB(str(pdb_path))
    pdb[0]["seq"] = wt_seq
    f = tied_featurize([pdb[0]], device, None, None, None, None, None, None, ca_only=False)
    X, S, mask, chain_M, chain_enc, residue_idx = f[0], f[1], f[2], f[4], f[5], f[12]
    pos0 = torch.tensor(df["pos"].to_numpy() - 1, dtype=torch.long, device=device)
    wt_i = torch.tensor([AA_IDX[a] for a in df["wt_aa"]], dtype=torch.long, device=device)
    mut_i = torch.tensor([AA_IDX[a] for a in df["mut_aa"]], dtype=torch.long, device=device)
    with torch.no_grad():
        all_hid, emb, _ = model.prot_mpnn(X, S, mask, chain_M, residue_idx, chain_enc, None)
        hid = torch.cat(all_hid[: model.num_final_layers], -1)[0]
        sel = torch.cat([hid, emb[0]], -1)[pos0]
        if model.lightattn:
            sel = model.light_attention(sel.unsqueeze(-1), mask).reshape(sel.shape)
        logits = model.ddg_out(model.both_out(sel).unsqueeze(-1)).squeeze(-1)
        rows = torch.arange(len(pos0), device=device)
        ddg = logits[rows, mut_i] - logits[rows, wt_i]
    return ddg.cpu().numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    PDB_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    with zipfile.ZipFile(BUNDLE) as z:
        for sp in ("train", "val", "test"):
            parts.append(pd.read_parquet(io.BytesIO(z.read(f"variants/{sp}.parquet"))).assign(_split=sp))
        work = pd.concat(parts, ignore_index=True)
        for pid in set(work["protein_id"]):
            tgt = PDB_DIR / pid
            if not tgt.exists():
                tgt.write_bytes(z.read(f"pdbs/{pid}"))
    work = work[work["mut_aa"].isin(AA_IDX) & work["wt_aa"].isin(AA_IDX)].copy()
    work["_pred"] = np.nan
    model = load_model(device)
    groups = list(work.groupby("protein_id", sort=False))
    print(f"device={device} proteins={len(groups)} variants={len(work)}", flush=True)
    for gi, (pid, df) in enumerate(groups):
        work.loc[df.index, "_pred"] = predict_protein(model, device, PDB_DIR / pid, df["wt_seq"].iloc[0], df)
        if (gi + 1) % 40 == 0 or gi + 1 == len(groups):
            print(f"[{gi+1}/{len(groups)}] {pid} elapsed={time.time()-t0:.0f}s", flush=True)
    out = (work[["protein_id", "pos", "wt_aa", "mut_aa", "_split", "_pred"]]
           .rename(columns={"_split": "split", "_pred": "ddG_pred"}).dropna(subset=["ddG_pred"]))
    out.to_parquet(OUT, index=False)
    from scipy.stats import pearsonr
    ov = out.merge(work[["protein_id", "pos", "mut_aa", "ddG_ML"]], on=["protein_id", "pos", "mut_aa"], how="left").dropna()
    summary = {"rows": int(len(out)), "proteins": int(out["protein_id"].nunique()),
               "by_split": out.groupby("split").size().to_dict(),
               "pearson_vs_ddG_ML_all": float(pearsonr(ov["ddG_pred"], ov["ddG_ML"])[0]),
               "elapsed_seconds": round(time.time() - t0, 1)}
    Path("/marimo/work/thermompnn_all_variants_summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
