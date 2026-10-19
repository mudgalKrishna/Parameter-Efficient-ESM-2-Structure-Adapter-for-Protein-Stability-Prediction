"""Vectorized ThermoMPNN inference over the Tsuboyama training bundle.

Produces ddG_pred for every test variant plus 25 sampled train variants per
protein (used only for label-sign validation inside the pipeline).
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

THERMO = Path(os.environ.get("THERMOMPNN_DIR", "assets/ThermoMPNN"))
BUNDLE = Path(os.environ.get("ESM2_BUNDLE", "training_bundle.zip"))
PDB_DIR = Path(os.environ.get("ESM2_PDB_DIR", "pdbs"))
OUT = Path(os.environ.get("ESM2_THERMO_PREDS", "thermompnn_all_preds.parquet"))
TRAIN_PER_PROTEIN = 25

sys.path.insert(0, str(THERMO))
from protein_mpnn_utils import parse_PDB, tied_featurize  # noqa: E402
from transfer_model import TransferModel  # noqa: E402

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_IDX = {a: i for i, a in enumerate(ALPHABET)}


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def load_model(device: str) -> TransferModel:
    cfg = AttrDict({
        "model": AttrDict({
            "hidden_dims": [64, 32],
            "subtract_mut": True,
            "num_final_layers": 2,
            "freeze_weights": True,
            "load_pretrained": True,
            "lightattn": True,
        }),
        "platform": AttrDict({"thermompnn_dir": str(THERMO)}),
    })
    model = TransferModel(cfg)
    ck = torch.load(THERMO / "models" / "thermoMPNN_default.pt", map_location="cpu", weights_only=False)
    sd = {k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model


def predict_protein(model, device, pdb_path: Path, wt_seq: str, df: pd.DataFrame):
    pdb = parse_PDB(str(pdb_path))
    pdb[0]["seq"] = wt_seq
    feats = tied_featurize(
        [pdb[0]], device, None, None, None, None, None, None, ca_only=False
    )
    X, S, mask, chain_M, chain_encoding_all = feats[0], feats[1], feats[2], feats[4], feats[5]
    residue_idx = feats[12]

    pos0 = torch.tensor((df["pos"].to_numpy() - 1), dtype=torch.long, device=device)
    wt_i = torch.tensor([AA_IDX[a] for a in df["wt_aa"]], dtype=torch.long, device=device)
    mut_i = torch.tensor([AA_IDX[a] for a in df["mut_aa"]], dtype=torch.long, device=device)

    with torch.no_grad():
        all_hid, emb, _ = model.prot_mpnn(
            X, S, mask, chain_M, residue_idx, chain_encoding_all, None
        )
        hid = torch.cat(all_hid[: model.num_final_layers], -1)[0]
        feat = torch.cat([hid, emb[0]], -1)
        sel = feat[pos0]
        if model.lightattn:
            x = sel.unsqueeze(-1)
            sel = model.light_attention(x, mask).reshape(sel.shape)
        logits = model.both_out(sel)
        logits = model.ddg_out(logits.unsqueeze(-1)).squeeze(-1)
        rows = torch.arange(len(pos0), device=device)
        ddg = logits[rows, mut_i] - logits[rows, wt_i]
    return ddg.detach().cpu().numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    t0 = time.time()

    PDB_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BUNDLE) as z:
        test = pd.read_parquet(io.BytesIO(z.read("variants/test.parquet")))
        train = pd.read_parquet(io.BytesIO(z.read("variants/train.parquet")))
        train_s = (
            train.sort_values(["protein_id", "pos", "mut_aa"])
            .groupby("protein_id", sort=False)
            .head(TRAIN_PER_PROTEIN)
            .copy()
        )
        needed = set(test["protein_id"]) | set(train_s["protein_id"])
        for pid in needed:
            name = f"pdbs/{pid}"
            target = PDB_DIR / pid
            if not target.exists():
                target.write_bytes(z.read(name))

    work = pd.concat([test.assign(_split="test"), train_s.assign(_split="train")], ignore_index=True)
    work = work[work["mut_aa"].isin(AA_IDX) & work["wt_aa"].isin(AA_IDX)].copy()
    work["_ddG_pred"] = np.nan

    model = load_model(device)
    print("model loaded", flush=True)

    groups = list(work.groupby("protein_id", sort=False))
    for gi, (pid, df) in enumerate(groups):
        wt_seq = df["wt_seq"].iloc[0]
        ddg = predict_protein(model, device, PDB_DIR / pid, wt_seq, df)
        work.loc[df.index, "_ddG_pred"] = ddg
        if (gi + 1) % 20 == 0 or gi + 1 == len(groups):
            print(f"[{gi+1}/{len(groups)}] {pid} n={len(df)} elapsed={time.time()-t0:.1f}s", flush=True)

    out = work[["protein_id", "pos", "wt_aa", "mut_aa", "_split", "_ddG_pred"]].rename(
        columns={"_split": "split", "_ddG_pred": "ddG_pred"}
    )
    out = out.dropna(subset=["ddG_pred"]).reset_index(drop=True)
    out.to_parquet(OUT, index=False)

    tr = out[out["split"] == "train"].merge(
        train[["protein_id", "pos", "mut_aa", "ddG_ML"]],
        on=["protein_id", "pos", "mut_aa"],
        how="left",
    )
    from scipy.stats import pearsonr
    r = float(pearsonr(tr["ddG_pred"], tr["ddG_ML"])[0])
    summary = {
        "rows": int(len(out)),
        "test_rows": int((out["split"] == "test").sum()),
        "train_rows": int((out["split"] == "train").sum()),
        "proteins": int(out["protein_id"].nunique()),
        "train_overlap_pearson_raw_vs_ddG_ML": r,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    Path("/marimo/work/thermompnn_preds_summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
