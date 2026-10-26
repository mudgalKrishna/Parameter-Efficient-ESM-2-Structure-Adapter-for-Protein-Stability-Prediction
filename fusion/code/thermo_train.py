"""Retrain ThermoMPNN's own architecture under the same protocol, for a fair comparison.

Released ThermoMPNN was trained on the mega-split train proteins, so scoring it on CV folds
(which repartition all 298 proteins) evaluates it partly on its own training data. This script
trains their architecture — frozen pretrained ProteinMPNN + LightAttention + MLP head, as in
transfer_model.TransferModel with their config.yaml — on the same fold-train proteins we use,
selects on fold validation, and scores the fold test set once.

Labels are our bundle's ddG_ML (negative = destabilising), so no sign flip is involved.

  python thermo_train.py --split mega --seed 0
"""
import argparse
import io
import json
import pickle
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

THERMO = Path("/marimo/work/ext/ThermoMPNN")
sys.path.insert(0, str(THERMO))
from protein_mpnn_utils import parse_PDB, tied_featurize  # noqa: E402
from transfer_model import TransferModel  # noqa: E402

BUNDLE = "/marimo/training_bundle_copy.zip"
PDB_DIR = Path("/marimo/work/pdbs")
SPLITS_PKL = THERMO / "dataset_splits/mega_splits.pkl"
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_IDX = {a: i for i, a in enumerate(ALPHABET)}

p = argparse.ArgumentParser()
p.add_argument("--split", default="mega")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--patience", type=int, default=15)
p.add_argument("--lr", type=float, default=1e-3)       # their config.yaml learn_rate
p.add_argument("--batch", type=int, default=2048)      # variants per step
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = "cuda"
OUT = Path("/marimo/work/runs_thermo")
OUT.mkdir(parents=True, exist_ok=True)


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


CFG = AttrDict({
    "model": AttrDict({"hidden_dims": [64, 32], "subtract_mut": True, "num_final_layers": 2,
                       "freeze_weights": True, "load_pretrained": True, "lightattn": True}),
    "platform": AttrDict({"thermompnn_dir": str(THERMO)}),
})

# ---------------------------------------------------------------- data
z = zipfile.ZipFile(BUNDLE)
var = pd.concat([pd.read_parquet(io.BytesIO(z.read(f"variants/{sp}.parquet"))).assign(_split=sp)
                 for sp in ("train", "val", "test")], ignore_index=True)
var = var[var["wt_aa"].isin(AA_IDX) & var["mut_aa"].isin(AA_IDX)].copy()
PDB_DIR.mkdir(parents=True, exist_ok=True)
for pid in set(var["protein_id"]):
    t = PDB_DIR / pid
    if not t.exists():
        t.write_bytes(z.read(f"pdbs/{pid}"))

if args.split == "mega":
    part = {sp: sorted(set(var.loc[var["_split"] == sp, "protein_id"])) for sp in ("train", "val", "test")}
else:
    k = int(args.split[2:])
    ms = pickle.load(open(SPLITS_PKL, "rb"))
    base = {}
    for q in set(var["protein_id"]):
        base.setdefault(q.replace(".pdb", ""), q)
    part = {name: [base[str(n).replace(".pdb", "")] for n in ms[key] if str(n).replace(".pdb", "") in base]
            for name, key in (("train", f"cv_train_{k}"), ("val", f"cv_val_{k}"), ("test", f"cv_test_{k}"))}
print({k2: len(v) for k2, v in part.items()}, flush=True)

model = TransferModel(CFG).to(dev)
model.prot_mpnn.eval()

# cache frozen ProteinMPNN features once per protein (the backbone never trains in their setup)
feat_cache = {}
with torch.no_grad():
    for pid in sorted(set(var["protein_id"])):
        g = var[var["protein_id"] == pid]
        pdb = parse_PDB(str(PDB_DIR / pid))
        pdb[0]["seq"] = g["wt_seq"].iloc[0]
        f = tied_featurize([pdb[0]], dev, None, None, None, None, None, None, ca_only=False)
        X, S, mask, chain_M, chain_enc, residue_idx = f[0], f[1], f[2], f[4], f[5], f[12]
        all_hid, emb, _ = model.prot_mpnn(X, S, mask, chain_M, residue_idx, chain_enc, None)
        hid = torch.cat(all_hid[: model.num_final_layers], -1)[0]
        feat_cache[pid] = torch.cat([hid, emb[0]], -1).float().cpu()
print("cached features for", len(feat_cache), "proteins", flush=True)


def pack(pids):
    pids = set(pids)
    g = var[var["protein_id"].isin(pids)]
    feats = torch.cat([feat_cache[r.protein_id][r.pos - 1].unsqueeze(0) for r in g.itertuples()]).to(dev)
    wt = torch.tensor([AA_IDX[a] for a in g["wt_aa"]], device=dev)
    mut = torch.tensor([AA_IDX[a] for a in g["mut_aa"]], device=dev)
    y = torch.tensor(g["label_raw"].to_numpy(np.float32), device=dev)
    return feats, wt, mut, y, g["protein_id"].to_numpy()


data = {k2: pack(v) for k2, v in part.items()}
print({k2: len(v[3]) for k2, v in data.items()}, "variants", flush=True)

head = [q for n, q in model.named_parameters() if q.requires_grad]
print("trainable head params", sum(q.numel() for q in head), flush=True)
opt = torch.optim.AdamW(head, lr=args.lr)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)


def forward(feats):
    sel = feats
    if model.lightattn:
        sel = model.light_attention(sel.unsqueeze(-1), None).reshape(sel.shape)
    logits = model.ddg_out(model.both_out(sel).unsqueeze(-1)).squeeze(-1)
    return logits


def pred_ddg(feats, wt, mut, bs=16384):
    outs = []
    for s in range(0, len(wt), bs):
        lg = forward(feats[s:s + bs])
        r = torch.arange(lg.shape[0], device=dev)
        outs.append(lg[r, mut[s:s + bs]] - lg[r, wt[s:s + bs]])
    return torch.cat(outs)


def median_spearman(split):
    feats, wt, mut, y, pids = data[split]
    model.eval()
    with torch.no_grad():
        pr = pred_ddg(feats, wt, mut).cpu().numpy()
    yt = y.cpu().numpy()
    rs = []
    for pid in np.unique(pids):
        m = pids == pid
        if m.sum() >= 3 and np.std(yt[m]) > 1e-9 and np.std(pr[m]) > 1e-9:
            rs.append(spearmanr(pr[m], yt[m]).correlation)
    return float(np.median(rs)), {str(p): float(r) for p, r in zip(np.unique(pids), rs)} if len(rs) == len(np.unique(pids)) else {}


ftr, wtr, mtr, ytr, _ = data["train"]
n = len(ytr)
best, best_ep, bad, best_test_per_prot = -1, -1, 0, {}
t0 = time.time()
for ep in range(args.epochs):
    model.train()
    model.prot_mpnn.eval()
    perm = torch.randperm(n, device=dev)
    tot = 0.0
    for s in range(0, n, args.batch):
        idx = perm[s:s + args.batch]
        pr = pred_ddg(ftr[idx], wtr[idx], mtr[idx])
        loss = F.mse_loss(pr, ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tot += loss.item() * len(idx)
    vs, _ = median_spearman("val")
    sched.step(vs)
    if ep % 5 == 0 or ep == args.epochs - 1:
        print(json.dumps({"epoch": ep, "train_mse": tot / n, "val_median_spearman": round(vs, 4),
                          "t": round(time.time() - t0)}), flush=True)
    if vs > best:
        best, best_ep, bad = vs, ep, 0
        ts, per_prot = median_spearman("test")
        best_test_per_prot = per_prot
        best_test = ts
    else:
        bad += 1
        if bad >= args.patience:
            break

res = {"arch": "ThermoMPNN (TransferModel, their config) retrained under our protocol",
       "split": args.split, "seed": args.seed, "best_val_median_spearman": round(best, 4),
       "best_epoch": best_ep, "test_median_spearman": round(best_test, 4),
       "n_train_proteins": len(part["train"]), "test_per_protein": best_test_per_prot,
       "elapsed_seconds": round(time.time() - t0, 1)}
(OUT / f"thermo_{args.split}_seed{args.seed}.json").write_text(json.dumps(res, indent=1))
print("DONE", json.dumps({k2: v for k2, v in res.items() if k2 != "test_per_protein"}), flush=True)
