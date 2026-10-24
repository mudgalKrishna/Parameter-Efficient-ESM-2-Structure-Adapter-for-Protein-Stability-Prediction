"""Retrain ThermoMPNN's architecture on train_s669 and score it on S669 — the fair transfer baseline.

Mirrors thermo_train.py (frozen pretrained ProteinMPNN + LightAttention + MLP head, their
config.yaml), but:
  * trains on mega `train_s669` -- the train split with every S669 homologue removed,
  * selects on mega val (contains no S669 homologue),
  * scores the external S669 benchmark, pooled over all variants, the way S669 is scored.

Both sides of the comparison therefore see the same training proteins, the same selection set,
and the same converted S669 tensors, so the only difference is the model.

  python s669_thermo.py --seed 0
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
from scipy.stats import pearsonr, spearmanr

THERMO = Path("/marimo/work/ext/ThermoMPNN")
sys.path.insert(0, str(THERMO))
from protein_mpnn_utils import parse_PDB, tied_featurize  # noqa: E402
from transfer_model import TransferModel  # noqa: E402

BUNDLE = "/marimo/training_bundle_copy.zip"
PDB_DIR = Path("/marimo/work/pdbs")
S669_PDB = Path("/marimo/data_s669/pdbs")
S669_CHAIN_DIR = Path("/marimo/work/pdbs_s669")
SPLITS_PKL = THERMO / "dataset_splits/mega_splits.pkl"
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_IDX = {a: i for i, a in enumerate(ALPHABET)}
AA = "ACDEFGHIKLMNPQRSTVWY"

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--patience", type=int, default=15)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--batch", type=int, default=2048)
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

# ---------------------------------------------------------------- megascale side
z = zipfile.ZipFile(BUNDLE)
var = pd.concat([pd.read_parquet(io.BytesIO(z.read(f"variants/{sp}.parquet"))).assign(_split=sp)
                 for sp in ("train", "val", "test")], ignore_index=True)
var = var[var["wt_aa"].isin(AA_IDX) & var["mut_aa"].isin(AA_IDX)].copy()
PDB_DIR.mkdir(parents=True, exist_ok=True)
for pid in set(var["protein_id"]):
    t = PDB_DIR / pid
    if not t.exists():
        t.write_bytes(z.read(f"pdbs/{pid}"))

ms = pickle.load(open(SPLITS_PKL, "rb"))
base = {}
for q in set(var["protein_id"]):
    base.setdefault(q.replace(".pdb", ""), q)
part = {name: [base[str(n).replace(".pdb", "")] for n in ms[key] if str(n).replace(".pdb", "") in base]
        for name, key in (("train", "train_s669"), ("val", "val"))}
print({k: len(v) for k, v in part.items()}, flush=True)

# ---------------------------------------------------------------- S669 side
# write one single-chain PDB per S669 entry so parse_PDB sees exactly the scored chain
S669_CHAIN_DIR.mkdir(parents=True, exist_ok=True)
s6 = pd.read_csv("/marimo/data_s669/s669_clean_dir.csv")
for tag in s6.PDB.unique():
    pid, chain = tag[:-1], tag[-1]
    dst = S669_CHAIN_DIR / f"{tag}.pdb"
    if dst.exists():
        continue
    keep = []
    for line in open(S669_PDB / f"{pid}.pdb"):
        if line.startswith(("ATOM", "HETATM")) and line[21] == chain and line[16] in (" ", "A"):
            keep.append(line)
        elif line.startswith("TER") and len(line) > 21 and line[21] == chain:
            keep.append(line)
    dst.write_text("".join(keep) + "END\n")
print("wrote", len(list(S669_CHAIN_DIR.glob('*.pdb'))), "single-chain S669 pdbs", flush=True)

model = TransferModel(CFG).to(dev)
model.prot_mpnn.eval()


def feats_for(pdb_path, seq_override=None):
    pdb = parse_PDB(str(pdb_path))
    if seq_override is not None:
        pdb[0]["seq"] = seq_override
    f = tied_featurize([pdb[0]], dev, None, None, None, None, None, None, ca_only=False)
    X, S, mask, chain_M, chain_enc, residue_idx = f[0], f[1], f[2], f[4], f[5], f[12]
    all_hid, emb, _ = model.prot_mpnn(X, S, mask, chain_M, residue_idx, chain_enc, None)
    hid = torch.cat(all_hid[: model.num_final_layers], -1)[0]
    return torch.cat([hid, emb[0]], -1).float().cpu(), pdb[0]["seq"]


feat_cache = {}
need = set(part["train"]) | set(part["val"])
with torch.no_grad():
    for pid in sorted(need):
        g = var[var["protein_id"] == pid]
        feat_cache[pid], _ = feats_for(PDB_DIR / pid, g["wt_seq"].iloc[0])
print("cached megascale features for", len(feat_cache), "proteins", flush=True)

# S669 variants, mapped by author residue numbering exactly as in s669_build.py
s6_rows, s6_feats, dropped = [], {}, 0
with torch.no_grad():
    for tag, g in s6.groupby("PDB"):
        path = S669_CHAIN_DIR / f"{tag}.pdb"
        fv, seq = feats_for(path)
        resn = []
        seen = set()
        for line in open(path):
            if line.startswith("ATOM") and line[17:20].strip() and line[16] in (" ", "A"):
                k = line[22:27]
                if k not in seen:
                    seen.add(k)
                    resn.append(line[22:26].strip())
        s6_feats[tag] = fv
        for r in g.itertuples():
            wt, mut, pos = r.MUT[0], r.MUT[-1], r.MUT[1:-1]
            if pos not in resn:
                dropped += 1
                continue
            i = resn.index(pos)
            if i >= len(seq) or seq[i] != wt or wt not in AA_IDX or mut not in AA_IDX:
                dropped += 1
                continue
            s6_rows.append({"tag": tag, "idx": i, "wt": wt, "mut": mut, "y": float(r.DDG)})
print(f"S669: {len(s6_rows)} variants usable, {dropped} dropped", flush=True)


def pack(pids):
    g = var[var["protein_id"].isin(set(pids))]
    feats = torch.cat([feat_cache[r.protein_id][r.pos - 1].unsqueeze(0) for r in g.itertuples()]).to(dev)
    wt = torch.tensor([AA_IDX[a] for a in g["wt_aa"]], device=dev)
    mut = torch.tensor([AA_IDX[a] for a in g["mut_aa"]], device=dev)
    y = torch.tensor(g["label_raw"].to_numpy(np.float32), device=dev)
    return feats, wt, mut, y, g["protein_id"].to_numpy()


def pack_s669():
    feats = torch.cat([s6_feats[r["tag"]][r["idx"]].unsqueeze(0) for r in s6_rows]).to(dev)
    wt = torch.tensor([AA_IDX[r["wt"]] for r in s6_rows], device=dev)
    mut = torch.tensor([AA_IDX[r["mut"]] for r in s6_rows], device=dev)
    y = torch.tensor(np.array([r["y"] for r in s6_rows], np.float32), device=dev)
    return feats, wt, mut, y, np.array([r["tag"] for r in s6_rows])


data = {"train": pack(part["train"]), "val": pack(part["val"]), "s669": pack_s669()}
print({k: len(v[3]) for k, v in data.items()}, "variants", flush=True)

head = [q for n, q in model.named_parameters() if q.requires_grad]
print("trainable head params", sum(q.numel() for q in head), flush=True)
opt = torch.optim.AdamW(head, lr=args.lr)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)


def forward(feats):
    sel = feats
    if model.lightattn:
        sel = model.light_attention(sel.unsqueeze(-1), None).reshape(sel.shape)
    return model.ddg_out(model.both_out(sel).unsqueeze(-1)).squeeze(-1)


def pred_ddg(feats, wt, mut, bs=16384):
    outs = []
    for s in range(0, len(wt), bs):
        lg = forward(feats[s:s + bs])
        r = torch.arange(lg.shape[0], device=dev)
        outs.append(lg[r, mut[s:s + bs]] - lg[r, wt[s:s + bs]])
    return torch.cat(outs)


@torch.no_grad()
def median_spearman(split):
    feats, wt, mut, y, pids = data[split]
    model.eval()
    pr = pred_ddg(feats, wt, mut).float().cpu().numpy()
    yy = y.cpu().numpy()
    rs = []
    for q in np.unique(pids):
        s = pids == q
        if s.sum() >= 3:
            r = spearmanr(pr[s], yy[s]).correlation
            if np.isfinite(r):
                rs.append(r)
    return float(np.median(rs))


@torch.no_grad()
def pooled_s669():
    feats, wt, mut, y, pids = data["s669"]
    model.eval()
    pr = pred_ddg(feats, wt, mut).float().cpu().numpy()
    yy = y.cpu().numpy()
    return float(spearmanr(pr, yy).correlation), float(pearsonr(pr, yy)[0]), pr, yy


best, best_ep, bad, best_state = -1, -1, 0, None
t0 = time.time()
ftr, wtr, mtr, ytr, _ = data["train"]
n = len(ytr)
for ep in range(args.epochs):
    model.train()
    order = torch.randperm(n, device=dev)
    for s in range(0, n, args.batch):
        i = order[s:s + args.batch]
        loss = F.mse_loss(pred_ddg(ftr[i], wtr[i], mtr[i]), ytr[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    vs = median_spearman("val")
    sched.step(vs)
    if vs > best:
        best, best_ep, bad = vs, ep, 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if v.requires_grad or True}
    else:
        bad += 1
        if bad >= args.patience:
            break
    if ep % 10 == 0:
        print(f"ep {ep} val {vs:.4f} best {best:.4f} t {time.time() - t0:.0f}s", flush=True)

if best_state is not None:
    model.load_state_dict(best_state)
sp, pe, pr, yy = pooled_s669()
rng = np.random.default_rng(0)
idx = rng.integers(0, len(pr), (800, len(pr)))
bs = np.array([spearmanr(pr[i], yy[i]).correlation for i in idx])
lo, hi = np.nanquantile(bs, [0.025, 0.975])

res = {"seed": args.seed, "best_val_median_spearman": round(best, 4), "best_epoch": best_ep,
       "n_train_proteins": len(part["train"]), "n_s669_variants": int(len(pr)),
       "s669_pooled_spearman": round(sp, 4), "s669_pooled_pearson": round(pe, 4),
       "s669_ci95_spearman": [round(float(lo), 4), round(float(hi), 4)],
       "s669_rmse": round(float(np.sqrt(((pr - yy) ** 2).mean())), 4),
       "elapsed_s": round(time.time() - t0)}
(OUT / f"thermo_s669_seed{args.seed}.json").write_text(json.dumps(res, indent=1))
# variant identities are saved so this can be intersected with the fusion model's predictions:
# the two PDB-parsing paths place slightly different variant subsets, so the comparison must be
# made on the variants both sides actually scored
np.savez_compressed(OUT / f"thermo_s669_seed{args.seed}.npz",
                    pred=pr.astype(np.float32), truth=yy.astype(np.float32),
                    tags=data["s669"][4],
                    idx=np.array([r["idx"] for r in s6_rows], np.int64),
                    wt=np.array([r["wt"] for r in s6_rows]),
                    mut=np.array([r["mut"] for r in s6_rows]))
print(json.dumps(res), flush=True)
