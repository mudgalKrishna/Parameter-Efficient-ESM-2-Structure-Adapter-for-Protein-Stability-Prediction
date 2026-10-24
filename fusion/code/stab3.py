"""ESM-2 + official ProteinMPNN fusion for Tsuboyama ddG — round 3.

Round 2's logs showed what the binding constraint is: validation peaked at a mean epoch of 8.4
(max 22) against a 40-epoch cap, so every run early-stopped and overfit after that; 15B scored
below 3B; and +13% more variants changed nothing. There are 215k training variants but only 239
training *proteins*, and variants within a protein are strongly correlated — the effective
sample size is the protein count. So round 3 attacks proteins and regularisation, not capacity
or steps.

Adds over stab2.py:
  --fireprot      88 extra training proteins from FireProtDB (sign-corrected and rescaled by
                  fireprot_build.py). Only FireProt's `homologue-free` split is used, so none of
                  them is homologous to ANY megascale protein — safe for every CV fold and for
                  the official test split, with no per-fold exclusion needed.
  --lora_dropout  dropout on the LoRA branch, and --head_drop on the head: the epoch-8 overfit
                  says regularisation is the lever, not more epochs.
  --softrank      differentiable soft-rank Spearman surrogate, replacing the pairwise hinge.
  length-bucketed batching with per-batch padding — FireProt proteins run to 448 residues
  against Tsuboyama's 72, so a single global pad length would waste ~6x the compute.

FireProt proteins are added to the FIT set only. They never enter val or test, so every
reported number is still measured on Tsuboyama proteins alone and stays comparable to round 2.

One run = one (config, seed, split). Predictions for the split's val and test proteins are
written to disk keyed by protein_id; nothing is scored against the official test split here.
"""
import argparse
import io
import json
import math
import pickle
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, "/marimo/work/ext/ThermoMPNN")
from protein_mpnn_utils import ProteinMPNN  # noqa: E402

BUNDLE = "/marimo/training_bundle_copy.zip"
MPNN_DIR = "/marimo/work/ext/ProteinMPNN/vanilla_model_weights"
SPLITS_PKL = "/marimo/work/ext/ThermoMPNN/dataset_splits/mega_splits.pkl"
AA = "ACDEFGHIKLMNPQRSTVWY"

p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--esm", default="facebook/esm2_t36_3B_UR50D")
p.add_argument("--esm_bf16", type=int, default=0)
p.add_argument("--use_esm", type=int, default=1)
p.add_argument("--use_mpnn", type=int, default=1)
p.add_argument("--mpnn_ver", default="v_48_020")
p.add_argument("--mpnn_lr", type=float, default=0.0)
p.add_argument("--lora_rank", type=int, default=32)
p.add_argument("--lora_lr", type=float, default=2e-4)
p.add_argument("--head_lr", type=float, default=5e-4)
p.add_argument("--wd", type=float, default=0.01)
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--patience", type=int, default=10)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--corr_w", type=float, default=1.0)
p.add_argument("--rank_w", type=float, default=0.0)      # pairwise ranking loss weight
p.add_argument("--aux_w", type=float, default=0.0)       # auxiliary burial loss weight
p.add_argument("--fuse_layers", type=int, default=2)
p.add_argument("--fuse_mode", default="concat", choices=["concat", "xattn"])
p.add_argument("--aug_eps", type=float, default=0.0)
p.add_argument("--tta", type=int, default=0)             # extra noisy passes averaged at predict time
p.add_argument("--grad_ckpt", type=int, default=0)
p.add_argument("--split", default="mega")                # mega | cv0..cv4
p.add_argument("--train_on", default="train", choices=["train", "trainval"])
p.add_argument("--fixed_epochs", type=int, default=0)    # for train_on=trainval: stop here, no val selection
p.add_argument("--fireprot", type=int, default=0)        # add the 88 homologue-free FireProt proteins to the fit set
p.add_argument("--fp_w", type=float, default=1.0)        # loss weight on FireProt proteins
p.add_argument("--lora_dropout", type=float, default=0.0)
p.add_argument("--head_drop", type=float, default=0.1)
p.add_argument("--softrank", type=float, default=0.0)    # weight on the soft-rank Spearman surrogate
p.add_argument("--softrank_tau", type=float, default=0.1)
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = "cuda"
out_dir = Path(f"/marimo/work/runs3/{args.tag}")
out_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- data pool
z = zipfile.ZipFile(BUNDLE)
raw = {}
for sp in ("train", "val", "test"):
    t = np.load(io.BytesIO(z.read(f"tensors/{sp}_tensors.npz")))
    raw[sp] = {k: t[k] for k in t.files}
FP_TENSORS = "/marimo/work/fireprot_tensors.npz"
fp_raw = None
if args.fireprot:
    t = np.load(FP_TENSORS, allow_pickle=True)
    fp_raw = {k: t[k] for k in t.files}
    print(f"fireprot: {len(fp_raw['protein_ids'])} proteins, {int(fp_raw['label_mask'].sum())} variants", flush=True)

LMAX = max(d["seq_indices"].shape[1] for d in raw.values())
if fp_raw is not None:
    LMAX = max(LMAX, fp_raw["seq_indices"].shape[1])


def pad(a, L, val=0):
    if a.shape[1] == L:
        return a
    w = [(0, 0), (0, L - a.shape[1])] + [(0, 0)] * (a.ndim - 2)
    return np.pad(a, w, constant_values=val)


pids, store = [], {}
orig_split = {}
for sp, d in raw.items():
    seq = pad(d["seq_indices"], LMAX, -1)
    co = pad(np.nan_to_num(d["backbone_coords"]), LMAX)
    vm = pad(d["residue_valid_mask"], LMAX, False)
    lr = pad(d["labels_raw"], LMAX)
    lm = pad(d["label_mask"], LMAX, False)
    for i, pid in enumerate(d["protein_ids"]):
        pid = str(pid)
        pids.append(pid)
        orig_split[pid] = sp
        store[pid] = dict(seq=seq[i], L=int(d["lengths"][i]), X=co[i], vm=vm[i].astype(np.float32),
                          y=np.nan_to_num(lr[i]), m=lm[i], w=1.0)

fp_pids = []
if fp_raw is not None:
    d = fp_raw
    seq = pad(d["seq_indices"], LMAX, -1)
    co = pad(np.nan_to_num(d["backbone_coords"]), LMAX)
    vm = pad(d["residue_valid_mask"], LMAX, False)
    lr = pad(d["labels_raw"], LMAX)
    lm = pad(d["label_mask"], LMAX, False)
    for i, pid in enumerate(d["protein_ids"]):
        pid = str(pid)
        fp_pids.append(pid)
        store[pid] = dict(seq=seq[i], L=int(d["lengths"][i]), X=co[i], vm=vm[i].astype(np.float32),
                          y=np.nan_to_num(lr[i]), m=lm[i], w=args.fp_w)

if args.split == "mega":
    part = {sp: [q for q in pids if orig_split[q] == sp] for sp in ("train", "val", "test")}
else:
    k = int(args.split[2:])
    ms = pickle.load(open(SPLITS_PKL, "rb"))
    base = {}
    for q in pids:
        base.setdefault(q.replace(".pdb", ""), q)
    part = {}
    for name, key in (("train", f"cv_train_{k}"), ("val", f"cv_val_{k}"), ("test", f"cv_test_{k}")):
        got = [base[str(n).replace(".pdb", "")] for n in ms[key] if str(n).replace(".pdb", "") in base]
        part[name] = got
    miss = len(ms[f"cv_test_{k}"]) - len(part["test"])
    print(f"split {args.split}: " + str({q: len(v) for q, v in part.items()}) + f" (test names unmatched: {miss})", flush=True)

fit_pids = part["train"] + (part["val"] if args.train_on == "trainval" else []) + fp_pids
print({k2: len(v) for k2, v in part.items()}, "fit:", len(fit_pids),
      f"(+{len(fp_pids)} fireprot)" if fp_pids else "", flush=True)


def batch_of(plist):
    """Pad only to the longest protein in this batch.

    FireProt proteins reach 448 residues against Tsuboyama's 72, so padding every batch to a
    global LMAX would spend most of the compute on padding.
    """
    Lc = max(store[q]["L"] for q in plist)
    seq = torch.as_tensor(np.stack([store[q]["seq"][:Lc] for q in plist]).astype(np.int64))
    L = torch.as_tensor(np.array([store[q]["L"] for q in plist], dtype=np.int64))
    X = torch.as_tensor(np.stack([store[q]["X"][:Lc] for q in plist]))
    vm = torch.as_tensor(np.stack([store[q]["vm"][:Lc] for q in plist]))
    y = torch.as_tensor(np.stack([store[q]["y"][:Lc] for q in plist]))
    m = torch.as_tensor(np.stack([store[q]["m"][:Lc] for q in plist]))
    w = torch.as_tensor(np.array([store[q]["w"] for q in plist], dtype=np.float32))
    return [t.to(dev) for t in (seq, L, X, vm, y, m, w)]


def length_batches(plist, bs):
    """Batches of similar-length proteins, so per-batch padding actually pays off."""
    srt = sorted(plist, key=lambda q: store[q]["L"])
    return [srt[s:s + bs] for s in range(0, len(srt), bs)]


# ---------------------------------------------------------------- model
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        for q in base.parameters():
            q.requires_grad = False
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.s = alpha / r
        self.drop = nn.Dropout(args.lora_dropout) if args.lora_dropout > 0 else None

    def forward(self, x):
        h = self.drop(x) if self.drop is not None else x
        return self.base(x) + F.linear(F.linear(h, self.A.to(x.dtype)), self.B.to(x.dtype)) * self.s


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        d = 512
        self.use_esm, self.use_mpnn = bool(args.use_esm), bool(args.use_mpnn)
        if self.use_esm:
            from transformers import AutoTokenizer, EsmForMaskedLM

            tok = AutoTokenizer.from_pretrained(args.esm)
            self.esm = EsmForMaskedLM.from_pretrained(
                args.esm, torch_dtype=torch.bfloat16 if args.esm_bf16 else torch.float32)
            if args.grad_ckpt:
                self.esm.gradient_checkpointing_enable()
            for q in self.esm.parameters():
                q.requires_grad = False
            n = 0
            for name, mod in list(self.esm.named_modules()):
                for cn, ch in list(mod.named_children()):
                    if isinstance(ch, nn.Linear) and cn in ("query", "key", "value", "dense") and ".layer." in f"{name}.":
                        setattr(mod, cn, LoRALinear(ch, args.lora_rank, 2 * args.lora_rank))
                        n += 1
            print("lora modules", n, flush=True)
            self.aa_tok = torch.tensor([tok.convert_tokens_to_ids(a) for a in AA])
            self.cls_id, self.eos_id, self.pad_id = tok.cls_token_id, tok.eos_token_id, tok.pad_token_id
            self.layer_w = nn.Parameter(torch.zeros(self.esm.config.num_hidden_layers + 1))
            self.esm_proj = nn.Sequential(nn.LayerNorm(self.esm.config.hidden_size),
                                          nn.Linear(self.esm.config.hidden_size, d))
            self.alpha_llr = nn.Parameter(torch.tensor(0.1))
        if self.use_mpnn:
            ck = torch.load(f"{MPNN_DIR}/{args.mpnn_ver}.pt", map_location="cpu", weights_only=False)
            self.mpnn = ProteinMPNN(ca_only=False, num_letters=21, node_features=128, edge_features=128,
                                    hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                                    k_neighbors=ck["num_edges"], augment_eps=0.0)
            self.mpnn.load_state_dict(ck["model_state_dict"])
            for q in self.mpnn.parameters():
                q.requires_grad = args.mpnn_lr > 0
            self.mpnn_proj = nn.Sequential(nn.LayerNorm(128 * 4), nn.Linear(128 * 4, d))

        if args.fuse_mode == "xattn" and self.use_esm and self.use_mpnn:
            self.xattn = nn.MultiheadAttention(d, 8, batch_first=True)
            self.xnorm = nn.LayerNorm(d)
            fuse_in_dim = d
        else:
            fuse_in_dim = d * (int(self.use_esm) + int(self.use_mpnn))
        self.fuse_in = nn.Linear(fuse_in_dim, d)
        if args.fuse_layers > 0:
            enc = nn.TransformerEncoderLayer(d, 8, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
            self.fuse = nn.TransformerEncoder(enc, args.fuse_layers)
        else:
            self.fuse = None
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(),
                                  nn.Dropout(args.head_drop), nn.Linear(256, 20))
        self.aux = nn.Linear(d, 1) if args.aux_w > 0 else None

    def forward(self, seq, L, X, vmask, noise=0.0):
        B, Lm = seq.shape
        pos = torch.arange(Lm, device=seq.device)[None] < L[:, None]
        safe = seq.clamp(0, 19)
        esm_f = mpnn_f = None
        llr = 0.0
        if self.use_esm:
            ids = torch.full((B, Lm + 2), self.pad_id, device=seq.device)
            ids[:, 0] = self.cls_id
            ids[:, 1:Lm + 1] = torch.where(pos, self.aa_tok.to(seq.device)[safe], torch.full_like(safe, self.pad_id))
            ids[torch.arange(B), L + 1] = self.eos_id
            am = (ids != self.pad_id).long()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = self.esm(input_ids=ids, attention_mask=am, output_hidden_states=True)
            hs = torch.stack(o.hidden_states, 0)[:, :, 1:Lm + 1].float()
            w = torch.softmax(self.layer_w, 0)
            esm_f = self.esm_proj((w[:, None, None, None] * hs).sum(0))
            lp = F.log_softmax(o.logits[:, 1:Lm + 1].float()[..., self.aa_tok.to(seq.device)], -1)
            llr = self.alpha_llr * (lp - lp.gather(-1, safe[..., None]))
        if self.use_mpnn:
            if noise > 0:
                X = X + noise * torch.randn_like(X)
            Sm = torch.where(pos, safe, torch.full_like(safe, 20))
            mk = vmask * pos.float()
            ridx = torch.arange(Lm, device=seq.device)[None].expand(B, -1)
            ctx = torch.enable_grad() if args.mpnn_lr > 0 else torch.no_grad()
            with ctx:
                hid, emb, _ = self.mpnn(X, Sm, mk, mk, ridx, torch.ones_like(mk), None,
                                        use_input_decoding_order=True, decoding_order=ridx)
            mpnn_f = self.mpnn_proj(torch.cat(list(hid) + [emb], -1))

        if esm_f is not None and mpnn_f is not None:
            if args.fuse_mode == "xattn":
                a, _ = self.xattn(esm_f, mpnn_f, mpnn_f, key_padding_mask=~pos)
                x = self.fuse_in(self.xnorm(esm_f + a))
            else:
                x = self.fuse_in(torch.cat([esm_f, mpnn_f], -1))
        else:
            x = self.fuse_in(esm_f if esm_f is not None else mpnn_f)
        if self.fuse is not None:
            x = self.fuse(x, src_key_padding_mask=~pos)
        raw_out = self.head(x)
        out = raw_out - raw_out.gather(-1, safe[..., None]) + llr
        return (out, self.aux(x).squeeze(-1)) if self.aux is not None else (out, None)


model = Model().to(dev)
gp = [
    {"params": [q for n, q in model.named_parameters() if q.requires_grad and (n.endswith(".A") or n.endswith(".B"))], "lr": args.lora_lr},
    {"params": [q for n, q in model.named_parameters() if q.requires_grad and n.startswith("mpnn.")], "lr": args.mpnn_lr},
    {"params": [q for n, q in model.named_parameters() if q.requires_grad and not (n.endswith(".A") or n.endswith(".B") or n.startswith("mpnn."))], "lr": args.head_lr},
]
gp = [g for g in gp if g["params"]]
opt = torch.optim.AdamW(gp, weight_decay=args.wd)
nfit = len(fit_pids)
total = args.epochs * math.ceil(nfit / args.batch)
warm = max(1, total // 20)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
print("trainable", sum(q.numel() for q in model.parameters() if q.requires_grad), flush=True)


def burial_target(X, vm):
    """#CA neighbours within 10 A, standardised — a free structural auxiliary signal."""
    ca = X[:, :, 1, :]
    dist = torch.cdist(ca, ca)
    cnt = ((dist < 10.0).float() * vm[:, None, :]).sum(-1) - 1.0
    return (cnt - 8.0) / 5.0


def soft_spearman(a, b, tau):
    """1 - Pearson(soft_rank(a), rank(b)) — a differentiable stand-in for Spearman.

    soft_rank(x)_i = sum_j sigmoid((x_i - x_j) / tau) approaches the true rank as tau -> 0,
    so this optimises the evaluation metric directly instead of a pairwise proxy for it.
    """
    ra = torch.sigmoid((a[:, None] - a[None, :]) / tau).sum(1)
    rb = torch.argsort(torch.argsort(b)).float()
    rac, rbc = ra - ra.mean(), rb - rb.mean()
    return 1 - (rac * rbc).sum() / (rac.norm() * rbc.norm() + 1e-8)


def losses(pred, aux_pred, y, m, X, vm, w=None):
    if w is None:
        w = torch.ones(pred.shape[0], device=pred.device)
    # per-protein weights, broadcast over that protein's labelled variants
    wm = w[:, None, None].expand_as(m)[m]
    se = (pred[m] - y[m]) ** 2
    mse = (se * wm).sum() / wm.sum().clamp(min=1e-8)
    corr = rank = srank = 0.0
    wsum = 0.0
    for i in range(pred.shape[0]):
        mi = m[i]
        k = int(mi.sum())
        if k < 3:
            continue
        wi = float(w[i])
        a, b = pred[i][mi], y[i][mi]
        ac, bc = a - a.mean(), b - b.mean()
        corr = corr + wi * (1 - (ac * bc).sum() / (ac.norm() * bc.norm() + 1e-8))
        if args.rank_w > 0:
            n = min(k, 256)
            i1 = torch.randint(0, k, (n,), device=a.device)
            i2 = torch.randint(0, k, (n,), device=a.device)
            dy = b[i1] - b[i2]
            dp = a[i1] - a[i2]
            keep = dy.abs() > 1e-6
            if keep.any():
                rank = rank + wi * F.softplus(-dp[keep] * torch.sign(dy[keep])).mean()
        if args.softrank > 0:
            n = min(k, 192)                      # the soft rank is O(n^2); 192 keeps it cheap
            sel = torch.randperm(k, device=a.device)[:n]
            srank = srank + wi * soft_spearman(a[sel], b[sel], args.softrank_tau)
        wsum += wi
    wsum = max(wsum, 1e-8)
    out = mse + args.corr_w * corr / wsum
    if args.rank_w > 0:
        out = out + args.rank_w * rank / wsum
    if args.softrank > 0:
        out = out + args.softrank * srank / wsum
    if aux_pred is not None:
        vmb = vm > 0.5
        out = out + args.aux_w * F.mse_loss(aux_pred[vmb], burial_target(X, vm)[vmb])
    return out


@torch.no_grad()
def predict(plist):
    """Predictions in the order of `plist`, padded back out to a common width."""
    model.eval()
    out = {}
    for chunk in length_batches(plist, 16):
        seq, L, X, vm, _, _, _ = batch_of(chunk)
        acc = model(seq, L, X, vm)[0].float()
        for _ in range(args.tta):
            acc = acc + model(seq, L, X, vm, noise=args.aug_eps or 0.1)[0].float()
        arr = (acc / (1 + args.tta)).cpu().numpy()
        for j, q in enumerate(chunk):
            out[q] = arr[j]
    W = max(v.shape[0] for v in out.values())
    stacked = np.zeros((len(plist), W, 20), dtype=np.float32)
    for i, q in enumerate(plist):
        v = out[q]
        stacked[i, :v.shape[0]] = v
    return stacked


def med_spearman(pred, plist):
    r = []
    W = pred.shape[1]
    for i, q in enumerate(plist):
        m = store[q]["m"][:W]          # pred is padded to this split's longest protein, not LMAX
        if m.sum() >= 3:
            r.append(spearmanr(pred[i][m], store[q]["y"][:W][m]).correlation)
    return float(np.median(r))


best, best_ep, bad = -1, -1, 0
best_val = best_test = None
hist = []
t0 = time.time()
stop_at = args.fixed_epochs if (args.train_on == "trainval" and args.fixed_epochs > 0) else args.epochs
batches = length_batches(fit_pids, args.batch)
print(f"{len(batches)} length-bucketed batches, max pad {max(store[b[-1]]['L'] for b in batches)}", flush=True)
for ep in range(stop_at):
    model.train()
    tl = 0.0
    for bi in np.random.permutation(len(batches)):   # shuffle batch order, keep buckets intact
        chunk = batches[bi]
        seq, L, X, vm, y, m, w = batch_of(chunk)
        pred, aux_pred = model(seq, L, X, vm, noise=args.aug_eps if model.training else 0.0)
        loss = losses(pred.float(), None if aux_pred is None else aux_pred.float(), y, m, X, vm, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([q for g in gp for q in g["params"]], 1.0)
        opt.step()
        sched.step()
        tl += loss.item() * len(chunk)
    vp = predict(part["val"])
    vs = med_spearman(vp, part["val"])
    hist.append({"epoch": ep, "train_loss": tl / nfit, "val_median_spearman": vs, "t": round(time.time() - t0)})
    print(json.dumps(hist[-1]), flush=True)
    improved = vs > best
    if args.train_on == "trainval" and args.fixed_epochs > 0:
        improved = ep == stop_at - 1  # val is inside the fit set: no selection, just take the last epoch
    if improved:
        best, best_ep, bad = vs, ep, 0
        best_val, best_test = vp, predict(part["test"])
    else:
        bad += 1
        if bad >= args.patience and not (args.train_on == "trainval" and args.fixed_epochs > 0):
            break

np.savez_compressed(out_dir / f"{args.split}_seed{args.seed}.npz",
                    val_pids=np.array(part["val"]), val=best_val.astype(np.float32),
                    test_pids=np.array(part["test"]), test=best_test.astype(np.float32))
(out_dir / f"{args.split}_seed{args.seed}.json").write_text(json.dumps(
    {"args": vars(args), "best_val_median_spearman": best, "best_epoch": best_ep,
     "n_fit_proteins": nfit, "history": hist}, indent=1))
print("DONE", args.tag, args.split, args.seed, "best_val", round(best, 4), "epoch", best_ep,
      "elapsed", round(time.time() - t0), flush=True)
