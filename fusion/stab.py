"""ESM-2 + official ProteinMPNN fusion model for Tsuboyama ddG (ThermoMPNN mega split).

One run = one (config, seed). Model selection uses validation median per-protein
Spearman only. Test predictions from the best-validation epoch are written to
disk but never scored here; scoring happens once in the final ensemble step.
"""
import argparse
import io
import json
import math
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
AA = "ACDEFGHIKLMNPQRSTVWY"  # same order as ProteinMPNN alphabet (minus X)

p = argparse.ArgumentParser()
p.add_argument("--esm", default="facebook/esm2_t33_650M_UR50D")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", required=True)
p.add_argument("--lora_rank", type=int, default=16)
p.add_argument("--lora_lr", type=float, default=3e-4)
p.add_argument("--head_lr", type=float, default=1e-3)
p.add_argument("--mpnn_lr", type=float, default=0.0)  # 0 = frozen
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--patience", type=int, default=10)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--corr_w", type=float, default=1.0)
p.add_argument("--fuse_layers", type=int, default=2)
p.add_argument("--use_esm", type=int, default=1)
p.add_argument("--use_mpnn", type=int, default=1)
p.add_argument("--wd", type=float, default=0.01)
p.add_argument("--esm_bf16", type=int, default=0)
p.add_argument("--mpnn_ver", default="v_48_020")
p.add_argument("--aug_eps", type=float, default=0.0)
p.add_argument("--grad_ckpt", type=int, default=0)
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = "cuda"
out_dir = Path(f"/marimo/work/runs/{args.tag}")
out_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- data
def load_split(z, sp):
    t = np.load(io.BytesIO(z.read(f"tensors/{sp}_tensors.npz")))
    d = {k: t[k] for k in t.files}
    return d


z = zipfile.ZipFile(BUNDLE)
data = {sp: load_split(z, sp) for sp in ("train", "val", "test")}
for sp, d in data.items():
    assert d["label_mask"].sum() > 0, sp
print({sp: (len(d["protein_ids"]), int(d["label_mask"].sum())) for sp, d in data.items()}, flush=True)


def to_dev(d, idx):
    seq = torch.as_tensor(d["seq_indices"][idx].astype(np.int64))
    L = torch.as_tensor(d["lengths"][idx].astype(np.int64))
    X = torch.nan_to_num(torch.as_tensor(d["backbone_coords"][idx]))
    vmask = torch.as_tensor(d["residue_valid_mask"][idx]).float()
    y = torch.as_tensor(np.nan_to_num(d["labels_raw"][idx]))
    m = torch.as_tensor(d["label_mask"][idx])
    return [x.to(dev) for x in (seq, L, X, vmask, y, m)]


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

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype)) * self.s


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        d_fuse = 512
        self.use_esm, self.use_mpnn = bool(args.use_esm), bool(args.use_mpnn)
        in_dim = 0
        if self.use_esm:
            from transformers import AutoTokenizer, EsmForMaskedLM

            tok = AutoTokenizer.from_pretrained(args.esm)
            self.esm = EsmForMaskedLM.from_pretrained(args.esm, torch_dtype=torch.bfloat16 if args.esm_bf16 else torch.float32)
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
            nl = self.esm.config.num_hidden_layers + 1
            self.layer_w = nn.Parameter(torch.zeros(nl))
            D = self.esm.config.hidden_size
            self.esm_proj = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d_fuse))
            self.alpha_llr = nn.Parameter(torch.tensor(0.1))
            in_dim += d_fuse
        if self.use_mpnn:
            ck = torch.load(f"{MPNN_DIR}/{args.mpnn_ver}.pt", map_location="cpu", weights_only=False)
            self.mpnn = ProteinMPNN(ca_only=False, num_letters=21, node_features=128, edge_features=128,
                                    hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                                    k_neighbors=ck["num_edges"], augment_eps=0.0)
            self.mpnn.load_state_dict(ck["model_state_dict"])  # strict: all tensors
            for q in self.mpnn.parameters():
                q.requires_grad = args.mpnn_lr > 0
            self.mpnn_proj = nn.Sequential(nn.LayerNorm(128 * 4), nn.Linear(128 * 4, d_fuse))
            in_dim += d_fuse
        self.fuse_in = nn.Linear(in_dim, d_fuse)
        enc = nn.TransformerEncoderLayer(d_fuse, 8, 4 * d_fuse, dropout=0.1, batch_first=True, norm_first=True)
        self.fuse = nn.TransformerEncoder(enc, args.fuse_layers) if args.fuse_layers > 0 else None
        self.head = nn.Sequential(nn.LayerNorm(d_fuse), nn.Linear(d_fuse, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 20))

    def forward(self, seq, L, X, vmask):
        B, Lm = seq.shape
        pos_mask = torch.arange(Lm, device=seq.device)[None] < L[:, None]
        safe = seq.clamp(0, 19)
        feats, llr = [], 0.0
        if self.use_esm:
            ids = torch.full((B, Lm + 2), self.pad_id, device=seq.device)
            ids[:, 0] = self.cls_id
            ids[:, 1:Lm + 1] = torch.where(pos_mask, self.aa_tok.to(seq.device)[safe], torch.full_like(safe, self.pad_id))
            ids[torch.arange(B), L + 1] = self.eos_id
            am = (ids != self.pad_id).long()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = self.esm(input_ids=ids, attention_mask=am, output_hidden_states=True)
            hs = torch.stack(o.hidden_states, 0)[:, :, 1:Lm + 1].float()
            w = torch.softmax(self.layer_w, 0)
            h = (w[:, None, None, None] * hs).sum(0)
            feats.append(self.esm_proj(h))
            lp = F.log_softmax(o.logits[:, 1:Lm + 1].float()[..., self.aa_tok.to(seq.device)], -1)
            llr = self.alpha_llr * (lp - lp.gather(-1, safe[..., None]))
        if self.use_mpnn:
            Sm = torch.where(pos_mask, safe, torch.full_like(safe, 20))
            mk = vmask * pos_mask.float()
            ridx = torch.arange(Lm, device=seq.device)[None].expand(B, -1)
            ch = torch.ones_like(mk)
            ctx = torch.enable_grad() if args.mpnn_lr > 0 else torch.no_grad()
            if self.training and args.aug_eps > 0:
                X = X + args.aug_eps * torch.randn_like(X)
            with ctx:
                hid, emb, _ = self.mpnn(X, Sm, mk, mk, ridx, ch, None, use_input_decoding_order=True, decoding_order=ridx)
            feats.append(self.mpnn_proj(torch.cat(list(hid) + [emb], -1)))
        x = self.fuse_in(torch.cat(feats, -1))
        if self.fuse is not None:
            x = self.fuse(x, src_key_padding_mask=~pos_mask)
        raw = self.head(x)
        out = raw - raw.gather(-1, safe[..., None]) + llr
        return out


model = Model().to(dev)
groups = [
    {"params": [q for n, q in model.named_parameters() if q.requires_grad and (n.endswith(".A") or n.endswith(".B"))], "lr": args.lora_lr},
    {"params": [q for n, q in model.named_parameters() if q.requires_grad and n.startswith("mpnn.")], "lr": args.mpnn_lr},
    {"params": [q for n, q in model.named_parameters() if q.requires_grad and not (n.endswith(".A") or n.endswith(".B") or n.startswith("mpnn."))], "lr": args.head_lr},
]
groups = [g for g in groups if g["params"]]
opt = torch.optim.AdamW(groups, weight_decay=args.wd)
ntr = len(data["train"]["protein_ids"])
steps = args.epochs * math.ceil(ntr / args.batch)
warm = max(1, steps // 20)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
print("trainable", sum(q.numel() for q in model.parameters() if q.requires_grad), flush=True)


def corr_loss(pred, y, m):
    tot, n = 0.0, 0
    for i in range(pred.shape[0]):
        mi = m[i]
        if mi.sum() < 3:
            continue
        a, b = pred[i][mi], y[i][mi]
        a, b = a - a.mean(), b - b.mean()
        tot = tot + (1 - (a * b).sum() / (a.norm() * b.norm() + 1e-8))
        n += 1
    return tot / max(n, 1)


@torch.no_grad()
def predict(sp):
    model.eval()
    d = data[sp]
    outs = []
    for s in range(0, len(d["protein_ids"]), 16):
        idx = np.arange(s, min(s + 16, len(d["protein_ids"])))
        seq, L, X, vm, _, _ = to_dev(d, idx)
        outs.append(model(seq, L, X, vm).float().cpu().numpy())
    return np.concatenate(outs)


def median_spearman(pred, d):
    r = []
    for i in range(len(d["protein_ids"])):
        m = d["label_mask"][i]
        if m.sum() >= 3:
            r.append(spearmanr(pred[i][m], d["labels_raw"][i][m]).correlation)
    return float(np.median(r))


best, best_ep, bad, hist = -1, -1, 0, []
t0 = time.time()
for ep in range(args.epochs):
    model.train()
    perm = np.random.permutation(ntr)
    tl = 0.0
    for s in range(0, ntr, args.batch):
        idx = perm[s:s + args.batch]
        seq, L, X, vm, y, m = to_dev(data["train"], idx)
        pred = model(seq, L, X, vm).float()
        loss = F.mse_loss(pred[m], y[m]) + args.corr_w * corr_loss(pred, y, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([q for g in groups for q in g["params"]], 1.0)
        opt.step()
        sched.step()
        tl += loss.item() * len(idx)
    vp = predict("val")
    vs = median_spearman(vp, data["val"])
    hist.append({"epoch": ep, "train_loss": tl / ntr, "val_median_spearman": vs, "t": round(time.time() - t0)})
    print(json.dumps(hist[-1]), flush=True)
    if vs > best:
        best, best_ep, bad = vs, ep, 0
        best_val = vp
        best_test = predict("test")  # stored, not scored
    else:
        bad += 1
        if bad >= args.patience:
            break

np.savez_compressed(out_dir / f"seed{args.seed}.npz", val=best_val.astype(np.float32), test=best_test.astype(np.float32))
(out_dir / f"seed{args.seed}.json").write_text(json.dumps({"args": vars(args), "best_val_median_spearman": best, "best_epoch": best_ep, "history": hist}, indent=1))
print("DONE", args.tag, args.seed, "best_val", round(best, 4), "epoch", best_ep, "elapsed", round(time.time() - t0), flush=True)
