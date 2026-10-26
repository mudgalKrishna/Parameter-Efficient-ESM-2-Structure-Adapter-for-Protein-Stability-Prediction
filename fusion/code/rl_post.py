"""RL post-training of the ESM-2 + ProteinMPNN stability model, directly on the eval metric.

WHY RL IS THE RIGHT TOOL HERE
-----------------------------
The model is scored by **median per-protein Spearman rho**, which is a function of the *ranking*
of its predictions. Spearman is piecewise constant, so its gradient is zero almost everywhere:
supervised training cannot optimise it and instead optimises differentiable proxies -- MSE,
per-protein Pearson, and a pairwise hinge. Round 3 tried a differentiable soft-rank surrogate and
gained +0.0003 on validation, i.e. nothing.

Policy gradients do not need the reward to be differentiable; they only need it to be computable.
That is exactly the situation here, so RL post-training optimises the real objective rather than a
stand-in for it.

THE POLICY
----------
The supervised network f_theta assigns a score to every candidate variant of a protein. Make it
stochastic by emitting a Gaussian over scores:

    s_i = f_theta(x_i) + sigma * eps_i ,   eps_i ~ N(0, 1)

so log pi(s | theta) = -sum_i (s_i - f_theta(x_i))^2 / (2 sigma^2) + const. One "action" is a
whole score vector for one protein; the reward is the quantity actually cared about:

    R(s) = Spearman( s , y )        over that protein's labelled variants

This is a one-step bandit per protein: no transitions, no value bootstrapping, no discounting.

METHODS COMPARED (--algo)
-------------------------
    reinforce    plain REINFORCE: grad = R * grad log pi                     (high variance)
    baseline     REINFORCE with the group mean as a baseline: (R - mean)
    grpo         group-relative, (R - mean) / (std + eps), PPO-style clipped ratio + KL to the
                 frozen supervised reference policy
    grpo_noclip  grpo without the clip, to isolate what clipping contributes

G samples are drawn per protein per step; a protein whose G rewards are identical carries zero
advantage and is skipped. The reference policy is the frozen supervised checkpoint, so the KL term
penalises drifting away from a model that already works.

    python rl_post.py --ckpt <path>.pt --algo grpo --tag rl_grpo_s0 --seed 0

Selection is on the split's validation proteins; the split's test proteins are scored once at the
end, exactly as in the supervised pipeline.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

# ---- reuse the supervised pipeline's data loading and model definition verbatim.
# stab3 parses argv with parse_known_args and guards its training loop under __main__, so
# importing it here builds the store, the splits and the Model without training anything.
_rl_argv = sys.argv[1:]
_sup = argparse.ArgumentParser(add_help=False)
_sup.add_argument("--ckpt", required=True)
_sup.add_argument("--split", default="cv0")
_sup.add_argument("--esm", default="facebook/esm2_t33_650M_UR50D")
_sup.add_argument("--lora_rank", type=int, default=32)
_known, _rest = _sup.parse_known_args()
sys.argv = [sys.argv[0], "--tag", "_rl_import", "--split", _known.split,
            "--esm", _known.esm, "--lora_rank", str(_known.lora_rank)]

import stab3 as S  # noqa: E402

sys.argv = [sys.argv[0]] + _rl_argv

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--ckpt", required=True, help="supervised checkpoint from stab3.py --save_ckpt 1")
ap.add_argument("--tag", required=True)
ap.add_argument("--algo", default="grpo", choices=["reinforce", "baseline", "grpo", "grpo_noclip"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--split", default="cv0")
ap.add_argument("--esm", default="facebook/esm2_t33_650M_UR50D")
ap.add_argument("--lora_rank", type=int, default=32)
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--proteins_per_step", type=int, default=4)
ap.add_argument("--group", type=int, default=16)        # G samples per protein
ap.add_argument("--sigma", type=float, default=0.3)     # exploration width, in label units
ap.add_argument("--lr", type=float, default=2e-6)
ap.add_argument("--beta", type=float, default=0.01)     # KL to the frozen reference
ap.add_argument("--clip", type=float, default=0.2)
ap.add_argument("--max_variants", type=int, default=256, help="subsample per protein per step")
ap.add_argument("--eval_every", type=int, default=25)
ap.add_argument("--patience", type=int, default=6)
args = ap.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
rng = np.random.default_rng(args.seed)
dev = "cuda"
OUT = S.paths.WORK / "runs_rl" / args.tag
OUT.mkdir(parents=True, exist_ok=True)

# ---- load the supervised policy, and a frozen copy as the reference
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
S.model.load_state_dict(ck["model"])
policy = S.model.to(dev)
print(f"loaded {args.ckpt}  (supervised val rho {ck.get('best_val_median_spearman')})", flush=True)

trainable = [q for q in policy.parameters() if q.requires_grad]
opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
print(f"{sum(q.numel() for q in trainable)} trainable params | algo={args.algo} "
      f"G={args.group} sigma={args.sigma} beta={args.beta}", flush=True)

part = S.part
fit = part["train"]
print({k: len(v) for k, v in part.items()}, flush=True)


def variant_scores(pid, idx=None, grad=True):
    """Model scores and truths for one protein's labelled variants.

    Returns (scores[k], truth[k]). `idx` subsamples the variant list for cost control.
    """
    rec = S.store[pid]
    seq, L, X, vm, y, m, _ = S.batch_of([pid])
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        pred, _ = policy(seq, L, X, vm)
    pred = pred.float()[0]
    mm = torch.as_tensor(rec["m"][: pred.shape[0]], device=dev)
    sc = pred[mm]
    tr = torch.as_tensor(rec["y"][: pred.shape[0]], device=dev)[mm]
    if idx is not None and sc.numel() > len(idx):
        sel = torch.as_tensor(idx, device=dev)
        sc, tr = sc[sel], tr[sel]
    return sc, tr


def spearman_reward(sample: np.ndarray, truth: np.ndarray) -> float:
    r = spearmanr(sample, truth).correlation
    return 0.0 if not np.isfinite(r) else float(r)


def evaluate(split_key):
    """Median per-protein Spearman of the deterministic policy mean (sigma = 0)."""
    policy.eval()
    rs = []
    with torch.no_grad():
        for pid in part[split_key]:
            sc, tr = variant_scores(pid, None, grad=False)
            if sc.numel() >= 3:
                rs.append(spearman_reward(sc.cpu().numpy(), tr.cpu().numpy()))
    policy.train()
    return float(np.median(rs)), len(rs)


# The reference policy is the frozen supervised checkpoint, so its scores are constant for the
# whole run: compute them once here rather than reloading weights mid-step. (Swapping the live
# model's state_dict during a step mutates parameters in place and breaks the autograd graph.)
REF = {}
with torch.no_grad():
    for pid in fit:
        sc, _ = variant_scores(pid, None, grad=False)
        REF[pid] = sc.detach().clone()
print(f"cached reference scores for {len(REF)} training proteins", flush=True)

sup_val, nval = evaluate("val")
sup_test, ntest = evaluate("test")
print(f"supervised start: val {sup_val:.4f} (n={nval})  test {sup_test:.4f} (n={ntest})", flush=True)

hist = []
best, best_step, bad = sup_val, 0, 0
best_sd = {k: v.detach().clone() for k, v in policy.state_dict().items()}
t0 = time.time()
sigma = args.sigma

for step in range(1, args.steps + 1):
    policy.train()
    pids = [fit[i] for i in rng.integers(0, len(fit), args.proteins_per_step)]
    opt.zero_grad(set_to_none=True)
    step_R, step_adv, step_kl, used = [], [], [], 0

    for pid in pids:
        k_all = int(S.store[pid]["m"].sum())
        if k_all < 5:
            continue
        idx = (rng.choice(k_all, args.max_variants, replace=False)
               if k_all > args.max_variants else np.arange(k_all))

        mu, truth = variant_scores(pid, idx, grad=True)      # one forward, reused for all G
        mu_d = mu.detach()
        t_np = truth.detach().cpu().numpy()

        eps = torch.randn(args.group, mu.numel(), device=dev)
        samples = mu_d.unsqueeze(0) + sigma * eps
        R = np.array([spearman_reward(s, t_np) for s in samples.cpu().numpy()])
        step_R.append(R.mean())

        if args.algo == "reinforce":
            A = torch.as_tensor(R, dtype=torch.float32, device=dev)
        elif args.algo == "baseline":
            A = torch.as_tensor(R - R.mean(), dtype=torch.float32, device=dev)
        else:
            if R.std() < 1e-8:
                continue                                     # degenerate group: no signal
            A = torch.as_tensor((R - R.mean()) / (R.std() + 1e-6), dtype=torch.float32, device=dev)
        step_adv.append(float(A.abs().mean()))

        # log pi of each sample under the current policy, up to a constant in sigma
        logp = -((samples - mu.unsqueeze(0)) ** 2).sum(1) / (2 * sigma ** 2)
        with torch.no_grad():
            logp_old = -((samples - mu_d.unsqueeze(0)) ** 2).sum(1) / (2 * sigma ** 2)

        if args.algo == "grpo":
            ratio = torch.exp((logp - logp_old).clamp(-20, 20))
            loss = -torch.min(ratio * A, ratio.clamp(1 - args.clip, 1 + args.clip) * A).mean()
        else:
            loss = -(logp * A).mean() / max(mu.numel(), 1)

        if args.beta > 0:
            # Gaussian KL between current and reference policy at matched sigma
            mu_ref = REF[pid][torch.as_tensor(idx, device=dev)]
            kl = ((mu - mu_ref) ** 2).mean() / (2 * sigma ** 2)
            loss = loss + args.beta * kl
            step_kl.append(float(kl.detach()))

        (loss / args.proteins_per_step).backward()
        used += 1

    if used == 0:
        continue
    gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()

    if step % 5 == 0 or step == 1:
        rec = dict(step=step, mean_reward=round(float(np.mean(step_R)), 4),
                   mean_abs_adv=round(float(np.mean(step_adv)), 4) if step_adv else None,
                   kl=round(float(np.mean(step_kl)), 5) if step_kl else None,
                   grad_norm=round(float(gn), 3), proteins=used, t=round(time.time() - t0))
        hist.append(rec)
        print(json.dumps(rec), flush=True)

    if args.eval_every and step % args.eval_every == 0:
        v, _ = evaluate("val")
        hist.append({"step": step, "val_median_spearman": round(v, 4),
                     "t": round(time.time() - t0)})
        print(f"  [val] step {step}: {v:.4f}  (best {best:.4f}, supervised {sup_val:.4f})",
              flush=True)
        if v > best:
            best, best_step, bad = v, step, 0
            best_sd = {k: q.detach().clone() for k, q in policy.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop at step {step}", flush=True)
                break

policy.load_state_dict(best_sd)                 # val-selected, never test-selected
final_val, _ = evaluate("val")
final_test, _ = evaluate("test")

res = {
    "tag": args.tag, "algo": args.algo, "args": vars(args),
    "supervised_val": round(sup_val, 4), "supervised_test": round(sup_test, 4),
    "rl_val": round(final_val, 4), "rl_test": round(final_test, 4),
    "delta_val": round(final_val - sup_val, 4), "delta_test": round(final_test - sup_test, 4),
    "best_step": best_step, "n_val": nval, "n_test": ntest,
    "elapsed_s": round(time.time() - t0),
}
(OUT / "history.json").write_text(json.dumps(hist, indent=1))
(OUT / "result.json").write_text(json.dumps(res, indent=1))
torch.save({"model": policy.state_dict(), "args": vars(args)}, OUT / "model.pt")
print(json.dumps(res, indent=1), flush=True)
print(f"DONE {args.tag}: val {sup_val:.4f} -> {final_val:.4f} "
      f"({final_val - sup_val:+.4f}) | test {sup_test:.4f} -> {final_test:.4f} "
      f"({final_test - sup_test:+.4f})", flush=True)
