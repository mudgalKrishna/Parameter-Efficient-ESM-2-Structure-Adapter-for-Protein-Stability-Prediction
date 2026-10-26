"""Pick the RL algorithm on cv0 validation, then write the phase-2 queue for all 5 folds.

  python rl_select.py              report the cv0 comparison
  python rl_select.py --emit       also write run_rl_folds.sh for the winning algorithm

Selection is on validation only. Phase 2 runs the winner across cv0..cv4 so the result rests on
five folds rather than one, matching the evidence standard the supervised project is held to.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

W = Path("/marimo/work")
RL = W / "runs_rl"

ap = argparse.ArgumentParser()
ap.add_argument("--emit", action="store_true")
ap.add_argument("--seeds", default="0 1 2")
args = ap.parse_args()

rows = []
for f in sorted(RL.glob("*/result.json")):
    d = json.loads(f.read_text())
    if d.get("args", {}).get("split") != "cv0":
        continue
    rows.append(d)

if not rows:
    print("no cv0 results yet")
    raise SystemExit(0)

print(f"{'algo':13} {'seed':>4} {'sup val':>8} {'rl val':>8} {'d val':>8} "
      f"{'sup test':>9} {'rl test':>8} {'d test':>8} {'step':>5}")
print("-" * 82)
agg = defaultdict(list)
for d in sorted(rows, key=lambda r: (r.get("method", r.get("algo")), r["args"]["seed"])):
    print(f"{d.get('method', d.get('algo')):13} {d['args']['seed']:>4} {d['supervised_val']:8.4f} {d['rl_val']:8.4f} "
          f"{d['delta_val']:+8.4f} {d['supervised_test']:9.4f} {d['rl_test']:8.4f} "
          f"{d['delta_test']:+8.4f} {d['best_step']:>5}")
    agg[d.get("method", d.get("algo"))].append(d)

print(f"\n{'algo':13} {'mean d val':>11} {'std':>7} {'mean d test':>12} {'n':>3}")
print("-" * 50)
summary = {}
for a, ds in agg.items():
    dv = [x["delta_val"] for x in ds]
    dt = [x["delta_test"] for x in ds]
    summary[a] = {"mean_delta_val": float(np.mean(dv)), "std_delta_val": float(np.std(dv)),
                  "mean_delta_test": float(np.mean(dt)), "n": len(ds),
                  "per_seed_delta_val": [round(x, 4) for x in dv]}
    print(f"{a:13} {np.mean(dv):+11.4f} {np.std(dv):7.4f} {np.mean(dt):+12.4f} {len(ds):>3}")

winner = max(summary, key=lambda a: summary[a]["mean_delta_val"])
best_dv = summary[winner]["mean_delta_val"]
print(f"\nval-selected algorithm: {winner}  (mean delta val {best_dv:+.4f})")
if best_dv <= 0:
    print("NOTE: no algorithm improved validation. Phase 2 would be measuring noise;")
    print("      the honest output is the null result on cv0 with all four arms reported.")

out = {"phase1_split": "cv0", "per_run": rows, "summary": summary, "winner": winner,
       "winner_mean_delta_val": round(best_dv, 4),
       "improves_val": bool(best_dv > 0)}
(W / "rl_phase1.json").write_text(json.dumps(out, indent=1))
print("wrote", W / "rl_phase1.json")

if args.emit and best_dv > 0:
    seeds = args.seeds.split()
    lines = ["#!/bin/bash",
             "export ESM2_WORK=/marimo/work",
             "export ESM2_BUNDLE=/marimo/training_bundle_copy.zip",
             "cd /marimo/work",
             f"# phase 2: {winner} (chosen on cv0 val) across all five folds",
             ""]
    for k in range(5):
        for sd in seeds:
            tag = f"RLF_{winner}_cv{k}_s{sd}"
            ck = f"/marimo/work/runs3/SUP650/cv{k}_seed0_model.pt"
            lines += [
                f'if [ -s "/marimo/work/runs_rl/{tag}/result.json" ]; then',
                f'  echo "skip {tag}" >> logs/rlf_stage.log',
                "else",
                f'  echo "start {tag}" >> logs/rlf_stage.log',
                f"  python rl_post.py --ckpt {ck} --tag {tag} --algo {winner} --seed {sd} \\",
                f"    --split cv{k} --steps 300 --proteins_per_step 4 --group 16 --sigma 0.3 \\",
                "    --lr 2e-6 --eval_every 25 --patience 6 --max_variants 256 \\",
                f"    > logs/{tag}.log 2>&1",
                f'  echo "end {tag} rc=$?" >> logs/rlf_stage.log',
                "fi",
            ]
    lines += ["echo RLFOLDSDONE >> logs/rlf_stage.log", ""]
    (W / "run_rl_folds.sh").write_text("\n".join(lines))
    print("wrote", W / "run_rl_folds.sh", f"({5 * len(seeds)} runs)")
elif args.emit:
    print("not emitting phase 2: nothing beat the supervised model on validation")
