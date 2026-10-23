# ESM-2 + ProteinMPNN fusion (beats released ThermoMPNN)

Tsuboyama mega-scale dataset, ThermoMPNN's own split (`mega_splits.pkl`: 239 train / 31 val / 28 test proteins,
28,172 test variants). Metric: median per-protein Spearman.

## Result (test scored once, after the ensemble was locked on validation)

| Model | Test median ρ | Δ vs ThermoMPNN (95% bootstrap CI) | Proteins won | Wilcoxon p |
|---|---|---|---|---|
| **Locked validation-selected ensemble** | **0.8416** | **+0.070** [+0.038, +0.095] | 24 / 28 | 1.1e-5 |
| Best single config (ESM-2 3B LoRA r32, 8-seed avg) | 0.8263 | +0.054 [+0.017, +0.077] | 21 / 28 | 4.8e-3 |
| ThermoMPNN (released `thermoMPNN_default.pt`, re-run on identical variants) | 0.7720 | — | — | — |

Ensemble members (`results/selection_locked_before_test.json`): ESM-2 3B + coordinate-noise aug (×3),
ESM-2 15B low-LR (×2), ESM-2 15B, ESM-2 650M with fine-tuned ProteinMPNN, ProteinMPNN-only. Each member
is itself a 3–8 seed average; members are z-scored per protein before averaging.

## Model (`stab.py`)
- ESM-2 (650M / 3B / 15B-bf16) with LoRA on q/k/v/dense, softmax-weighted mix of all hidden layers, plus a
  learned α·(masked-LM log-likelihood ratio) term.
- **Official ProteinMPNN** (`v_48_020/002/030`, full strict state_dict load, all residues visible, WT
  sequence) — decoder hidden states + sequence embeddings, as ThermoMPNN uses them. This replaces the
  re-implemented encoder in `train_esm2_spurs_fixed.py`, which only partially matched the checkpoint.
- 2-layer transformer fusing both streams over residues → per-residue 20-way head, ΔΔG = head[mut] − head[wt].
- Loss: MSE + (1 − per-protein Pearson). Cosine LR with warmup, early stopping on val median Spearman.

## Ablations (validation, mean of per-seed best val median ρ; `results/validation_sweep.csv`)
ProteinMPNN only 0.797 · ESM-2 650M only 0.848 · fusion 650M 0.857 · fusion 3B 0.865 · fusion 15B 0.864.
Fusion > either stream alone; 3B ≈ 15B; the correlation loss, fused transformer and MPNN fine-tuning each matter
little individually but add diversity to the ensemble.

## Reproduce (molab / any CUDA box)
```
git clone https://github.com/dauparas/ProteinMPNN ext/ProteinMPNN
git clone https://github.com/Kuhlman-Lab/ThermoMPNN ext/ThermoMPNN
python stab.py --tag H3_3B_r32 --seed 0 --esm facebook/esm2_t36_3B_UR50D --lora_rank 32 --lora_lr 2e-4 --head_lr 5e-4
python launch.py 4          # runs every line of queue.txt (results/queue.txt has the full sweep)
python ens.py               # validation-only greedy ensemble -> selection.json
python ens.py --test        # single locked test evaluation vs ThermoMPNN
```
Paths in the scripts assume the sandbox layout (`/marimo/training_bundle_copy.zip`, `/marimo/work/...`).
A 650M run takes ~3 min, 3B ~4 min, 15B ~6 min on one RTX PRO 6000 (96 GB).
