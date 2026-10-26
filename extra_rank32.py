"""Extra improvement step: LoRA rank-32 + SPURS structure adapter.

Motivation (from the 150M run): more trainable capacity helped (frozen 0.7985 ->
LoRA 0.8153 -> full 0.8211 val median rho), and the test winner was LoRA + SPURS
at rank 16. This step doubles LoRA capacity (rank 32, alpha 64, same alpha/rank
ratio) while keeping every other hyperparameter identical.

Selection discipline: the seed-ensemble validation median Spearman is compared
against the main pipeline's selected structure-adapter winner. The test split is
evaluated ONCE, and only if this variant wins on validation.
"""
import ast
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

NOTEBOOK = Path(os.environ.get("ESM2_NOTEBOOK", str(Path(__file__).resolve().with_name("train_esm2_spurs_fixed.py"))))
BUNDLE = Path(os.environ.get("ESM2_BUNDLE", "training_bundle.zip"))
MPNN = os.environ.get("ESM2_MPNN_CKPT", "assets/v_48_020.pt")
TAG = sys.argv[1] if len(sys.argv) > 1 else "650"
MODEL = {
    "150": "facebook/esm2_t30_150M_UR50D",
    "650": "facebook/esm2_t33_650M_UR50D",
}[TAG]
MAIN_DIR = Path(os.environ.get("ESM2_MAIN_DIR", f"esm2_structure_track_{TAG}M"))
OUT = Path(os.environ.get("ESM2_EXTRA_OUT", f"esm2_extra_rank32_{TAG}M"))
STATUS = Path(os.environ.get("ESM2_STATUS", f"extra_rank32_{TAG}.status"))
SEEDS = [42, 101, 202, 303, 404, 505, 606, 707]
RANK, ALPHA = 32, 64.0
BATCH, EPOCHS, PATIENCE = 16, 100, 20
HEAD_LR = 1e-3
BACKBONE_LR = 2e-5 if TAG == "150" else 1e-5
DEVICE = "cuda"


def load_cells(src):
    tree = ast.parse(src)
    return [(n.name, n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("_cell_")]


def exec_cell(node, G):
    # Runs one `_cell_*` function body from this project's own marimo notebook
    # (train_esm2_spurs_fixed.py), parsed from that file with `ast` above. The source is a
    # fixed path inside this repository -- nothing is fetched, decoded or executed from the
    # network. This exists because marimo stores cells as functions, and the notebook has to
    # be driven as a plain script here.
    body = list(node.body)
    while body and isinstance(body[-1], ast.Return):
        body.pop()
    mod = ast.Module(body=body, type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), f"<{node.name}>", "exec"), G)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    STATUS.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} START rank32 {MODEL}\n")

    src = NOTEBOOK.read_text()
    src = src.replace('    _cli = mo.cli_args() if hasattr(mo, "cli_args") else {}', "    _cli = {}")
    G = {}
    for name, node in load_cells(src):
        if name == "_cell_execute_pipeline":
            break
        print("exec", name, flush=True)
        exec_cell(node, G)

    print("loading bundle", flush=True)
    bundle = G["load_and_verify_bundle_manifest"](BUNDLE)
    G["verify_and_apply_sign_once"](bundle, sign_multiplier=1.0)
    proteins, _folds, _audit = G["build_sequence_identity_clusters_on_train"](
        bundle["proteins_df"],
        bundle["variants_by_split"]["train"],
        identity_threshold=0.25,
        n_folds=5,
        seed=SEEDS[0],
    )
    bundle["proteins_df"] = proteins
    tr = bundle["tensors_by_split"]["train"]
    va = bundle["tensors_by_split"]["val"]
    te = bundle["tensors_by_split"]["test"]

    print("env3d", flush=True)
    env = {
        sp: G["extract_per_residue_3d_env_features"](
            bundle["tensors_by_split"][sp]["seq_indices"],
            bundle["tensors_by_split"][sp]["lengths"],
            bundle["tensors_by_split"][sp]["backbone_coords"],
            bundle["tensors_by_split"][sp]["residue_valid_mask"],
        )
        for sp in ("train", "val", "test")
    }

    print("zeroshot llr", flush=True)
    zs = G["compute_zeroshot_esm2_masked_marginal_and_wt_llr"](
        model_name=MODEL,
        proteins_df=proteins,
        train_tensors=tr,
        val_tensors=va,
        allow_smoke_surrogate=False,
        device=DEVICE,
        batch_size=BATCH * 2,
    )
    wt = zs["unmasked_wt_llr"]
    test_zs = wt["predict_split_fn"](te, "test")

    status = {"model": MODEL, "rank": RANK, "alpha": ALPHA, "seeds": SEEDS, "per_seed_val": [], "fns": []}
    val_preds = []
    for sd in SEEDS:
        print(f"train seed {sd}", flush=True)
        STATUS.write_text(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} TRAIN seed={sd} elapsed={time.time()-t0:.0f}s\n"
        )
        res = G["train_esm2_finetune_mode_single_seed"](
            step_id=f"EX01_LoRA_R{RANK}_Plus_ProteinMPNN_SPURSAdapter",
            model_name=MODEL,
            finetune_mode="lora",
            proteins_df=proteins,
            train_tensors=tr,
            val_tensors=va,
            train_env_3d=env["train"],
            val_env_3d=env["val"],
            train_zs_llr=wt["train_preds"],
            val_zs_llr=wt["val_preds"],
            llr_sign_multiplier=wt["sign_multiplier"],
            out_dir=OUT,
            loss_fn_name="huber",
            huber_beta=1.0,
            lora_rank=RANK,
            lora_alpha=ALPHA,
            lora_dropout=0.05,
            head_lr=HEAD_LR,
            backbone_lr=BACKBONE_LR,
            weight_decay=0.01,
            batch_size=BATCH,
            max_epochs=EPOCHS,
            patience=PATIENCE,
            use_spurs_adapter=True,
            proteinmpnn_ckpt_path=MPNN,
            include_handcrafted_3d=False,
            train_protein_mask=None,
            allow_smoke_surrogate=False,
            seed=sd,
            device=DEVICE,
        )
        val_preds.append(res["val_predictions"])
        status["per_seed_val"].append(float(res["best_val_median_spearman_rho"]))
        status["fns"].append(res["predict_split_fn"])
        print(f"  seed {sd} val_med={res['best_val_median_spearman_rho']:.4f}", flush=True)

    ens_val = np.mean(np.stack(val_preds, axis=0), axis=0)
    med, n_ev, n_ex, _ = G["compute_validation_median_spearman"](
        va["y_target"], ens_val, va["label_mask"], va["protein_ids"]
    )
    status["ensemble_val_median"] = float(med)
    status["val_proteins"] = int(n_ev)

    locked = json.loads((MAIN_DIR / "selected_model_before_test.json").read_text())
    ref = locked["selected_structure_adapter_winner"]["val_median_spearman"]
    status["main_structure_winner"] = ref["step_id"] if isinstance(ref, dict) else ref
    status["main_structure_winner_val"] = (
        ref["val_median_spearman"] if isinstance(ref, dict) else float(ref)
    )
    better = bool(np.isfinite(med) and med > status["main_structure_winner_val"])
    status["beats_main_on_val"] = better
    print(f"ENSEMBLE val_med={med:.4f} vs main={status['main_structure_winner_val']:.4f} better={better}", flush=True)

    if better:
        ens_test = np.mean(
            np.stack([f("test", te, env["test"], test_zs) for f in fns], axis=0), axis=0
        )
        tmed, tn, tex, _ = G["compute_validation_median_spearman"](
            te["y_target"], ens_test, te["label_mask"], te["protein_ids"]
        )
        status["test_median_spearman"] = float(tmed)
        status["test_proteins"] = int(tn)
        status["test_evaluated_once"] = True

        rows = []
        from scipy import stats as _st

        for i in range(len(te["protein_ids"])):
            m = te["label_mask"][i]
            if int(np.sum(m)) < 3:
                continue
            yt = te["y_target"][i][m].astype(np.float64)
            yp = ens_test[i][m].astype(np.float64)
            if np.std(yt) <= 1e-9 or np.std(yp) <= 1e-9:
                continue
            rho = float(_st.spearmanr(yt, yp).correlation)
            if np.isfinite(rho):
                rows.append({"protein_id": str(te["protein_ids"][i]), "spearman_rho": rho})
        pd.DataFrame(rows).to_csv(OUT / "per_protein__EX01_rank32.csv", index=False)
        print(f"TEST median={tmed:.4f} on {tn} proteins", flush=True)
    else:
        status["test_evaluated_once"] = False
        status["note"] = "variant did not beat the main winner on validation; test not evaluated"

    status["elapsed_seconds"] = round(time.time() - t0, 1)
    status.pop("fns", None)
    (OUT / "extra_run_summary.json").write_text(json.dumps(status, indent=2))
    STATUS.write_text(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} COMPLETE rank32 {status['elapsed_seconds']}s\n"
    )
    print("COMPLETE", json.dumps({k: v for k, v in status.items() if k != "fns"}),
          flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        STATUS.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} FAILED\n{tb[-3000:]}\n")
        print(tb, flush=True)
        raise
