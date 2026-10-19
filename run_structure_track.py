"""Run the recovered ESM-2 structure-adapter track on the molab sandbox.

Executes the recovered notebook cells (upb/train_esm2_spurs_fixed.py) in a plain
Python process with the best-known hyperparameters, for one model at a time.

Usage: python run_structure_track.py <150|650>
"""
import ast
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

NOTEBOOK = Path(os.environ.get("ESM2_NOTEBOOK", str(Path(__file__).resolve().with_name("train_esm2_spurs_fixed.py"))))
BUNDLE = os.environ.get("ESM2_BUNDLE", "training_bundle.zip")
MPNN_CKPT = os.environ.get("ESM2_MPNN_CKPT", "assets/v_48_020.pt")
THERMO_PREDS = os.environ.get("ESM2_THERMO_PREDS", "thermompnn_all_preds.parquet")
SEEDS = "42,101,202,303,404,505,606,707"

LORA_LR_PATCH_OLD = """            param_groups = [{"params": head_params, "lr": float(head_lr), "weight_decay": float(weight_decay)}]
            if backbone_params:
                param_groups.append(
                    {"params": backbone_params, "lr": float(backbone_lr), "weight_decay": float(weight_decay)}
                )"""
LORA_LR_PATCH_NEW = """            _lora_ps = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
            if _lora_ps:
                head_params = [p for p in head_params if not any(p is q for q in _lora_ps)]
            param_groups = [{"params": head_params, "lr": float(head_lr), "weight_decay": float(weight_decay)}]
            if _lora_ps:
                param_groups.append(
                    {"params": _lora_ps, "lr": 1e-4, "weight_decay": float(weight_decay)}
                )
            if backbone_params:
                param_groups.append(
                    {"params": backbone_params, "lr": float(backbone_lr), "weight_decay": float(weight_decay)}
                )"""


PREFLIGHT_OLD = '''    # Fix 33: Run pre-flight smoke test before launching full training / baselines
    preflight_smoke_report = run_preflight_smoke_test(
        proteins_df=proteins_clustered_df,
        train_tensors=bundle_data["tensors_by_split"]["train"],
        val_tensors=bundle_data["tensors_by_split"]["val"],
        train_env_3d=env_3d_by_split["train"],
        val_env_3d=env_3d_by_split["val"],
        out_dir=_out_dir,
        train_single_seed_fn=train_esm2_finetune_mode_single_seed,
        allow_smoke_surrogate=_allow_smoke_surrogate,
        model_name=_esm2_model,
        device=_device,
    )'''
PREFLIGHT_NEW = '''    # Pre-flight smoke test skipped: it hardcodes an empty ProteinMPNN checkpoint
    # path and therefore cannot run with the real (non-surrogate) structure adapter.
    preflight_smoke_report = {"status": "skipped", "reason": "structure adapter enabled"}'''


GC_OLD = '''                handcrafted_3d_dim=train_env_3d.shape[-1],
            ).to(device)

            grad_audit = audit_lora_injection_and_mode_gradients(model, finetune_mode)'''
GC_NEW = '''                handcrafted_3d_dim=train_env_3d.shape[-1],
            ).to(device)

            # Memory guard: full fine-tuning at 650M with split-wide max-length
            # padding exceeds the 95 GiB device budget (OOM observed 2026-10-19).
            # Recompute activations instead of storing them.
            if finetune_mode == "full":
                try:
                    model.esm.gradient_checkpointing_enable()
                    model.esm.config.use_cache = False
                    print("[mem] gradient checkpointing enabled for full fine-tune", flush=True)
                except Exception as _gc_err:
                    print(f"[mem] WARNING: gradient checkpointing unavailable: {_gc_err}", flush=True)

            grad_audit = audit_lora_injection_and_mode_gradients(model, finetune_mode)'''

PREDICT_OLD = '''            def _predict_any_split_torch(
                sp_name: str,
                sp_tensors: Dict[str, np.ndarray],
                sp_env_3d: np.ndarray,
                sp_zs_llr: np.ndarray,
            ) -> np.ndarray:
                sb_any = _prepare_torch_split(sp_name, sp_tensors, sp_env_3d)
                return _predict_sb_torch(sb_any)'''
PREDICT_NEW = '''            def _predict_any_split_torch(
                sp_name: str,
                sp_tensors: Dict[str, np.ndarray],
                sp_env_3d: np.ndarray,
                sp_zs_llr: np.ndarray,
            ) -> np.ndarray:
                # Weights are parked on the meta device between steps (see the
                # return site below). Materialize them from this seed's
                # best-validation checkpoint for the prediction pass only.
                _reload_path = getattr(model, "_meta_reload_path", None)
                if _reload_path:
                    _payload = torch.load(_reload_path, map_location="cpu", weights_only=False)
                    model.load_state_dict(_payload["model_state_dict"], assign=True)
                    del _payload
                    # Non-persistent buffers (e.g. transformers' ESM
                    # esm.embeddings.position_ids) are absent from state_dict(),
                    # so assign=True leaves them on meta and model.to(device)
                    # raises "Cannot copy out of meta tensor; no data!".
                    # Refill them from the snapshot taken at park time.
                    for _bn, _bt in getattr(model, "_meta_buffers", {}).items():
                        _pname, _, _leaf = _bn.rpartition(".")
                        _parent = model.get_submodule(_pname) if _pname else model
                        _parent._buffers[_leaf] = _bt
                model.to(device)
                sb_any = _prepare_torch_split(sp_name, sp_tensors, sp_env_3d)
                try:
                    return _predict_sb_torch(sb_any)
                finally:
                    if _reload_path:
                        model.to("meta")
                        import gc as _gc_pred
                        _gc_pred.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()'''

CACHE_OLD = '''            return {
                "backend": f"PyTorch+HF ({model_name}, mode={finetune_mode}, spurs_adapter={use_spurs_adapter})",'''
CACHE_NEW = '''            # Each seed's predict closure keeps its model alive for the ensemble
            # test pass. Every completed step therefore pinned 8 x 2.6 GB of
            # 650M weights in host RAM; at ~5 checkpoints the container crossed
            # its 160 GiB limit and was killed (filesystem reset) twice.
            # Park the weights on the meta device: the module skeleton stays,
            # and the closure reloads the checkpoint for its single pass.
            del optimizer
            del scheduler
            try:
                model._meta_reload_path = str(ckpt_file)
                # Snapshot buffers first: buffers flagged non-persistent by the
                # HF module (transformers 5.x ESM's position_ids) are not in
                # state_dict(), so they must be restored by hand on reload.
                model._meta_buffers = {
                    _bn: _bb.detach().to("cpu").clone()
                    for _bn, _bb in model.named_buffers()
                }
                model.to("meta")
                best_state = None
                import gc as _gc_meta
                _gc_meta.collect()
                print("[mem] model parked on meta device", flush=True)
            except Exception as _park_err:
                # Restore full weights if a partial park left meta tensors.
                try:
                    _payload = torch.load(ckpt_file, map_location="cpu", weights_only=False)
                    model.load_state_dict(_payload["model_state_dict"], assign=True)
                    del _payload
                    for _bn, _bt in getattr(model, "_meta_buffers", {}).items():
                        _pname, _, _leaf = _bn.rpartition(".")
                        _parent = model.get_submodule(_pname) if _pname else model
                        _parent._buffers[_leaf] = _bt
                    del model._meta_reload_path
                except Exception:
                    pass
                model.cpu()
                print(f"[mem] WARNING: meta parking unavailable ({_park_err}); parked on CPU", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                "backend": f"PyTorch+HF ({model_name}, mode={finetune_mode}, spurs_adapter={use_spurs_adapter})",'''

KEEP_OLD = '''            print(
                f"[Hill-Climb Val-Only {step_id}] ({eff_status}) {model_family} [{architecture_track}] "
'''
KEEP_NEW = '''            # Only the validation leader of each selection group can ever be
            # selected for test, so every other checkpoint is dead weight.
            # Drop them here (no step is training at this point) instead of
            # letting 51 x 2.6 GB accumulate on disk.
            try:
                import json as _json, pathlib as _pl
                _keep_groups = {}
                for _row in self.validation_rows:
                    _entry = self.entries_by_id.get(_row.get("step_id"))
                    if _entry is None or _entry.get("is_preregistered_baseline"):
                        continue
                    _cpaths = [q for q in str(_row.get("checkpoint_paths", "")).split(";") if q]
                    if not _cpaths:
                        continue
                    _reg = _row.get("regime")
                    _grp = (_reg, _row.get("architecture_track")) if _reg == "benchmark_all_239" else (_reg,)
                    _val = float(_row.get("val_median_spearman", float("-inf")))
                    if _grp not in _keep_groups or _val > _keep_groups[_grp][0]:
                        _keep_groups[_grp] = (_val, _cpaths)
                _keep_paths = sorted({q for _v, _ps in _keep_groups.values() for q in _ps})
                _pl.Path("/marimo/work/keep_ckpts.json").write_text(_json.dumps(_keep_paths))
                if _keep_paths:
                    _keep_set = set(_keep_paths)
                    _ckpt_dir = _pl.Path(_keep_paths[0]).parent
                    for _stale in _ckpt_dir.glob("*.pt"):
                        if str(_stale) not in _keep_set:
                            try:
                                _stale.unlink()
                            except Exception:
                                pass
            except Exception as _keep_err:
                print(f"[keep] WARNING: keep-list update failed: {_keep_err}", flush=True)

            # Crash resilience: persist the validation table after every step so
            # a late failure (e.g. in the test stage) never costs the hill-climb.
            try:
                import pathlib as _pl2
                _ck_dirs = [
                    _pl2.Path(q).parent
                    for q in str(checkpoint_paths[0]).split(";")
                    if q
                ] if checkpoint_paths else []
                if _ck_dirs:
                    _val_csv = _ck_dirs[0].parent / "hill_climb_validation_log.csv"
                    pd.DataFrame(self.validation_rows).to_csv(_val_csv, index=False)
            except Exception as _csv_err:
                print(f"[keep] WARNING: incremental validation CSV failed: {_csv_err}", flush=True)

            print(
                f"[Hill-Climb Val-Only {step_id}] ({eff_status}) {model_family} [{architecture_track}] "'''


PRETEST_OLD = '''    hill_climb_val_df = tracker.get_validation_log_df()
    final_test_df, per_prot_test_tables, paired_tests_list = tracker.evaluate_locked_selection_on_test_once('''
PRETEST_NEW = '''    hill_climb_val_df = tracker.get_validation_log_df()
    # Persist the validation table BEFORE the single locked test evaluation, so
    # a failure inside the test stage cannot destroy the experiment record.
    hill_climb_val_df.to_csv(_out_dir / "hill_climb_validation_log.csv", index=False)
    final_test_df, per_prot_test_tables, paired_tests_list = tracker.evaluate_locked_selection_on_test_once('''

# Winners-only mode: stage 03 runs only the full-FT sequence track (the locked
# sequence winner), stage 04 runs only the LoRA+SPURS structure track (the
# locked structure winner), and the supplementary regimes are skipped.
STAGE04_OLD = '''        for _smode in ["frozen", "lora"]:'''
STAGE04_NEW = '''        for _smode in ["lora"]:'''


def load_cell_nodes(source: str):
    tree = ast.parse(source)
    return [(n.name, n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("_cell_")]


def exec_cell(node, G):
    body = list(node.body)
    while body and isinstance(body[-1], ast.Return):
        body.pop()
    mod = ast.Module(body=body, type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), f"<{node.name}>", "exec"), G)


def shim(value):
    return SimpleNamespace(value=value)


def main(model_tag: str):
    is_150 = model_tag == "150"
    import os
    smoke = os.environ.get("ESM_SMOKE", "0") == "1"
    winners_only = os.environ.get("ESM_WINNERS_ONLY", "0") == "1"
    out_dir = (
        "/marimo/esm2_struct_smoke2"
        if smoke
        else f"/marimo/esm2_structure_track_{'150M' if is_150 else '650M'}"
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    status = Path(
        "/marimo/work/structure_track_smoke2.status"
        if smoke
        else f"/marimo/work/structure_track_{model_tag}.status"
    )
    started = time.time()
    status.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} START {out_dir}\n")

    src = NOTEBOOK.read_text()
    # marimo's runtime context does not exist in a plain Python process
    src = src.replace(
        '    _cli = mo.cli_args() if hasattr(mo, "cli_args") else {}',
        "    _cli = {}",
    )
    for _anchor_name, _anchor in (
        ("lora_lr", LORA_LR_PATCH_OLD),
        ("preflight", PREFLIGHT_OLD),
        ("grad_ckpt", GC_OLD),
        ("meta_park", CACHE_OLD),
        ("meta_reload", PREDICT_OLD),
        ("keep_list", KEEP_OLD),
        ("pretest_csv", PRETEST_OLD),
    ):
        _count = src.count(_anchor)
        print(f"anchor {_anchor_name}: {_count}", flush=True)
        if _count != 1:
            raise SystemExit(f"anchor {_anchor_name} occurs {_count} times, expected 1")
    if winners_only:
        _count = src.count(STAGE04_OLD)
        print(f"anchor stage04: {_count}", flush=True)
        if _count != 1:
            raise SystemExit(f"anchor stage04 occurs {_count} times, expected 1")

    if LORA_LR_PATCH_OLD in src:
        src = src.replace(LORA_LR_PATCH_OLD, LORA_LR_PATCH_NEW)
        print("applied lora_lr=1e-4 patch", flush=True)
    else:
        print("WARNING: lora_lr patch site not found", flush=True)

    if PREFLIGHT_OLD in src:
        src = src.replace(PREFLIGHT_OLD, PREFLIGHT_NEW)
        print("applied preflight-skip patch", flush=True)
    else:
        print("WARNING: preflight patch site not found", flush=True)

    if GC_OLD in src:
        src = src.replace(GC_OLD, GC_NEW)
        print("applied gradient-checkpointing patch", flush=True)
    else:
        print("WARNING: gradient-checkpointing patch site not found", flush=True)

    if CACHE_OLD in src:
        src = src.replace(CACHE_OLD, CACHE_NEW)
        print("applied meta-parking patch", flush=True)
    else:
        print("WARNING: meta-parking patch site not found", flush=True)

    if PREDICT_OLD in src:
        src = src.replace(PREDICT_OLD, PREDICT_NEW)
        print("applied meta-reload patch", flush=True)
    else:
        print("WARNING: meta-reload patch site not found", flush=True)

    if KEEP_OLD in src:
        src = src.replace(KEEP_OLD, KEEP_NEW)
        print("applied checkpoint keep-list patch", flush=True)
    else:
        print("WARNING: keep-list patch site not found", flush=True)

    if PRETEST_OLD in src:
        src = src.replace(PRETEST_OLD, PRETEST_NEW)
        print("applied pre-test CSV patch", flush=True)
    else:
        print("WARNING: pre-test CSV patch site not found", flush=True)

    if winners_only:
        src = src.replace(STAGE04_OLD, STAGE04_NEW)
        print("applied winners-only stage-04 patch", flush=True)

    cells = load_cell_nodes(src)
    names = [n for n, _ in cells]
    print("cells:", names, flush=True)

    G = {}
    for name, node in cells:
        if name == "_cell_execute_pipeline":
            break
        print(f"exec {name}", flush=True)
        exec_cell(node, G)

    overrides = {
        "bundle_path_ui": BUNDLE,
        "out_dir_ui": out_dir,
        "esm2_model_ui": (
            "facebook/esm2_t30_150M_UR50D" if is_150 else "facebook/esm2_t33_650M_UR50D"
        ),
        "esm2_650m_toggle_ui": False,
        "allow_smoke_surrogate_ui": False,
        "proteinmpnn_ckpt_ui": MPNN_CKPT,
        "enable_struct_adapter_ui": True,
        "finetune_modes_ui": ["frozen", "lora", "full"],
        "loss_fn_ui": "huber",
        "huber_beta_ui": 1.0,
        "lora_rank_ui": 16,
        "lora_alpha_ui": 32.0,
        "lora_dropout_ui": 0.05,
        "head_lr_ui": 1e-3,
        "backbone_lr_ui": 2e-5 if is_150 else 1e-5,
        "weight_decay_ui": 0.01,
        "batch_size_ui": 16,
        "epochs_ui": 100,
        "patience_ui": 20,
        "seeds_ui": SEEDS,
        "sign_mult_ui": 1.0,
        "cluster_thr_ui": 0.25,
        "run_leak_filter_exp_ui": True,
        "include_handcrafted_3d_ui": True,
        "raw_csv_recovery_ui": "",
        "thermompnn_preds_ui": THERMO_PREDS,
        "external_ddg_csv_ui": "",
        "device_ui": "cuda",
        "run_btn": True,
    }
    if smoke:
        overrides.update({
            "finetune_modes_ui": ["lora"],
            "epochs_ui": 1,
            "patience_ui": 1,
            "seeds_ui": "42",
            "run_leak_filter_exp_ui": False,
            "include_handcrafted_3d_ui": True,
        })
    if winners_only:
        overrides.update({
            "finetune_modes_ui": ["full"],
            "run_leak_filter_exp_ui": False,
            "include_handcrafted_3d_ui": False,
        })
    G.update({k: shim(v) for k, v in overrides.items()})

    try:
        exec_cell(dict(cells)["_cell_execute_pipeline"], G)
    except Exception:
        tb = traceback.format_exc()
        status.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} FAILED {out_dir}\n{tb[-4000:]}\n")
        print(tb, flush=True)
        raise

    ppt = G.get("per_prot_test_tables") or {}
    with open(Path(out_dir) / "per_prot_test_tables.pkl", "wb") as fh:
        pickle.dump(ppt, fh)
    for step_id, df in ppt.items():
        safe = step_id.replace("/", "_")
        df.to_csv(Path(out_dir) / f"per_protein__{safe}.csv", index=False)

    summary = {
        "model": model_tag,
        "out_dir": out_dir,
        "elapsed_seconds": round(time.time() - started, 1),
        "steps_with_per_protein_tables": list(ppt.keys()),
        "final_test_rows": int(len(G["final_test_df"])),
        "hill_climb_rows": int(len(G["hill_climb_val_df"])),
        "selected": {
            k: (v.get("step_id") if isinstance(v, dict) else None)
            for k, v in (G.get("locked_selection_manifest") or {}).items()
            if k.startswith("selected_")
        },
    }
    (Path(out_dir) / "run_summary.json").write_text(json.dumps(summary, indent=2))
    status.write_text(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} COMPLETE {out_dir} {summary['elapsed_seconds']}s\n"
    )
    print("COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "150")
