"""Filesystem layout for the pipeline, resolved once from the environment.

Every path any script in this directory touches is defined here, so the pipeline can run
anywhere. Nothing is hardcoded to the machine it was developed on.

Environment variables, all optional:

    ESM2_WORK     scratch root: run outputs, caches, derived tensors   (default: ./work)
    ESM2_BUNDLE   the Tsuboyama training bundle, a .zip               (default: $ESM2_WORK/training_bundle.zip)
    ESM2_EXT      parent of the ProteinMPNN and ThermoMPNN clones     (default: $ESM2_WORK/ext)
    ESM2_DATA     downloaded external datasets (FireProtDB, S669)     (default: $ESM2_WORK/data)

A minimal setup is therefore::

    export ESM2_WORK=/path/to/scratch
    export ESM2_BUNDLE=/path/to/training_bundle.zip
    python setup_env.py          # fetches everything else

`setup_env.py` populates `ESM2_EXT` and `ESM2_DATA`. Call `require()` on an input before
reading it to get an actionable error instead of a bare FileNotFoundError.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "WORK", "BUNDLE", "EXT", "DATA",
    "PROTEINMPNN_DIR", "MPNN_WEIGHTS", "THERMOMPNN_DIR", "SPLITS_PKL",
    "FIREPROT_DIR", "FIREPROT_CSV", "FIREPROT_SPLITS", "FIREPROT_PDBS",
    "S669_DIR", "S669_CSV", "S669_PDBS", "SSYM_DIR",
    "LOGS", "PDB_DIR", "PDB_S669_DIR", "RUNS_THERMO",
    "FP_TENSORS", "S669_TENSORS", "FIGURE_PNG",
    "runs_dir", "require", "ensure_dirs", "describe",
]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


# --- roots -------------------------------------------------------------------------------
WORK = _env_path("ESM2_WORK", Path.cwd() / "work")
BUNDLE = _env_path("ESM2_BUNDLE", WORK / "training_bundle.zip")
EXT = _env_path("ESM2_EXT", WORK / "ext")
DATA = _env_path("ESM2_DATA", WORK / "data")

# --- third-party checkpoints and splits --------------------------------------------------
PROTEINMPNN_DIR = EXT / "ProteinMPNN"
MPNN_WEIGHTS = PROTEINMPNN_DIR / "vanilla_model_weights"
THERMOMPNN_DIR = EXT / "ThermoMPNN"
SPLITS_PKL = THERMOMPNN_DIR / "dataset_splits" / "mega_splits.pkl"

# --- external datasets -------------------------------------------------------------------
FIREPROT_DIR = DATA / "fireprot" / "fireprot_upload"
FIREPROT_CSV = FIREPROT_DIR / "csvs" / "4_fireprotDB_bestpH.csv"
FIREPROT_SPLITS = FIREPROT_DIR / "csvs" / "fireprot_splits.pkl"
FIREPROT_PDBS = FIREPROT_DIR / "pdbs"

S669_DIR = DATA / "s669"
S669_CSV = S669_DIR / "s669_clean_dir.csv"
S669_PDBS = S669_DIR / "pdbs"
SSYM_DIR = S669_DIR

# --- derived artefacts ------------------------------------------------------------------
LOGS = WORK / "logs"
PDB_DIR = WORK / "pdbs"                  # per-protein PDBs unpacked from the bundle
PDB_S669_DIR = WORK / "pdbs_s669"        # single-chain PDBs written for S669
RUNS_THERMO = WORK / "runs_thermo"       # retrained ThermoMPNN baselines
FP_TENSORS = WORK / "fireprot_tensors.npz"
S669_TENSORS = WORK / "s669_tensors.npz"
FIGURE_PNG = WORK / "results.png"


def runs_dir(group: str, tag: str | None = None) -> Path:
    """Output directory for a run group ("runs2", "runs3", ...), optionally per tag."""
    p = WORK / group
    return p / tag if tag else p


def require(*paths: Path | str) -> None:
    """Fail early, and say what to do about it, if a required input is absent."""
    missing = [Path(p) for p in paths if not Path(p).exists()]
    if not missing:
        return
    lines = ["Required input(s) not found:"]
    lines += [f"  {p}" for p in missing]
    lines.append("")
    lines.append("Set ESM2_WORK / ESM2_BUNDLE / ESM2_EXT / ESM2_DATA to match your layout,")
    lines.append("or run `python setup_env.py` to fetch the third-party checkpoints and datasets.")
    raise FileNotFoundError("\n".join(lines))


def ensure_dirs() -> None:
    """Create the scratch directories that outputs are written into."""
    for d in (WORK, LOGS, PDB_DIR, RUNS_THERMO, DATA, EXT):
        d.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    """Human-readable resolution of the layout, for logging at the top of a run."""
    rows = [("WORK", WORK), ("BUNDLE", BUNDLE), ("EXT", EXT), ("DATA", DATA)]
    width = max(len(k) for k, _ in rows)
    return "\n".join(f"{k:<{width}} = {v}" + ("" if Path(v).exists() else "   [missing]")
                     for k, v in rows)


if __name__ == "__main__":
    print(describe())
