"""Fetch the third-party checkpoints and datasets the pipeline needs.

    python setup_env.py              # everything
    python setup_env.py --only mpnn thermompnn
    python setup_env.py --check      # report what is present, download nothing

What it fetches, into the locations `paths.py` resolves:

    mpnn         ProteinMPNN repo + the v_48_020 vanilla checkpoint
    thermompnn   ThermoMPNN repo + dataset_splits/ (mega_splits.pkl and the homology tables)
    fireprot     FireProtDB from Zenodo (record 8169288) -- needed only for round 3
    s669         the S669 and Ssym CSVs, plus the 94 S669 PDBs from RCSB -- transfer test

The Tsuboyama training bundle is NOT fetched: it is not redistributable here. Point
ESM2_BUNDLE at your own copy. Everything else is public and downloaded on demand.

Already-present files are left alone, so this is safe to re-run after an interrupted setup.
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import paths

MPNN_REPO = "https://github.com/dauparas/ProteinMPNN"
THERMO_REPO = "https://github.com/Kuhlman-Lab/ThermoMPNN"
MPNN_WEIGHT_URL = "https://raw.githubusercontent.com/dauparas/ProteinMPNN/main/vanilla_model_weights/v_48_020.pt"
THERMO_RAW = "https://raw.githubusercontent.com/Kuhlman-Lab/ThermoMPNN/main/"
FIREPROT_ZIP = "https://zenodo.org/api/records/8169289/files/fireprot_upload.zip/content"
RCSB = "https://files.rcsb.org/download/{}.pdb"

SPLIT_FILES = [
    "dataset_splits/mega_splits.pkl",
    "dataset_splits/fireprot_splits.pkl",
    "dataset_splits/megascale_vs_fireprot_homologues.m8",
    "dataset_splits/s669_vs_megascale_homologues.m8",
    "dataset_splits/fireprot_clusters.tsv",
    "dataset_splits/megascale_clusters.tsv",
]
S669_CSVS = ["s669_clean_dir.csv", "ssym-5fold_clean_dir.csv", "ssym-5fold_clean_inv.csv"]


def fetch(url: str, dest: Path, label: str = "") -> bool:
    """Download `url` to `dest` unless it is already there. True if the file ends up present."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        print(f"    failed: {label or dest.name}: {type(exc).__name__} {exc}")
        return False
    print(f"    {label or dest.name}  ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


def clone(repo: str, dest: Path) -> None:
    if (dest / ".git").exists() or any(dest.glob("*.py")):
        print(f"    {dest.name} already present")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", repo, str(dest)], check=True)


def setup_mpnn() -> None:
    print("ProteinMPNN")
    clone(MPNN_REPO, paths.PROTEINMPNN_DIR)
    fetch(MPNN_WEIGHT_URL, paths.MPNN_WEIGHTS / "v_48_020.pt", "v_48_020.pt")


def setup_thermompnn() -> None:
    print("ThermoMPNN")
    clone(THERMO_REPO, paths.THERMOMPNN_DIR)
    # the clone does not always carry dataset_splits/, and TransferModel looks for the
    # ProteinMPNN checkpoint under its own tree
    for rel in SPLIT_FILES:
        fetch(THERMO_RAW + rel, paths.THERMOMPNN_DIR / rel, rel)
    fetch(MPNN_WEIGHT_URL, paths.THERMOMPNN_DIR / "vanilla_model_weights" / "v_48_020.pt",
          "v_48_020.pt (for TransferModel)")


def setup_fireprot() -> None:
    print("FireProtDB")
    root = paths.FIREPROT_DIR.parent
    zp = root / "fireprot_upload.zip"
    if not fetch(FIREPROT_ZIP, zp, "fireprot_upload.zip"):
        return
    if not paths.FIREPROT_CSV.exists():
        zipfile.ZipFile(zp).extractall(root)
        print(f"    extracted to {root}")


def setup_s669() -> None:
    print("S669 / Ssym")
    for name in S669_CSVS:
        fetch(THERMO_RAW + "data_all/testing/" + name, paths.S669_DIR / name, name)
    if not paths.S669_CSV.exists():
        print("    no S669 csv; skipping structures")
        return
    import pandas as pd

    ids = sorted({t[:-1] for t in pd.read_csv(paths.S669_CSV).PDB.unique()})
    have = [p for p in ids if (paths.S669_PDBS / f"{p}.pdb").exists()]
    print(f"    {len(ids)} structures needed, {len(have)} already present")
    bad = 0
    for pid in ids:
        if not fetch(RCSB.format(pid), paths.S669_PDBS / f"{pid}.pdb"):
            bad += 1
    print(f"    {len(ids) - bad}/{len(ids)} structures available")


STEPS = {"mpnn": setup_mpnn, "thermompnn": setup_thermompnn,
         "fireprot": setup_fireprot, "s669": setup_s669}


def check() -> int:
    required = {
        "training bundle (set ESM2_BUNDLE)": paths.BUNDLE,
        "ProteinMPNN checkpoint": paths.MPNN_WEIGHTS / "v_48_020.pt",
        "ThermoMPNN source": paths.THERMOMPNN_DIR / "transfer_model.py",
        "mega_splits.pkl": paths.SPLITS_PKL,
    }
    optional = {
        "FireProtDB csv (round 3)": paths.FIREPROT_CSV,
        "S669 csv (transfer test)": paths.S669_CSV,
        "S669 structures": paths.S669_PDBS,
    }
    print(paths.describe())
    missing = 0
    for group, items in (("required", required), ("optional", optional)):
        print(f"\n{group}:")
        for label, p in items.items():
            ok = p.exists()
            if group == "required" and not ok:
                missing += 1
            print(f"  [{'x' if ok else ' '}] {label:<36} {p}")
    print()
    if missing:
        print(f"{missing} required input(s) missing -- run `python setup_env.py`")
    else:
        print("ready")
    return 1 if missing else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", choices=sorted(STEPS), help="run just these steps")
    ap.add_argument("--check", action="store_true", help="report what is present, download nothing")
    args = ap.parse_args()

    if args.check:
        return check()

    print(paths.describe(), "\n")
    paths.ensure_dirs()
    for name in (args.only or list(STEPS)):
        STEPS[name]()
    print()
    return check()


if __name__ == "__main__":
    sys.exit(main())
