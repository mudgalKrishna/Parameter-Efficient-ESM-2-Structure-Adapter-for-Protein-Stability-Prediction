#!/bin/bash
# Retrain the ThermoMPNN baseline: 6 splits x 5 seeds.
# Honours ESM2_WORK / ESM2_BUNDLE / ESM2_EXT; see paths.py.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="${ESM2_WORK:-$PWD/work}/logs_thermo_train.txt"
mkdir -p "$(dirname "$LOG")"
: > "$LOG"
for sp in mega cv0 cv1 cv2 cv3 cv4; do
  for sd in 0 1 2 3 4; do
    echo "=== $sp seed $sd" >> "$LOG"
    python "$HERE/thermo_train.py" --split "$sp" --seed "$sd" --epochs 100 --patience 15 >> "$LOG" 2>&1
  done
done
echo ALLDONE >> "$LOG"
