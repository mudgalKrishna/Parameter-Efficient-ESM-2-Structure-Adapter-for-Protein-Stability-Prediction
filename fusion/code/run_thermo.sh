#!/bin/bash
cd /marimo/work
: > logs_thermo_train.txt
for sp in mega cv0 cv1 cv2 cv3 cv4; do
  for sd in 0 1 2 3 4; do
    echo "=== $sp seed $sd" >> logs_thermo_train.txt
    python thermo_train.py --split "$sp" --seed "$sd" --epochs 100 --patience 15 >> logs_thermo_train.txt 2>&1
  done
done
echo ALLDONE >> logs_thermo_train.txt
