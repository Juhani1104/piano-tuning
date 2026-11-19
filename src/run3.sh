#!/bin/sh
cd "$(dirname "$0")/.."
P=${P:-python}
until grep -q '^done' results/get2.log; do sleep 10; done
$P -I src/smoke.py > results/smoke.txt 2>&1 || exit 1
$P -I src/exp3.py > results/exp3.log 2>&1
$P -I src/analyze3.py > results/summary3.txt 2>&1
