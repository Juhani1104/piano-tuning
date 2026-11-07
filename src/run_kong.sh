#!/bin/sh
cd "$(dirname "$0")/.."
P=${P:-python}
$P -I src/exp.py > results/exp.log 2>&1
$P -I src/analyze.py > results/summary.txt 2>&1
