#!/bin/sh
cd "$(dirname "$0")/.."
P=${P:-python}
$P -I src/exp2.py > results/exp2.log 2>&1
$P -I src/analyze2.py > results/summary2.txt 2>&1
