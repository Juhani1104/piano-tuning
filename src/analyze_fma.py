"""Summarise results/fma_tuning.json: how far real songs sit from A4 = 440 Hz.

usage: python src/analyze_fma.py > results/summary_fma.txt

Each FMA small clip has one librosa.estimate_tuning value in cents. The shares say how
often a song is off by more than the offsets tested in exp.py. Estimates wrap at +-50
cents and the estimator is not validated on mixed, non-piano music, so the larger
shares are upper bounds.
"""

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
v = np.array(list(json.load(open(f"{ROOT}/results/fma_tuning.json")).values()), float)
v = v[np.isfinite(v)]
print(
    f"clips {len(v)}  median {np.median(v):+.1f} cents  mean |offset| {abs(v).mean():.1f}"
)
for t in (5, 10, 20, 35):
    print(f"|offset| > {t:2d} cents: {100 * np.mean(abs(v) > t):5.1f}%")
print(f"sharp by 5-20 cents: {100 * np.mean((v >= 5) & (v <= 20)):5.1f}%")
print(f"flat by 5-20 cents:  {100 * np.mean((v <= -5) & (v >= -20)):5.1f}%")

# common concert pitches, as offsets from A4 = 440 Hz
for hz in (415, 432, 442, 443):
    print(f"A4 = {hz} Hz is {1200 * np.log2(hz / 440):+6.1f} cents")
