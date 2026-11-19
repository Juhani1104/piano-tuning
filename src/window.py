"""How many seconds does the tuning estimate need? Error of estimate_tuning on short windows.

usage: python src/window.py > results/window.txt

Each of the 30 pieces is shifted by a known offset with a resample, and the estimate
is taken on windows of several lengths at random positions. The error is measured
against the applied offset plus the piece's own full-length estimate, so the piece's
true tuning cancels. Live transcription can only retune from the audio heard so far.
"""

import json
import os

import librosa
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 16000
WINDOWS = [1, 2, 5, 10, 20, 45]  # seconds
SHIFTS = [-35, -20, -10, 10, 20, 35]
DRAWS = 5  # random window positions per piece, offset and length
pick = json.load(open(f"{ROOT}/data/pick.json"))
rng = np.random.default_rng(0)


def est(x):
    """Tuning offset of x from A440 in cents, as in exp.py."""
    return float(
        librosa.estimate_tuning(y=x, sr=SR, resolution=0.005, bins_per_octave=12) * 100
    )


err = {w: [] for w in WINDOWS}
for r in pick:
    x, _ = librosa.load(
        f"{ROOT}/data/maestro/{r['audio_filename']}", sr=SR, mono=True, duration=93
    )
    base = est(x)
    for c in SHIFTS:
        ratio = 2 ** (c / 1200)
        y = librosa.resample(x, orig_sr=SR * ratio, target_sr=SR, res_type="soxr_hq")
        for w in WINDOWS:
            n = w * SR
            for _ in range(DRAWS):
                s = rng.integers(0, len(y) - n)
                e = est(y[s : s + n]) - c - base
                # wrap to +-50, the estimator's own range
                err[w].append((e + 50) % 100 - 50)
print("window_s  median_abs  p90_abs  share_over_5c  share_over_10c")
for w in WINDOWS:
    a = np.abs(np.array(err[w]))
    print(
        f"{w:8d}  {np.median(a):10.2f}  {np.quantile(a, 0.9):7.2f}"
        f"  {(a > 5).mean():13.3f}  {(a > 10).mean():14.3f}"
    )
