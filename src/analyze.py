"""Summarise results/exp.json: F1 drop per method and |offset|, before and after retuning.

usage: python src/analyze.py > results/summary.txt

A drop is a piece's F1 minus its own unshifted F1, averaged over +c and -c, so every
piece is compared with itself. The 95% intervals bootstrap over pieces, because the
offsets of one piece share its music and are not independent. "est err" is how far
the estimate is from the applied offset plus the piece's own tuning.
"""

import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = ROOT + "/results"
rows = json.load(open(f"{R}/exp.json"))
rng = np.random.default_rng(0)
P = sorted({r["piece"] for r in rows})
print("pieces", len(P), "rows", len(rows))
base = {
    r["piece"]: r["raw"] for r in rows if r["method"] == "resample" and r["shift"] == 0
}
print("baseline F1 %.2f" % np.mean(list(base.values())))
be = np.array(
    [r["base_est"] for r in rows if r["method"] == "resample" and r["shift"] == 0]
)
print(
    "measured tuning of original audio, cents: mean %+.1f sd %.1f min %+.1f max %+.1f, |x|>10: %d/%d"
    % (be.mean(), be.std(), be.min(), be.max(), (abs(be) > 10).sum(), len(be))
)
yr = defaultdict(list)
for r in rows:
    if r["method"] == "resample" and r["shift"] == 0:
        yr[r["year"]].append(r["base_est"])
print("by year", {k: [round(v, 1) for v in x] for k, x in sorted(yr.items())})


def ci(d):
    """Mean of d with a 95% bootstrap interval, as text."""
    d = np.array(d)
    b = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(4000)]
    return "%6.2f [%6.2f,%6.2f]" % (
        d.mean(),
        np.quantile(b, 0.025),
        np.quantile(b, 0.975),
    )


print(
    "\nmethod  |c|  raw-drop from baseline (CI)      retuned-drop (CI)        est err"
)
# stretch and 50 cents exist only for the first ten pieces, see exp.py
for m in ("resample", "stretch", "vocoder"):
    for c in (5, 10, 20, 35, 50):
        g = defaultdict(list)
        for r in rows:
            if r["method"] == m and abs(r["shift"]) == c:
                g[r["piece"]].append(r)
        # only pieces that have both signs
        k = [p for p in g if len(g[p]) == 2]
        if not k:
            continue
        a = [np.mean([x["raw"] for x in g[p]]) - base[p] for p in k]
        b = [np.mean([x["retuned"] for x in g[p]]) - base[p] for p in k]
        e = np.mean(
            [abs(x["est"] - x["shift"] - x["base_est"]) for p in k for x in g[p]]
        )
        print(f"{m:9s}{c:3d}  {ci(a)}  {ci(b)}  {e:.1f}")
# retuning must not hurt audio that was never shifted
c = [
    r["retuned"] - r["raw"]
    for r in rows
    if r["method"] == "resample" and r["shift"] == 0
]
print("\nretune on unshifted audio, F1 change", ci(c))
