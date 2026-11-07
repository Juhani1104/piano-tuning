"""Summarise exp.json with a bootstrap over pieces, grouped by |shift|."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np
from collections import defaultdict
R = ROOT + "/results"
rows = json.load(open(f"{R}/exp.json")); rng = np.random.default_rng(0)
P = sorted({r["piece"] for r in rows}); print("pieces", len(P), "rows", len(rows))
base = {r["piece"]: r["raw"] for r in rows if r["method"] == "resample" and r["shift"] == 0}
print("baseline F1 %.2f" % np.mean(list(base.values())))
be = np.array([r["base_est"] for r in rows if r["method"] == "resample" and r["shift"] == 0])
print("measured tuning of original audio, cents: mean %+.1f sd %.1f min %+.1f max %+.1f, |x|>10: %d/%d" % (be.mean(), be.std(), be.min(), be.max(), (abs(be) > 10).sum(), len(be)))
yr = defaultdict(list)
for r in rows:
    if r["method"] == "resample" and r["shift"] == 0:
        yr[r["year"]].append(r["base_est"])
print("by year", {k: [round(v, 1) for v in x] for k, x in sorted(yr.items())})


def ci(d):
    d = np.array(d); b = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(4000)]
    return "%6.2f [%6.2f,%6.2f]" % (d.mean(), np.quantile(b, .025), np.quantile(b, .975))


print("\nmethod  |c|  raw-drop from baseline (CI)      retuned-drop (CI)        est err")
for m in ("resample", "stretch", "vocoder"):
    for c in (5, 10, 20, 35, 50):
        g = defaultdict(list)
        for r in rows:
            if r["method"] == m and abs(r["shift"]) == c:
                g[r["piece"]].append(r)
        k = [p for p in g if len(g[p]) == 2]
        if not k:
            continue
        a = [np.mean([x["raw"] for x in g[p]]) - base[p] for p in k]
        b = [np.mean([x["retuned"] for x in g[p]]) - base[p] for p in k]
        e = np.mean([abs(x["est"] - x["shift"] - x["base_est"]) for p in k for x in g[p]])
        print(f"{m:9s}{c:3d}  {ci(a)}  {ci(b)}  {e:.1f}")
# clean audio: does retuning hurt?
c = [r["retuned"] - r["raw"] for r in rows if r["method"] == "resample" and r["shift"] == 0]
print("\nretune on unshifted audio, F1 change", ci(c))
