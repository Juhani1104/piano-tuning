"""Summarise results/exp3.json: studio tuning and retuning gains, then the synthetic breakdown.

usage: python src/analyze3.py > results/summary3.txt

Studio gains are paired per piece (retuned F1 minus raw F1) with a bootstrap over
pieces. In the synthetic part every number is the mean over the +c and -c runs of a
piece, minus that piece's unshifted value, so P, R and the error counts are changes.
"""

import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = json.load(open(f"{ROOT}/results/exp3.json"))
rng = np.random.default_rng(0)


def ci(v):
    """Mean of v with a 95% bootstrap interval over pieces, as text."""
    v = np.array(v, float)
    b = [v[rng.integers(len(v), size=len(v))].mean() for _ in range(4000)]
    return "%6.2f [%6.2f,%6.2f]" % (
        v.mean(),
        np.quantile(b, 0.025),
        np.quantile(b, 0.975),
    )


st = [r for r in rows if r["stage"] == "studio"]
e = np.array([r["est"] for r in st if r["model"] == "kong"])
print(
    "studio pieces",
    len(e),
    "tuning cents: mean %+.1f sd %.1f min %+.1f max %+.1f"
    % (e.mean(), e.std(), e.min(), e.max()),
)
for m in ("kong", "edwards"):
    R = [r for r in st if r["model"] == m]
    print(f"  {m:8s} raw F1 {np.mean([r['raw']['F'] for r in R]):.2f}  retuned {np.mean([r['retuned']['F'] for r in R]):.2f}"
          f"  gain {ci([r['retuned']['F'] - r['raw']['F'] for r in R])}"
          f"  pieces improved {sum(r['retuned']['F'] > r['raw']['F'] for r in R)}/{len(R)}")  # fmt: skip
    print(
        f"           P gain {ci([r['retuned']['P'] - r['raw']['P'] for r in R])}  R gain {ci([r['retuned']['R'] - r['raw']['R'] for r in R])}"
    )
kr = {r["piece"]: r for r in st if r["model"] == "kong"}
er = {r["piece"]: r for r in st if r["model"] == "edwards"}
print(
    "  retuned Kong minus raw Edwards",
    ci([kr[p]["retuned"]["F"] - er[p]["raw"]["F"] for p in kr if p in er]),
)

sy = [r for r in rows if r["stage"] == "synth"]
KEYS = ["F", "P", "R", "n_est", "dev_med", "fp_semitone", "fp_octave", "fp_other"]
for m in ("kong", "edwards"):
    R = [r for r in sy if r["model"] == m]
    base = {r["piece"]: r for r in R if r["method"] == "resample" and r["shift"] == 0}
    print(f"\nsynth {m}: pieces {len(base)} baseline " + " ".join(
        f"{k} {np.mean([b['raw'][k] for b in base.values()]):.2f}" for k in KEYS))  # fmt: skip
    for meth, cs in (("resample", (10, 20, 35)), ("vocoder", (5, 10))):
        for c in cs:
            g = defaultdict(list)
            for r in R:
                if r["method"] == meth and abs(r["shift"]) == c:
                    g[r["piece"]].append(r)
            k = [p for p in g if len(g[p]) == 2 and p in base]
            for when in ("raw", "retuned"):
                d = {
                    key: [
                        np.mean([x[when][key] for x in g[p]]) - base[p]["raw"][key]
                        for p in k
                    ]
                    for key in KEYS
                }
                print(f"  {meth:8s}{c:3d} {when:7s} dF {ci(d['F'])} dP {np.mean(d['P']):+6.2f} dR {np.mean(d['R']):+6.2f}"
                      f" dN {np.mean(d['n_est']):+6.1f} dDev {np.mean(d['dev_med']):+5.2f}ms"
                      f" dSemi {np.mean(d['fp_semitone']):+6.1f} dOct {np.mean(d['fp_octave']):+5.1f} dOther {np.mean(d['fp_other']):+6.1f}")  # fmt: skip
