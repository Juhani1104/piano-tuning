import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np
from collections import defaultdict
rows = json.load(open(ROOT + "/results/exp2.json")); rng = np.random.default_rng(0)
for mdl in ("transkun", "basicpitch"):
    R = [r for r in rows if r["model"] == mdl]; P = sorted({r["piece"] for r in R})
    base = {r["piece"]: r["raw"] for r in R if r["method"] == "resample" and r["shift"] == 0}
    print(f"\n{mdl}: pieces {len(P)} baseline F1 {np.mean(list(base.values())):.2f}")
    for m, cs in (("resample", (10, 20, 35)), ("vocoder", (5, 10))):
        for c in cs:
            g = defaultdict(list)
            for r in R:
                if r["method"] == m and abs(r["shift"]) == c:
                    g[r["piece"]].append(r)
            k = [p for p in g if len(g[p]) == 2 and p in base]
            def ci(v):
                v = np.array(v); b = [v[rng.integers(len(v), size=len(v))].mean() for _ in range(4000)]
                return "%6.2f [%6.2f,%6.2f]" % (v.mean(), np.quantile(b, .025), np.quantile(b, .975))
            print(f"  {m:8s}{c:3d} raw drop {ci([np.mean([x['raw'] for x in g[p]]) - base[p] for p in k])} retuned {ci([np.mean([x['retuned'] for x in g[p]]) - base[p] for p in k])}")
    z = [r["retuned"] - r["raw"] for r in R if r["method"] == "resample" and r["shift"] == 0]
    print("  retune on unshifted %.2f" % np.mean(z))
