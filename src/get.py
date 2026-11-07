"""Fetch a year-stratified sample of the MAESTRO test split by range requests."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import json, os, numpy as np
from collections import defaultdict
from remotezip import RemoteZip
R = ROOT + "/data"
rows = [r for r in json.load(open(f"{R}/maestro.json")) if r["split"] == "test"]
by = defaultdict(list)
for r in rows:
    by[r["year"]].append(r)
rng = np.random.default_rng(0); pick = []
for y in sorted(by):
    i = rng.permutation(len(by[y]))[:3]; pick += [by[y][k] for k in i]
json.dump(pick, open(f"{R}/pick.json", "w"), indent=0)
z = RemoteZip("https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip")
for k, r in enumerate(pick):
    for f in (r["audio_filename"], r["midi_filename"]):
        out = f"{R}/maestro/{f}"; os.makedirs(os.path.dirname(out), exist_ok=True)
        if not os.path.exists(out):
            d = z.read("maestro-v3.0.0/" + f); open(out + ".tmp", "wb").write(d); os.replace(out + ".tmp", out)
    print(k, r["year"], flush=True)
print("done", flush=True)
