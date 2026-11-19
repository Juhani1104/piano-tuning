"""Download the MAESTRO v3 test pieces used by exp.py and exp2.py.

usage: python src/get.py

Reads data/maestro.json (the rows of maestro-v3.0.0.csv) and picks three test pieces
from each of the ten recording years with a fixed seed, so every year and piano setup
is covered. The pick is written to data/pick.json. Each WAV and MIDI file is read out
of the 108 GB MAESTRO zip with HTTP range requests, so the whole archive is never
downloaded; a file already on disk is skipped, so the download resumes.
"""

import json
import os
from collections import defaultdict

import numpy as np
from remotezip import RemoteZip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = ROOT + "/data"
URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
PER_YEAR = 3

rows = [r for r in json.load(open(f"{R}/maestro.json")) if r["split"] == "test"]
by = defaultdict(list)
for r in rows:
    by[r["year"]].append(r)
rng = np.random.default_rng(0)
pick = []
for y in sorted(by):
    i = rng.permutation(len(by[y]))[:PER_YEAR]
    pick += [by[y][k] for k in i]
json.dump(pick, open(f"{R}/pick.json", "w"), indent=0)

z = RemoteZip(URL)
for k, r in enumerate(pick):
    for f in (r["audio_filename"], r["midi_filename"]):
        out = f"{R}/maestro/{f}"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if not os.path.exists(out):
            # write to a temporary name first, so an interrupted file is never kept
            d = z.read("maestro-v3.0.0/" + f)
            open(out + ".tmp", "wb").write(d)
            os.replace(out + ".tmp", out)
    print(k, r["year"], flush=True)
print("done", flush=True)
