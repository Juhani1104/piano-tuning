"""Download Edwards et al.'s augmented model and the Studio MAESTRO recordings of our pieces.

usage: python src/get2.py

Studio MAESTRO re-performs MAESTRO on a Disklavier in a studio, so its MIDI is the
original MAESTRO MIDI and its tuning is set by another piano. The 16 kHz mono mix is
read out of its 18 GB zip with range requests, only for the pieces in data/pick.json.
"""

import json
import os
import urllib.request

from remotezip import RemoteZip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = ROOT + "/data"
WEIGHTS = "https://zenodo.org/api/records/10610212/files/high_resolution_MAESTRO_augmentations.pth/content"
STUDIO = "https://zenodo.org/records/10082144/files/studio_maestro.zip?download=1"

out = f"{R}/edwards_aug.pth"
if not os.path.exists(out):
    urllib.request.urlretrieve(WEIGHTS, out + ".tmp")
    os.replace(out + ".tmp", out)
print("weights", os.path.getsize(out), flush=True)

z = RemoteZip(STUDIO)
for k, name in enumerate(json.load(open(f"{R}/studio_pick.json"))):
    dst = f"{R}/studio/{os.path.basename(name)}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        open(dst + ".tmp", "wb").write(z.read(name))
        os.replace(dst + ".tmp", dst)
    print(k, flush=True)
print("done", flush=True)
