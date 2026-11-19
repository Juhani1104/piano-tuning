"""How far real recordings sit from A440: the tuning offset of every FMA-small clip.

usage: python src/fma_tuning.py        (FMA_ZIP=<fma_small.zip>, default data/)

FMA-small is 8000 30 s clips of popular and independent music, not piano, so this
measures how often offsets occur in practice, not how pianos are tuned. The estimate
is the one exp.py uses for retuning. It is not checked on this music: unpitched or
non-12-TET material and offsets near +-50 cents, which fold to the other side, all
end up in the counts, so the shares above 35 cents are upper bounds. Clips that fail
to decode are skipped. Writes results/fma_tuning.json, clip name to cents.
"""

import io
import json
import os
import zipfile
from multiprocessing import Pool

import librosa
import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP = os.environ.get("FMA_ZIP", ROOT + "/data/fma_small.zip")


def one(name):
    """(name, offset in cents) for one clip, or (name, None) if it does not decode."""
    try:
        # each worker opens the zip itself; a ZipFile cannot be shared across processes
        z = zipfile.ZipFile(ZIP)
        x, sr = sf.read(io.BytesIO(z.read(name)), dtype="float32", always_2d=True)
        y = librosa.resample(x.mean(1), orig_sr=sr, target_sr=16000, res_type="soxr_hq")
        return name, float(
            librosa.estimate_tuning(y=y, sr=16000, resolution=0.005, bins_per_octave=12)
            * 100
        )
    except Exception:
        return name, None


if __name__ == "__main__":
    names = sorted(n for n in zipfile.ZipFile(ZIP).namelist() if n.endswith(".mp3"))
    with Pool(10) as p:
        res = [
            r for r in p.imap_unordered(one, names, chunksize=16) if r[1] is not None
        ]
    json.dump(dict(res), open(ROOT + "/results/fma_tuning.json", "w"))
    t = np.abs(np.array([v for _, v in res]))
    print(
        "tracks", len(t), "median |offset|", np.median(t),
        {f">{c}c": round(float((t > c).mean()) * 100, 1) for c in (5, 10, 20, 35, 45)},
    )  # fmt: skip
