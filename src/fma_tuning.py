"""How often are real recordings off concert pitch? Tuning of 30 s FMA-small clips, by genre.

FMA-small is popular and independent music, not piano, so this measures how often tuning offsets
occur in practice, not piano tuning. MP3 decoding and the estimator both add error of about 1-2 cents.
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import io, json, sys, zipfile, numpy as np, librosa, soundfile as sf
from multiprocessing import Pool
ZIP = os.environ.get("FMA_ZIP", ROOT + "/data/fma_small.zip")


def one(name):
    try:
        z = zipfile.ZipFile(ZIP); x, sr = sf.read(io.BytesIO(z.read(name)), dtype="float32", always_2d=True)
        y = librosa.resample(x.mean(1), orig_sr=sr, target_sr=16000, res_type="soxr_hq")
        return name, float(librosa.estimate_tuning(y=y, sr=16000, resolution=.005, bins_per_octave=12) * 100)
    except Exception:
        return name, None


if __name__ == "__main__":
    names = sorted(n for n in zipfile.ZipFile(ZIP).namelist() if n.endswith(".mp3"))
    with Pool(10) as p:
        res = [r for r in p.imap_unordered(one, names, chunksize=16) if r[1] is not None]
    json.dump(dict(res), open(ROOT + "/results/fma_tuning.json", "w"))
    t = np.abs(np.array([v for _, v in res]))
    print("tracks", len(t), "median |offset|", np.median(t), {f">{c}c": round(float((t > c).mean()) * 100, 1) for c in (5, 10, 20, 35, 45)})
