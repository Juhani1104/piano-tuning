"""The tuning offset test of exp.py on two more transcribers: Transkun 2.0 and Basic Pitch.

usage: python src/exp2.py

Transkun is a piano model and Basic Pitch is a general, instrument-agnostic one, so
the pair shows whether the result depends on the model. Ten pieces are used, every
third entry of data/pick.json, which is one per recording year. Offsets are made at
44.1 kHz and each model resamples to its own rate; retuning uses the estimator of
exp.py, run at 16 kHz. Rows are appended to results/exp2.json one piece at a time,
and pieces already there are skipped.
"""

import json
import os
import tempfile

import librosa
import mir_eval
import moduleconf
import numpy as np
import pretty_midi
import soundfile as sf
import torch
import transkun
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import Model, predict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = ROOT
DUR = 90
SR = 44100
TMP = tempfile.mkdtemp()  # Basic Pitch reads audio from a file
pick = json.load(open(f"{R}/data/pick.json"))[::3]
COND = [
    ("resample", 0),
    ("resample", 10),
    ("resample", -10),
    ("resample", 20),
    ("resample", -20),
    ("resample", 35),
    ("resample", -35),
    ("vocoder", 5),
    ("vocoder", -5),
    ("vocoder", 10),
    ("vocoder", -10),
]

# Transkun ships its weights and config inside the package, loaded as its own CLI does
d = os.path.dirname(transkun.__file__)
c = moduleconf.parseFromFile(f"{d}/pretrained/2.0.conf")
TK = c["Model"].module.TransKun(conf=c["Model"].config)
ck = torch.load(f"{d}/pretrained/2.0.pt", map_location="cpu", weights_only=False)
TK.load_state_dict(ck.get("best_state_dict", ck.get("state_dict")), strict=False)
TK = TK.cuda().eval()
torch.set_grad_enabled(False)
BP = Model(ICASSP_2022_MODEL_PATH)


def run_tk(x):
    """Onset times and MIDI pitches from Transkun; mono input is given as two channels."""
    y = (
        librosa.resample(x, orig_sr=SR, target_sr=TK.fs, res_type="soxr_hq")
        if TK.fs != SR
        else x
    )
    n = TK.transcribe(torch.from_numpy(np.stack([y, y], 1).astype("float32")).cuda())
    # events with pitch <= 0 are pedals
    n = [v for v in n if v.pitch > 0]
    return np.array([v.start for v in n]), np.array([v.pitch for v in n])


def run_bp(x):
    """Onset times and MIDI pitches from Basic Pitch with its default thresholds."""
    f = f"{TMP}/bp.wav"
    sf.write(f, x, SR)
    _, _, ev = predict(f, BP)
    return np.array([e[0] for e in ev]), np.array([e[2] for e in ev])


def shift(x, m, c):
    """Shift x by c cents with method m; also returns the factor for GT onset times."""
    r = 2 ** (c / 1200)
    if m == "resample":
        return (
            librosa.resample(x, orig_sr=SR * r, target_sr=SR, res_type="soxr_hq"),
            1 / r,
        )
    return librosa.effects.pitch_shift(x, sr=SR, n_steps=c / 100), 1.0


def est(x):
    """Tuning offset of x from A440 in cents, estimated at 16 kHz as in exp.py."""
    y = librosa.resample(x, orig_sr=SR, target_sr=16000)
    return float(
        librosa.estimate_tuning(y=y, sr=16000, resolution=0.005, bins_per_octave=12)
        * 100
    )


def f1(on, pi, ref, t):
    """Onset F1 (%) of onsets on and pitches pi against ref, its onsets scaled by t."""
    ok = on < DUR - 1
    ri = ref[0] * t
    keep = ri < DUR - 1
    return (
        mir_eval.transcription.precision_recall_f1_overlap(
            np.c_[ri[keep], ri[keep] + 0.1],
            ref[1][keep],
            np.c_[on[ok], on[ok] + 0.1],
            librosa.midi_to_hz(pi[ok]),
            offset_ratio=None,
            onset_tolerance=0.05,
        )[2]
        * 100
    )


out = f"{R}/results/exp2.json"
rows = json.load(open(out)) if os.path.exists(out) else []
seen = {x["piece"] for x in rows}
for k, r in enumerate(pick):
    if k in seen:
        continue
    x, _ = librosa.load(
        f"{R}/data/maestro/{r['audio_filename']}", sr=SR, mono=True, duration=DUR + 3
    )
    n = sorted(
        pretty_midi.PrettyMIDI(f"{R}/data/maestro/{r['midi_filename']}")
        .instruments[0]
        .notes,
        key=lambda v: v.start,
    )
    ref = (
        np.array([v.start for v in n]),
        librosa.midi_to_hz(np.array([v.pitch for v in n])),
    )
    for m, cc in COND:
        y, t = shift(x, m, cc)
        e = est(y)
        z, t2 = shift(y, "resample", -e)
        for name, fn in (("transkun", run_tk), ("basicpitch", run_bp)):
            a = f1(*fn(y), ref, t)
            b = f1(*fn(z), ref, t * t2)
            rows.append(
                dict(
                    piece=k,
                    year=r["year"],
                    model=name,
                    method=m,
                    shift=cc,
                    est=e,
                    raw=a,
                    retuned=b,
                )
            )
            print(
                k, r["year"], name, m, cc, "est", round(e, 1),
                "F1", round(a, 2), "retuned", round(b, 2), flush=True,
            )  # fmt: skip
    json.dump(rows, open(out, "w"), indent=0)
print("done", flush=True)
