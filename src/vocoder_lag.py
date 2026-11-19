"""Is the vocoder's damage a constant delay or smeared onsets?

usage: python src/vocoder_lag.py > results/vocoder_lag.txt

First, the delay librosa.effects.pitch_shift adds to isolated clicks, which a constant
lag would show directly. Second, Kong on the vocoder audio of exp3's ten pieces scored
with the best constant lag in +-60 ms, so a pure delay would give back most of the F1.
"""

import json
import os
import sys

import librosa
import mir_eval
import numpy as np
import pretty_midi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import load  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 16000
DUR = 90
LAGS = np.arange(-0.06, 0.0601, 0.002)

# clicks every 0.5 s; the delay is where each output click peaks
x = np.zeros(SR * 5, "float32")
x[np.arange(SR // 4, len(x) - SR // 4, SR // 2)] = 1.0
for c in (5, 10, -10):
    y = librosa.effects.pitch_shift(x, sr=SR, n_steps=c / 100)
    d = []
    for k in np.where(x > 0)[0]:
        w = np.abs(y[k - SR // 8 : k + SR // 8])
        d.append((np.argmax(w) - SR // 8) / SR * 1000)
    print(
        f"click delay at {c:+d} cents: mean {np.mean(d):+.2f} ms, sd {np.std(d):.2f} ms",
        flush=True,
    )

kong = load("kong")
pick = json.load(open(f"{ROOT}/data/pick.json"))[::3]


def f1(on, pi, ro, rp, lag):
    """Onset F1 (%) with predicted onsets moved back by lag seconds."""
    on = on - lag
    ok = (on >= 0) & (on < DUR - 1)
    keep = ro < DUR - 1
    return mir_eval.transcription.precision_recall_f1_overlap(
        np.c_[ro[keep], ro[keep] + 0.1], librosa.midi_to_hz(rp[keep]),
        np.c_[on[ok], on[ok] + 0.1], librosa.midi_to_hz(pi[ok]),
        offset_ratio=None, onset_tolerance=0.05,
    )[2] * 100  # fmt: skip


rows = []
for r in pick:
    x, _ = librosa.load(
        f"{ROOT}/data/maestro/{r['audio_filename']}", sr=SR, mono=True, duration=DUR + 3
    )
    n = sorted(
        pretty_midi.PrettyMIDI(f"{ROOT}/data/maestro/{r['midi_filename']}")
        .instruments[0]
        .notes,
        key=lambda v: v.start,
    )
    ro, rp = np.array([v.start for v in n]), np.array([v.pitch for v in n])
    out = {}
    for name, a in (
        ("clean", x),
        ("vocoder10", librosa.effects.pitch_shift(x, sr=SR, n_steps=0.1)),
    ):
        e = kong.transcribe(a.astype("float32"), None)["est_note_events"]
        on, pi = (
            np.array([q["onset_time"] for q in e]),
            np.array([q["midi_note"] for q in e]),
        )
        s = {round(float(g), 3): f1(on, pi, ro, rp, g) for g in LAGS}
        best = max(s, key=s.get)
        out[name] = (s[0.0], best, s[best], len(on))
    rows.append(out)
    print(
        r["year"],
        {
            k: tuple(round(v, 3) if isinstance(v, float) else v for v in t)
            for k, t in out.items()
        },
        flush=True,
    )
for name in ("clean", "vocoder10"):
    a = np.array([o[name][:3] for o in rows])
    print(
        f"{name}: F1 at lag 0 {a[:, 0].mean():.2f}, best lag mean {a[:, 1].mean() * 1000:+.1f} ms, F1 at best lag {a[:, 2].mean():.2f}"
    )
