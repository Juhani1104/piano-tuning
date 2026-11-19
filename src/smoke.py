"""Check that both checkpoints load strictly and give sensible F1 on 30 s of one piece.

usage: python src/smoke.py
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
r = json.load(open(f"{ROOT}/data/pick.json"))[0]
x, _ = librosa.load(f"{ROOT}/data/maestro/{r['audio_filename']}", sr=16000, duration=30)
n = sorted(
    pretty_midi.PrettyMIDI(f"{ROOT}/data/maestro/{r['midi_filename']}")
    .instruments[0]
    .notes,
    key=lambda v: v.start,
)
ro = np.array([v.start for v in n])
rp = librosa.midi_to_hz(np.array([v.pitch for v in n]))
k = ro < 29
for name in ("kong", "edwards"):
    e = load(name).transcribe(x, None)["est_note_events"]
    on = np.array([q["onset_time"] for q in e])
    ok = on < 29
    p, rc, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        np.c_[ro[k], ro[k] + 0.1], rp[k], np.c_[on[ok], on[ok] + 0.1],
        librosa.midi_to_hz(np.array([q["midi_note"] for q in e])[ok]),
        offset_ratio=None, onset_tolerance=0.05,
    )  # fmt: skip
    print(
        name,
        "notes",
        int(ok.sum()),
        "P R F",
        round(p * 100, 2),
        round(rc * 100, 2),
        round(f * 100, 2),
        flush=True,
    )
    # a randomly initialised model would score near zero
    assert f > 0.8, name
print("smoke ok", flush=True)
