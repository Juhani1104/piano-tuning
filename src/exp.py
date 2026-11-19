"""Tuning offset test: Kong et al.'s piano transcriber on detuned MAESTRO excerpts.

usage: python src/exp.py

Every piece in data/pick.json is cut to its first 90 s, shifted by each offset in
SHIFTS, transcribed, and transcribed again after test-time retuning. Two ways of
making the offset are compared:
resample  pitch and speed change together, like a piano tuned off A440 or a tape
          played at the wrong speed; no time-stretch artifact, GT onsets are rescaled
vocoder   librosa's phase vocoder at fixed duration, the pre-0.39 audiomentations
          PitchShift default
Retuning estimates the offset with librosa.estimate_tuning and resamples it away.
The score is mir_eval onset-only note F1 at 50 ms and 50 cents.

Rows are appended to results/exp.json one piece at a time, and pieces already there
are skipped, so a stopped run resumes. The first ten pieces were run with two more
conditions, +-50 cents and signalsmith-stretch ("stretch"), and those rows are kept;
the resumed run dropped them to finish in time.
"""

import json
import os

import librosa
import mir_eval
import numpy as np
import pretty_midi
import python_stretch
import torch
from piano_transcription_inference import PianoTranscription

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = ROOT
SR = 16000  # the model's input rate
DUR = 90  # seconds of each piece; notes in the last second are not scored
# +-50 cents is half a semitone, where the direction of the offset is ambiguous
SHIFTS = [0, 5, -5, 10, -10, 20, -20, 35, -35]
pick = json.load(open(f"{R}/data/pick.json"))
m = PianoTranscription(
    device=torch.device("cuda"), checkpoint_path=f"{R}/data/note_F1.pth"
)


def rs(x, c):
    """Resample x so its pitch moves by c cents; also returns the speed ratio."""
    r = 2 ** (c / 1200)
    return librosa.resample(x, orig_sr=SR * r, target_sr=SR, res_type="soxr_hq"), r


def stretch(x, c):
    """Shift pitch by c cents at fixed duration with signalsmith-stretch."""
    s = python_stretch.Signalsmith.Stretch()
    s.preset(1, SR)
    s.setTransposeSemitones(c / 100)
    return s.process(x[None])[0].astype("float32")


def voc(x, c):
    """Shift pitch by c cents at fixed duration with librosa's phase vocoder."""
    return librosa.effects.pitch_shift(x, sr=SR, n_steps=c / 100)


def est(x):
    """Tuning offset of x from A440 in cents, within +-50."""
    return float(
        librosa.estimate_tuning(y=x, sr=SR, resolution=0.005, bins_per_octave=12) * 100
    )


def f1(x, ref, t):
    """Onset F1 (%) of the transcription of x against ref, its onsets scaled by t."""
    e = m.transcribe(x.astype("float32"), None)["est_note_events"]
    on = np.array([n["onset_time"] for n in e])
    ok = on < DUR - 1
    iv = np.c_[on[ok], on[ok] + 0.1]
    p = librosa.midi_to_hz(np.array([n["midi_note"] for n in e])[ok])
    ri = ref[0] * t
    keep = ri[:, 0] < DUR - 1
    # onset-only matching: the 0.1 s note length is a placeholder that is not scored
    return (
        mir_eval.transcription.precision_recall_f1_overlap(
            np.c_[ri[keep, 0], ri[keep, 0] + 0.1],
            ref[1][keep],
            iv,
            p,
            offset_ratio=None,
            onset_tolerance=0.05,
        )[2]
        * 100
    )


out = f"{R}/results/exp.json"
rows = json.load(open(out))
seen = {x["piece"] for x in rows}
for k, r in enumerate(pick):
    if k in seen:
        continue
    # 3 s past DUR, so notes near the end are not cut by the shift
    x, _ = librosa.load(
        f"{R}/data/maestro/{r['audio_filename']}", sr=SR, mono=True, duration=DUR + 3
    )
    pm = pretty_midi.PrettyMIDI(f"{R}/data/maestro/{r['midi_filename']}")
    n = sorted(pm.instruments[0].notes, key=lambda v: v.start)
    ref = (
        np.array([[v.start, v.end] for v in n]),
        librosa.midi_to_hz(np.array([v.pitch for v in n])),
    )
    base_est = est(x)
    for method in ("resample", "vocoder"):
        for c in SHIFTS:
            if method != "resample" and c == 0:
                continue
            if method == "resample":
                y, s = rs(x, c)
                t = 1 / s
            else:
                y = (stretch if method == "stretch" else voc)(x, c)
                t = 1.0
            e = est(y)
            F = f1(y, ref, t)
            # retuning is a resample too, so GT onsets move by its speed ratio
            z, s2 = rs(y, -e)
            G = f1(z, ref, t / s2)
            rows.append(
                dict(
                    piece=k,
                    year=r["year"],
                    method=method,
                    shift=c,
                    base_est=base_est,
                    est=e,
                    raw=F,
                    retuned=G,
                )
            )
            print(
                k, r["year"], method, c, "est", round(e, 1),
                "F1", round(F, 2), "retuned", round(G, 2), flush=True,
            )  # fmt: skip
    json.dump(rows, open(out, "w"), indent=0)
print("done", flush=True)
