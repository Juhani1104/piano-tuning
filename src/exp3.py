"""Real tuning on Studio MAESTRO, retuning against augmented training, and why the vocoder hurts.

usage: python src/exp3.py

studio  the 30 pieces of data/pick.json as re-performed in Studio MAESTRO, scored
        against the original MAESTRO MIDI. Each piece's tuning is measured, and Kong
        and Edwards are run before and after retuning. A constant audio-to-MIDI lag
        is searched once per piece on Kong's unshifted output and then fixed, because
        a re-performance can start a few milliseconds late.
synth   10 pieces of original MAESTRO with the offsets of exp.py, for both models.
        Besides F1, every run keeps precision, recall, onset timing of matched notes
        and what unmatched predictions are, so the cause of a drop can be read off.
Edwards et al. trained Kong's architecture with random +-10 cent pitch shifts and
other augmentations, so "retune + Kong" against "Edwards" is retuning against
retraining. Rows go to results/exp3.json; finished (stage, piece) pairs are skipped.
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
LAGS = np.arange(-0.1, 0.1001, 0.005)  # seconds, studio audio relative to MIDI
SYNTH = [
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
pick = json.load(open(f"{ROOT}/data/pick.json"))
studio = json.load(open(f"{ROOT}/data/studio_pick.json"))
MODELS = {n: load(n) for n in ("kong", "edwards")}


def shift(x, m, c):
    """Shift x by c cents with method m; also returns the factor for GT onset times."""
    r = 2 ** (c / 1200)
    if m == "resample":
        return librosa.resample(
            x, orig_sr=SR * r, target_sr=SR, res_type="soxr_hq"
        ), 1 / r
    return librosa.effects.pitch_shift(x, sr=SR, n_steps=c / 100), 1.0


def est(x):
    """Tuning offset of x from A440 in cents, as in exp.py."""
    return float(
        librosa.estimate_tuning(y=x, sr=SR, resolution=0.005, bins_per_octave=12) * 100
    )


def reference(r):
    """Onsets (s) and MIDI pitches of a piece's GT notes, in time order."""
    n = sorted(
        pretty_midi.PrettyMIDI(f"{ROOT}/data/maestro/{r['midi_filename']}")
        .instruments[0]
        .notes,
        key=lambda v: v.start,
    )
    return np.array([v.start for v in n]), np.array([v.pitch for v in n])


def notes(model, x):
    """Onsets (s) and MIDI pitches transcribed from x."""
    e = MODELS[model].transcribe(x.astype("float32"), None)["est_note_events"]
    return np.array([n["onset_time"] for n in e]), np.array([n["midi_note"] for n in e])


def score(est_notes, ref, t, lag=0.0):
    """P, R, F1 and error breakdown of est_notes against ref, its onsets scaled by t."""
    on, pi = est_notes
    on = on - lag
    ok = (on >= 0) & (on < DUR - 1)
    ro, rp = ref[0] * t, ref[1]
    keep = ro < DUR - 1
    ro, rp, on, pi = ro[keep], rp[keep], on[ok], pi[ok]
    ri, ei = np.c_[ro, ro + 0.1], np.c_[on, on + 0.1]
    rh, eh = librosa.midi_to_hz(rp), librosa.midi_to_hz(pi)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ri, rh, ei, eh, offset_ratio=None, onset_tolerance=0.05
    )
    match = mir_eval.transcription.match_notes(
        ri, rh, ei, eh, onset_tolerance=0.05, offset_ratio=None
    )
    dev = np.array([on[j] - ro[i] for i, j in match]) * 1000
    used = {j for _, j in match}
    free = np.setdiff1d(np.arange(len(ro)), [i for i, _ in match])
    # an unmatched prediction is "semitone" or "octave" if an unmatched GT note sits
    # within 50 ms at that pitch distance, so the onset was found but the pitch was not
    kind = {"semitone": 0, "octave": 0, "other": 0}
    for j in set(range(len(on))) - used:
        near = free[np.abs(ro[free] - on[j]) < 0.05]
        d = np.abs(rp[near] - pi[j]) if len(near) else np.array([])
        kind[
            "semitone" if (d == 1).any() else "octave" if (d == 12).any() else "other"
        ] += 1
    return dict(
        P=p * 100,
        R=r * 100,
        F=f * 100,
        n_est=int(len(on)),
        n_ref=int(len(ro)),
        dev_med=float(np.median(np.abs(dev))) if len(dev) else None,
        dev_mean=float(dev.mean()) if len(dev) else None,
        **{f"fp_{k}": v for k, v in kind.items()},
    )


out = f"{ROOT}/results/exp3.json"
rows = json.load(open(out)) if os.path.exists(out) else []
done = {(x["stage"], x["piece"]) for x in rows}

for k, (r, name) in enumerate(zip(pick, studio)):
    if ("studio", k) in done:
        continue
    x, _ = librosa.load(
        f"{ROOT}/data/studio/{os.path.basename(name)}",
        sr=SR,
        mono=True,
        duration=DUR + 3,
    )
    ref = reference(r)
    e = est(x)
    # shift() already returns the factor for GT onsets, and the lag scales with it
    z, s2 = shift(x, "resample", -e)
    kong0 = notes("kong", x)
    lag = max(LAGS, key=lambda g: score(kong0, ref, 1.0, g)["F"])
    for model in MODELS:
        raw = kong0 if model == "kong" else notes(model, x)
        rows.append(dict(stage="studio", piece=k, year=r["year"], model=model, est=e, lag=float(lag),
                         raw=score(raw, ref, 1.0, lag), retuned=score(notes(model, z), ref, s2, lag * s2)))  # fmt: skip
        print("studio", k, r["year"], model, "est", round(e, 1), "lag", round(lag, 3),
              "F1", round(rows[-1]["raw"]["F"], 2), "retuned", round(rows[-1]["retuned"]["F"], 2), flush=True)  # fmt: skip
    json.dump(rows, open(out, "w"), indent=0)

for k, r in enumerate(pick[::3]):
    if ("synth", k) in done:
        continue
    x, _ = librosa.load(
        f"{ROOT}/data/maestro/{r['audio_filename']}", sr=SR, mono=True, duration=DUR + 3
    )
    ref = reference(r)
    for m, c in SYNTH:
        y, t = shift(x, m, c)
        e = est(y)
        z, t2 = shift(y, "resample", -e)
        for model in MODELS:
            rows.append(dict(stage="synth", piece=k, year=r["year"], model=model, method=m, shift=c, est=e,
                             raw=score(notes(model, y), ref, t), retuned=score(notes(model, z), ref, t * t2)))  # fmt: skip
            print("synth", k, model, m, c, "F1", round(rows[-1]["raw"]["F"], 2),
                  "retuned", round(rows[-1]["retuned"]["F"], 2), flush=True)  # fmt: skip
    json.dump(rows, open(out, "w"), indent=0)
print("done", flush=True)
