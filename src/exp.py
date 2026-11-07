"""Does a small tuning offset hurt a pretrained piano transcriber, and does test-time retuning fix it?

Three ways of producing the offset on the same 90 s excerpts:
  resample  speed and pitch change together, no time-stretch artifact; GT times are rescaled
  stretch   signalsmith-stretch at fixed duration
  vocoder   librosa phase vocoder at fixed duration
Metric: mir_eval onset-only note F1 at 50 ms and 50 cents, as in Edwards et al.
Each condition also runs after retuning with librosa.estimate_tuning.
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import json, os, sys, numpy as np, librosa, mir_eval, pretty_midi, torch, python_stretch
from piano_transcription_inference import PianoTranscription
R = ROOT + ""; SR = 16000; DUR = 90
SHIFTS = [0, 5, -5, 10, -10, 20, -20, 35, -35]  # resume run drops the ambiguous +-50
pick = json.load(open(f"{R}/data/pick.json"))
m = PianoTranscription(device=torch.device("cuda"), checkpoint_path=f"{R}/data/note_F1.pth")


def rs(x, c):
    r = 2 ** (c / 1200)
    return librosa.resample(x, orig_sr=SR * r, target_sr=SR, res_type="soxr_hq"), r


def stretch(x, c):
    s = python_stretch.Signalsmith.Stretch(); s.preset(1, SR); s.setTransposeSemitones(c / 100)
    return s.process(x[None])[0].astype("float32")


def voc(x, c):
    return librosa.effects.pitch_shift(x, sr=SR, n_steps=c / 100)


def est(x):
    return float(librosa.estimate_tuning(y=x, sr=SR, resolution=.005, bins_per_octave=12) * 100)


def f1(x, ref, t):
    e = m.transcribe(x.astype("float32"), None)["est_note_events"]
    on = np.array([n["onset_time"] for n in e]); ok = on < DUR - 1
    iv = np.c_[on[ok], on[ok] + .1]; p = librosa.midi_to_hz(np.array([n["midi_note"] for n in e])[ok])
    ri = ref[0] * t; keep = ri[:, 0] < DUR - 1
    return mir_eval.transcription.precision_recall_f1_overlap(np.c_[ri[keep, 0], ri[keep, 0] + .1], ref[1][keep], iv, p, offset_ratio=None, onset_tolerance=.05)[2] * 100


out = f"{R}/results/exp.json"
rows = json.load(open(out)); seen = {x["piece"] for x in rows}
for k, r in enumerate(pick):
    if k in seen:
        continue
    x, _ = librosa.load(f"{R}/data/maestro/{r['audio_filename']}", sr=SR, mono=True, duration=DUR + 3)
    pm = pretty_midi.PrettyMIDI(f"{R}/data/maestro/{r['midi_filename']}")
    n = sorted(pm.instruments[0].notes, key=lambda v: v.start)
    ref = (np.array([[v.start, v.end] for v in n]), librosa.midi_to_hz(np.array([v.pitch for v in n])))
    base_est = est(x)
    for method in ("resample", "vocoder"):
        for c in SHIFTS:
            if method != "resample" and c == 0:
                continue
            if method == "resample":
                y, s = rs(x, c); t = 1 / s
            else:
                y = (stretch if method == "stretch" else voc)(x, c); t = 1.0
            e = est(y); F = f1(y, ref, t)
            z, s2 = rs(y, -e); G = f1(z, ref, t / s2)
            rows.append(dict(piece=k, year=r["year"], method=method, shift=c, base_est=base_est, est=e, raw=F, retuned=G))
            print(k, r["year"], method, c, "est", round(e, 1), "F1", round(F, 2), "retuned", round(G, 2), flush=True)
    json.dump(rows, open(out, "w"), indent=0)
print("done", flush=True)
