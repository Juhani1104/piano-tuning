"""Same tuning test on two more open transcribers: Transkun 2.0 (piano) and Basic Pitch (general).

Ten pieces, one per year. Offsets are made at 44.1 kHz, then resampled to each model's rate.
Retuning uses the same estimator as exp.py, computed at 16 kHz.
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import json, os, sys, tempfile, numpy as np, librosa, mir_eval, pretty_midi, soundfile as sf, torch, moduleconf
import transkun
R = ROOT + ""; DUR = 90; SR = 44100
TMP = tempfile.mkdtemp()
pick = json.load(open(f"{R}/data/pick.json"))[::3]
COND = [("resample", 0), ("resample", 10), ("resample", -10), ("resample", 20), ("resample", -20), ("resample", 35), ("resample", -35),
        ("vocoder", 5), ("vocoder", -5), ("vocoder", 10), ("vocoder", -10)]
d = os.path.dirname(transkun.__file__)
c = moduleconf.parseFromFile(f"{d}/pretrained/2.0.conf"); TK = c["Model"].module.TransKun(conf=c["Model"].config)
ck = torch.load(f"{d}/pretrained/2.0.pt", map_location="cpu", weights_only=False)
TK.load_state_dict(ck.get("best_state_dict", ck.get("state_dict")), strict=False); TK = TK.cuda().eval(); torch.set_grad_enabled(False)
from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH
BP = Model(ICASSP_2022_MODEL_PATH)


def run_tk(x):
    y = librosa.resample(x, orig_sr=SR, target_sr=TK.fs, res_type="soxr_hq") if TK.fs != SR else x
    n = TK.transcribe(torch.from_numpy(np.stack([y, y], 1).astype("float32")).cuda())
    n = [v for v in n if v.pitch > 0]
    return np.array([v.start for v in n]), np.array([v.pitch for v in n])


def run_bp(x):
    f = f"{TMP}/bp.wav"; sf.write(f, x, SR)
    _, _, ev = predict(f, BP)
    return np.array([e[0] for e in ev]), np.array([e[2] for e in ev])


def shift(x, m, c):
    r = 2 ** (c / 1200)
    if m == "resample":
        return librosa.resample(x, orig_sr=SR * r, target_sr=SR, res_type="soxr_hq"), 1 / r
    return librosa.effects.pitch_shift(x, sr=SR, n_steps=c / 100), 1.0


def est(x):
    return float(librosa.estimate_tuning(y=librosa.resample(x, orig_sr=SR, target_sr=16000), sr=16000, resolution=.005, bins_per_octave=12) * 100)


def f1(on, pi, ref, t):
    ok = on < DUR - 1; ri = ref[0] * t; keep = ri < DUR - 1
    return mir_eval.transcription.precision_recall_f1_overlap(np.c_[ri[keep], ri[keep] + .1], ref[1][keep], np.c_[on[ok], on[ok] + .1], librosa.midi_to_hz(pi[ok]), offset_ratio=None, onset_tolerance=.05)[2] * 100


out = f"{R}/results/exp2.json"
rows = json.load(open(out)) if os.path.exists(out) else []; seen = {x["piece"] for x in rows}
for k, r in enumerate(pick):
    if k in seen:
        continue
    x, _ = librosa.load(f"{R}/data/maestro/{r['audio_filename']}", sr=SR, mono=True, duration=DUR + 3)
    n = sorted(pretty_midi.PrettyMIDI(f"{R}/data/maestro/{r['midi_filename']}").instruments[0].notes, key=lambda v: v.start)
    ref = (np.array([v.start for v in n]), librosa.midi_to_hz(np.array([v.pitch for v in n])))
    for m, cc in COND:
        y, t = shift(x, m, cc); e = est(y); z, t2 = shift(y, "resample", -e)
        for name, fn in (("transkun", run_tk), ("basicpitch", run_bp)):
            a = f1(*fn(y), ref, t); b = f1(*fn(z), ref, t * t2)
            rows.append(dict(piece=k, year=r["year"], model=name, method=m, shift=cc, est=e, raw=a, retuned=b))
            print(k, r["year"], name, m, cc, "est", round(e, 1), "F1", round(a, 2), "retuned", round(b, 2), flush=True)
    json.dump(rows, open(out, "w"), indent=0)
print("done", flush=True)
