"""The README figure, drawn from results/summary.txt (no numbers typed in by hand).

usage: python src/make_figure.py

docs/retune.png  Kong's onset F1 change against the size of the pitch offset: a real
                 tuning offset (resample) before and after retuning, and the phase
                 vocoder before and after retuning, with 95% bootstrap intervals.
Colours follow tsm-confound's figures (phase vocoder blue), and every line is labelled
directly, so no reading depends on colour alone.
"""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/docs"
os.makedirs(OUT, exist_ok=True)

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a"}
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "lines.linewidth": 2,
})  # fmt: skip

# rows of summary.txt: method |c| raw-drop [lo, hi] retuned-drop [lo, hi] est-err
NUM = r"(-?\d+\.\d+)"
ROW = re.compile(
    rf"^(resample|vocoder)\s+(\d+)\s+{NUM} \[\s*{NUM},\s*{NUM}\]\s+{NUM} \[\s*{NUM},\s*{NUM}\]"
)
rows = {}
for line in open(f"{ROOT}/results/summary.txt"):
    m = ROW.match(line)
    if m and int(m[2]) <= 35:  # +-50 cents is ambiguous by half a semitone
        rows.setdefault(m[1], []).append(
            [int(m[2])] + [float(x) for x in m.groups()[2:]]
        )

fig, ax = plt.subplots(figsize=(7.6, 4.2))
series = (
    ("resample", 1, C["orange"], "-", "tuning offset (resample)"),
    ("resample", 4, C["aqua"], "-", "same, after retuning"),
    ("vocoder", 1, C["blue"], "-", "phase vocoder"),
    ("vocoder", 4, C["blue"], (0, (4, 3)), "phase vocoder, after retuning"),
)
for method, col, color, style, label in series:
    r = rows[method]
    x = [v[0] for v in r]
    y, lo, hi = ([v[col + k] for v in r] for k in (0, 1, 2))
    ax.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0)
    ax.plot(x, y, color=color, linestyle=style, marker="o", ms=5)
    ax.annotate(
        label,
        (x[-1], y[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        color=INK2,
        fontsize=9,
    )
ax.axhline(0, color=MUTED, lw=1)
ax.set_xticks([5, 10, 20, 35])
ax.set_xlim(3, 36)
ax.set_xlabel("pitch offset (cents, both directions averaged)")
ax.set_ylabel("onset F1 change (points)")
ax.set_title(
    "Retuning removes a tuning offset, but not the phase vocoder's damage",
    loc="left",
    pad=12,
)
fig.savefig(f"{OUT}/retune.png", dpi=160, bbox_inches="tight")
print("wrote docs/retune.png")
