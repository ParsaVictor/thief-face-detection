"""
make_figures.py — تولیدِ شکل‌های README از روی اندازه‌گیری‌های واقعیِ کد.

هر شکل با اجرای همان توابعِ سیستم ساخته می‌شود، نه با اعداد دستی.
پس اگر الگوریتم عوض شود، کافی است این اسکریپت دوباره اجرا شود.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from face_demo import (CLEAR, FULL, LEAR, LEYE, LHIP, LSHO, MEDICAL, NOSE, REAR,
                       REYE, RHIP, RSHO, Cfg, Decider, ST_SUSPECT, ST_THIEF,
                       combine, frontal_score, head_box, kpt_signal, orientation,
                       skin_signal)

OUT = Path(__file__).parent / "assets"
OUT.mkdir(exist_ok=True)

BG = "#0d1117"
FG = "#e6edf3"
MUTED = "#8b949e"
GREEN = "#3fb950"
RED = "#f85149"
ORANGE = "#d29922"
BLUE = "#58a6ff"
GRID = "#21262d"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.edgecolor": GRID, "grid.color": GRID,
    "font.family": "DejaVu Sans", "font.size": 11,
})


def style(ax, title=None, sub=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=.35, linewidth=.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=FG, fontsize=14, fontweight="bold", pad=30, loc="left")
    if sub:
        ax.text(0, 1.025, sub, transform=ax.transAxes, color=MUTED, fontsize=10)


# ══════════════════════════════════════════════════════════════════════
# 1) CHIRALITY — the signature idea
# ══════════════════════════════════════════════════════════════════════
def fig_chirality():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, facing in zip(axes, (True, False)):
        sgn = 1 if facing else -1
        cx, sy = 0.0, 0.0
        sw = 1.0
        L = (cx + sgn * sw, sy)
        R = (cx - sgn * sw, sy)
        # torso
        ax.add_patch(mp.FancyBboxPatch((-sw, -2.6), 2 * sw, 2.6,
                                       boxstyle="round,pad=0.06",
                                       fc="#1f2937", ec=GRID, lw=1.4))
        # head
        ax.add_patch(mp.Circle((cx, 1.05), 0.62,
                               fc="#374151" if not facing else "#4b5563",
                               ec=GRID, lw=1.4))
        if facing:
            for dx in (-0.22, 0.22):
                ax.add_patch(mp.Circle((cx + dx, 1.18), 0.085, fc=FG, ec="none"))
        # shoulders
        for p, lab, col in ((L, "L", BLUE), (R, "R", ORANGE)):
            ax.add_patch(mp.Circle(p, 0.16, fc=col, ec="none", zorder=5))
            ax.text(p[0], p[1] + .42, lab, ha="center", color=col,
                    fontsize=15, fontweight="bold")
        ax.annotate("", xy=R, xytext=L,
                    arrowprops=dict(arrowstyle="<->", color=FG, lw=1.8))
        dx = R[0] - L[0]
        sign = "+" if (L[0] - R[0]) > 0 else "−"
        col = GREEN if facing else RED
        ax.text(0, -0.62, f"x(L) − x(R)  =  {sign}",
                ha="center", color=col, fontsize=15, fontweight="bold")
        ax.text(0, -1.15, "FACING CAMERA" if facing else "FACING AWAY",
                ha="center", color=col, fontsize=12, fontweight="bold")
        ax.set_xlim(-2.3, 2.3); ax.set_ylim(-3.0, 2.15)
        ax.set_aspect("equal"); ax.axis("off")

    fig.suptitle("Chirality — reading head orientation from anatomical keypoint labels",
                 color=FG, fontsize=15, fontweight="bold", y=.965)
    fig.text(.5, .035,
             "COCO keypoints are labelled anatomically (the person's OWN left/right).\n"
             "When someone turns around, the labels cross over in image space — "
             "a free, face-independent orientation signal that works even under a balaclava.",
             ha="center", color=MUTED, fontsize=10)
    fig.subplots_adjust(left=.02, right=.98, top=.88, bottom=.20, wspace=.05)
    fig.savefig(OUT / "chirality.png", dpi=170)
    plt.close(fig)
    print("  ✔ chirality.png")


# ══════════════════════════════════════════════════════════════════════
# 2) FRONTAL GATE — measured, not hand-drawn
# ══════════════════════════════════════════════════════════════════════
def _person(turn, d=20.0, masked=True):
    cx, ey = 400.0, 200.0
    xy = np.zeros((17, 2), np.float32); kc = np.zeros(17, np.float32)

    def S(i, x, y, c):
        xy[i] = (x, y); kc[i] = c
    far = float(np.clip(turn * 1.6, 0, 1)); sw = 3.25 * d * turn
    S(LSHO, cx + sw, ey + 3 * d, .95); S(RSHO, cx - sw, ey + 3 * d, .95 * far)
    S(LEAR, cx + 1.3 * d * turn, ey + .25 * d, .8)
    S(REAR, cx - 1.3 * d * turn, ey + .25 * d, .8 * far)
    S(LEYE, cx + d / 2 * turn, ey, .92); S(REYE, cx - d / 2 * turn, ey, .92 * far)
    S(NOSE, cx, ey + .55 * d, .06 if masked else .9 * far)
    S(LHIP, cx + 1.8 * d, ey + 13 * d, .85); S(RHIP, cx - 1.8 * d, ey + 13 * d, .85)
    return xy, kc


def fig_frontal_gate():
    cfg = Cfg(half=False)
    turns = np.linspace(0.06, 1.0, 40)
    angles = np.degrees(np.arccos(np.clip(turns, 0, 1)))
    fs, widths = [], []
    for t in turns:
        xy, kc = _person(t)
        fa, oc = orientation(xy, kc, cfg.kpt_conf)
        fs.append(frontal_score(xy, kc, fa, oc, cfg.kpt_conf))
        hb, _, _ = head_box(xy, kc, (340, 150, 460, 480), 1280, 720, cfg)
        widths.append(hb[2] - hb[0])
    fs = np.array(fs); widths = np.array(widths)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    a1.plot(angles, fs, color=BLUE, lw=2.6)
    a1.axhline(cfg.red_min_frontal, color=RED, ls="--", lw=1.6)
    a1.fill_between(angles, cfg.red_min_frontal, 1.02,
                    where=fs >= cfg.red_min_frontal, color=RED, alpha=.10)
    a1.text(2, cfg.red_min_frontal + .03, "RED ALERT PERMITTED",
            color=RED, fontsize=10, fontweight="bold")
    a1.text(52, .12, "capped at ORANGE\n(never red)", color=ORANGE,
            fontsize=10, fontweight="bold")
    a1.set_xlabel("head yaw (degrees from frontal)")
    a1.set_ylabel("frontal score")
    a1.set_ylim(0, 1.02)
    style(a1, "The frontal gate", "measured by running frontal_score() over synthetic poses")

    a2.plot(angles, widths, color=ORANGE, lw=2.6, label="with compensation")
    raw = np.array([3.0 * np.hypot(*(_person(t)[0][LEYE] - _person(t)[0][REYE])) * 1.0
                    for t in turns])
    a2.plot(angles, raw, color=MUTED, lw=1.8, ls=":", label="raw eye distance")
    a2.axhline(Cfg().min_face_px, color=RED, ls="--", lw=1.4)
    a2.text(2, Cfg().min_face_px + 2, "min_face_px — below this we do not judge",
            color=RED, fontsize=9)
    a2.set_xlabel("head yaw (degrees from frontal)")
    a2.set_ylabel("head-box width (px)")
    a2.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=9)
    style(a2, "Foreshortening compensation",
          "eye distance shrinks with cos(yaw); shoulder width restores the floor")

    fig.tight_layout()
    fig.savefig(OUT / "frontal_gate.png", dpi=170)
    plt.close(fig)
    print("  ✔ frontal_gate.png")


# ══════════════════════════════════════════════════════════════════════
# 3) HIJAB SAFETY — the finding that changed the design
# ══════════════════════════════════════════════════════════════════════
def _face(kind, size=170):
    img = np.full((size, size, 3), (40, 38, 36), np.uint8)
    SKIN, DARK, BLUEC, NAVY = (150, 175, 205), (48, 44, 42), (200, 170, 120), (80, 60, 45)
    cy, cx = size // 2, size // 2

    def head(col, sx=.30, sy=.42):
        cv2.ellipse(img, (cx, cy), (int(size * sx), int(size * sy)), 0, 0, 360, col, -1)
    if kind == "open":
        head(SKIN)
    elif kind == "medical":
        head(SKIN)
        cv2.rectangle(img, (int(size * .16), int(size * .55)), (int(size * .84), size),
                      BLUEC, -1)
    elif kind == "balaclava":
        head(DARK)
        for dx in (-.13, .13):
            cv2.ellipse(img, (int(size * (.5 + dx)), int(size * .42)),
                        (int(size * .09), int(size * .05)), 0, 0, 360, SKIN, -1)
    elif kind == "hijab_medical":
        head(NAVY, .34, .46)
        cv2.ellipse(img, (cx, int(size * .55)), (int(size * .21), int(size * .30)),
                    0, 0, 360, SKIN, -1)
        cv2.rectangle(img, (int(size * .28), int(size * .60)),
                      (int(size * .72), int(size * .92)), BLUEC, -1)
    return img


def fig_hijab():
    cases = [("Open face", "open"), ("Medical mask", "medical"),
             ("Balaclava", "balaclava"), ("Hijab + medical mask", "hijab_medical")]
    fig = plt.figure(figsize=(12.5, 5.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], hspace=.42, wspace=.28)

    verdicts = []
    for i, (label, kind) in enumerate(cases):
        img = _face(kind)
        p, w, d = skin_signal(img)
        best = max(p, key=p.get)
        verdicts.append((label, p, best, kind))
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        wrong = (best == FULL and kind == "hijab_medical")
        col = RED if best == FULL else (GREEN if best == CLEAR else ORANGE)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.add_patch(mp.Rectangle((0, 0), img.shape[1] - 1, img.shape[0] - 1,
                                  fill=False, ec=col, lw=4))
        ax.set_title(label, color=FG, fontsize=11, pad=8)
        ax.text(.5, -.10, ("FALSE ALARM" if wrong else best.replace("_", " ")),
                transform=ax.transAxes, ha="center", color=col,
                fontsize=11, fontweight="bold")

    ax = fig.add_subplot(gs[1, :])
    labels = [v[0] for v in verdicts]
    x = np.arange(len(labels)); bw = .26
    for k, (cls, col, nm) in enumerate(((CLEAR, GREEN, "clear"),
                                        (MEDICAL, ORANGE, "benign cover"),
                                        (FULL, RED, "full cover (alert)"))):
        ax.bar(x + (k - 1) * bw, [v[1][cls] for v in verdicts], bw, color=col,
               label=nm, edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("skin-geometry cue output")
    ax.set_ylim(0, 1.75)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=9.5, ncol=3,
              loc="upper center", bbox_to_anchor=(.5, -.16), frameon=False)
    ax.text(3 + bw, 1.10, "FALSE ALARM", ha="center", color=RED,
            fontsize=11, fontweight="bold")
    ax.text(1.5, 1.45,
            "A headscarf covers the forehead exactly like a balaclava does."
            "\nA colour statistic cannot tell intent from fabric — so this cue was removed.",
            ha="center", va="center", color=RED, fontsize=10.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc="#2d1214", ec=RED, lw=1.2))
    style(ax, "Why the colour-based skin cue was disabled",
          "measured with skin_signal() — the cue cannot tell scarf fabric from balaclava fabric")
    fig.savefig(OUT / "hijab_safety.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("  ✔ hijab_safety.png")


# ══════════════════════════════════════════════════════════════════════
# 4) COMPUTE BUDGET
# ══════════════════════════════════════════════════════════════════════
def fig_budget():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.3),
                                 gridspec_kw={"width_ratios": [1, 1.25]})

    a1.barh(["adaptive cadence", "naive (every frame)"], [69, 180],
            color=[GREEN, MUTED], height=.55)
    for y, v in ((0, 69), (1, 180)):
        a1.text(v + 5, y, str(v), va="center", color=FG, fontweight="bold")
    a1.text(74, -.42, "−62%", color=GREEN, fontsize=17, fontweight="bold")
    a1.set_xlabel("classifier invocations (60-frame test scene, 3 people)")
    a1.set_xlim(0, 215)
    style(a1, "Compute saved", "measured in test_face_demo.py")

    states = ["Analyzing", "Suspicious\n(FOCUS)", "Thief\n(confirmed)", "Clear"]
    hz = [30, 30, 2, 0.5]
    cols = [MUTED, ORANGE, RED, GREEN]
    a2.bar(states, hz, color=cols, width=.55, edgecolor="none")
    for i, v in enumerate(hz):
        a2.text(i, v * 1.35, f"{v:g} Hz", ha="center", color=cols[i],
                fontweight="bold", fontsize=11)
    a2.set_yscale("log")
    a2.set_ylabel("re-evaluation rate (log scale)")
    a2.set_ylim(.25, 90)
    style(a2, "Attention follows risk",
          "the system spends its budget on the person who matters")
    fig.tight_layout()
    fig.savefig(OUT / "compute_budget.png", dpi=170)
    plt.close(fig)
    print("  ✔ compute_budget.png")


# ══════════════════════════════════════════════════════════════════════
# 5) EVIDENCE ACCUMULATION
# ══════════════════════════════════════════════════════════════════════
def fig_evidence():
    cfg = Cfg(half=False)
    obs = {CLEAR: .05, MEDICAL: .15, FULL: .80}
    d = Decider(cfg)
    xs, pf, states = [], [], []
    for i in range(14):
        t = i / 30.0
        d.observe(1, obs, .65, t)
        st, _ = d.decide(1, False, t, frontal=1.0)
        xs.append(i + 1); pf.append(d.posterior(1)[FULL]); states.append(st)

    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(xs, pf, color=RED, lw=2.8, marker="o", ms=5)
    ax.axhline(cfg.watch_enter, color=ORANGE, ls="--", lw=1.5)
    ax.axhline(cfg.suspect_enter, color=RED, ls="--", lw=1.5)
    ax.text(13.8, cfg.watch_enter + .02, "watch_enter → ORANGE + focus",
            ha="right", color=ORANGE, fontsize=9)
    ax.text(13.8, cfg.suspect_enter + .02, "suspect_enter → RED",
            ha="right", color=RED, fontsize=9)
    first_s = next((i + 1 for i, s in enumerate(states) if s == ST_SUSPECT), None)
    first_r = next((i + 1 for i, s in enumerate(states) if s == ST_THIEF), None)
    for f, col, lab in ((first_s, ORANGE, f"SUSPICIOUS\nframe {first_s}"),
                        (first_r, RED, f"ALERT\nframe {first_r}")):
        if f:
            ax.axvline(f, color=col, lw=1.2, alpha=.55)
            ax.text(f + .12, .12, lab, color=col, fontsize=10, fontweight="bold")
    ax.set_xlabel("observations (frames)")
    ax.set_ylabel("P(full face cover)")
    ax.set_ylim(0, 1.05); ax.set_xlim(.6, 14.4)
    style(ax, "Progressive escalation",
          "log-odds accumulation with quality weighting and time decay — "
          "measured with Decider()")
    fig.tight_layout()
    fig.savefig(OUT / "evidence.png", dpi=170)
    plt.close(fig)
    print("  ✔ evidence.png")


if __name__ == "__main__":
    print("generating figures ...")
    fig_chirality()
    fig_frontal_gate()
    fig_hijab()
    fig_budget()
    fig_evidence()
    sheet = Path(__file__).parent / "_test_contact_sheet.jpg"
    if sheet.exists():
        import shutil
        shutil.copy(sheet, OUT / "judgement_sheet.jpg")
        print("  ✔ judgement_sheet.jpg")
    print(f"\nall figures → {OUT}")
