# -*- coding: utf-8 -*-
"""
make_figures_v7.py - README figures for version 7.

Every figure is built from the algorithm's real numbers: either by
simulating the same logic (attention budget) or from the measurement
table recorded in STATUS.md (phase E). If the configuration changes,
re-run this script.

    python make_figures_v7.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "assets"
OUT.mkdir(exist_ok=True)

# GitHub-dark palette, so the figures sit inside the README seamlessly
BG, FG, MUTED, GRID = "#0d1117", "#e6edf3", "#8b949e", "#21262d"
GREEN, RED, ORANGE, BLUE, PURPLE = "#3fb950", "#f85149", "#d29922", "#58a6ff", "#a371f7"
PANEL = "#161b22"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.edgecolor": GRID, "grid.color": GRID,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titleweight": "bold", "figure.dpi": 130,
})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("  ok  assets/" + name)


# =====================================================================
#  1) Identity swap - why "stabilising" the tracker backfired
# =====================================================================
def fig_identity_swap():
    fig, axes = plt.subplots(2, 1, figsize=(12.6, 7.6))

    cases = [
        dict(ax=axes[0], bad=True,
             title="BEFORE   match_thresh 0.85  |  track_buffer 90",
             note="Track B is kept alive for 90 frames and Kalman-predicted onto A's body.\n"
                  "A match needs only IoU > 15%, so B's verdict lands on A."),
        dict(ax=axes[1], bad=False,
             title="AFTER   match_thresh 0.60  |  track_buffer 30  |  low-confidence stage alive",
             note="B keeps a weak detection (0.20-0.50) right through the occlusion, so its track\n"
                  "never goes free. A match now needs IoU > 40%. Verdicts stay with their owners."),
    ]

    for c in cases:
        ax = c["ax"]
        ax.set_xlim(0, 120)
        ax.set_ylim(-0.62, 1.30)
        ax.set_yticks([0.25, 0.85])
        ax.set_yticklabels(["Track A  (masked)", "Track B  (clear face)"], fontsize=10.5)
        ax.set_xlabel("frame", fontsize=10)
        ax.grid(axis="x", alpha=0.35, lw=0.7)
        ax.set_title(c["title"], fontsize=12.5, pad=12,
                     color=RED if c["bad"] else GREEN)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        ax.axvspan(48, 68, color=PANEL, zorder=0)
        ax.text(58, 1.17, "occlusion\n(A passes in front of B)", ha="center",
                va="center", fontsize=9, color=MUTED, linespacing=1.4)

        def bar(y, x0, x1, col, label=None, hatch=None, alpha=1.0, tc="#0d1117"):
            ax.add_patch(mp.FancyBboxPatch(
                (x0, y - 0.075), x1 - x0, 0.15,
                boxstyle="round,pad=0,rounding_size=0.03",
                facecolor=col, edgecolor="none", alpha=alpha, hatch=hatch, zorder=3))
            if label:
                ax.text((x0 + x1) / 2, y, label, ha="center", va="center",
                        fontsize=9, color=tc, fontweight="bold", zorder=4)

        bar(0.25, 4, 48, RED, "RED")
        bar(0.85, 4, 48, GREEN, "GREEN")

        if c["bad"]:
            bar(0.85, 48, 68, GREEN, None, hatch="////", alpha=0.28)
            ax.text(58, 0.85, "lost, still predicted", ha="center", va="center",
                    fontsize=8, color=GREEN, style="italic", zorder=5)
            bar(0.25, 68, 116, GREEN, "GREEN  <- wrong person")
            bar(0.85, 68, 116, "#30363d", "new ID, from scratch", tc=FG)
            ax.annotate("", xy=(70, 0.34), xytext=(66, 0.76),
                        arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.4,
                                        connectionstyle="arc3,rad=-0.25"))
            ax.text(80, 0.55, "identity swap", fontsize=10.5, color=RED,
                    fontweight="bold")
        else:
            bar(0.85, 48, 68, GREEN, None, alpha=0.5)
            ax.text(58, 0.85, "weak det. holds track", ha="center", va="center",
                    fontsize=8, color="#0d1117", fontweight="bold", zorder=5)
            bar(0.25, 68, 116, RED, "RED")
            bar(0.85, 68, 116, GREEN, "GREEN")
            ax.text(86, 0.55, "verdicts preserved", fontsize=10.5, color=GREEN,
                    fontweight="bold")

        ax.text(2, -0.34, c["note"], fontsize=8.6, color=MUTED,
                va="top", linespacing=1.5)

    fig.suptitle("Identity swap - the failure mode that tracker \"stabilisation\" created",
                 fontsize=14.5, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=4.2)
    save(fig, "identity_swap.png")


# =====================================================================
#  2) Attention budget - re-check cadence per state
# =====================================================================
def fig_attention_budget():
    # exactly TrackStateManager.CADENCE plus GREEN_RECHECK
    CADENCE = {"Analysing": 1, "RED (alert)": 1, "ORANGE (medical)": 3,
               "GREEN (locked)": 60}
    FRAMES = 300  # 10 seconds at 30 fps

    names = list(CADENCE)
    adaptive = [FRAMES / CADENCE[n] for n in names]
    uniform = [FRAMES] * len(names)
    cols = [MUTED, RED, ORANGE, GREEN]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6),
                                  gridspec_kw={"width_ratios": [1.45, 1]})

    y = np.arange(len(names))[::-1]
    ax.barh(y + 0.19, uniform, height=0.34, color="#30363d",
            label="every person, every frame")
    ax.barh(y - 0.19, adaptive, height=0.34, color=cols,
            label="attention follows risk")
    for yi, a, u in zip(y, adaptive, uniform):
        ax.text(a + 6, yi - 0.19, f"{a:.0f}", va="center", fontsize=9.5, color=FG)
        ax.text(u + 6, yi + 0.19, f"{u:.0f}", va="center", fontsize=9.5, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10.5)
    ax.set_xlabel("classifier runs per subject over 300 frames (10 s @ 30 fps)", fontsize=10)
    ax.set_xlim(0, 365)
    ax.set_title("Where the compute goes", fontsize=12.5, pad=10)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, frameon=False, fontsize=9.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.19), ncol=2)
    ax.grid(axis="x", alpha=0.3, lw=0.7)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    mix = {"GREEN (locked)": 0.55, "ORANGE (medical)": 0.20,
           "Analysing": 0.15, "RED (alert)": 0.10}
    saved = sum(w * (1 - 1 / CADENCE[k]) for k, w in mix.items())
    ax2.pie([saved, 1 - saved], startangle=90, counterclock=False,
            colors=[BLUE, "#30363d"],
            wedgeprops=dict(width=0.42, edgecolor=BG, lw=2))
    ax2.text(0, 0.06, f"-{saved*100:.0f}%", ha="center", va="center",
             fontsize=27, fontweight="bold", color=BLUE)
    ax2.text(0, -0.24, "classifier calls", ha="center", va="center",
             fontsize=10, color=MUTED)
    ax2.set_title("Typical retail scene\n55% resolved | 20% medical | 15% new | 10% alert",
                  fontsize=10.5, pad=10, color=MUTED, fontweight="normal")
    ax2.set_aspect("equal")

    fig.suptitle("Attention follows risk - the classifier is the cost, so it is rationed",
                 fontsize=14, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "attention_budget.png")


# =====================================================================
#  3) Phase E - the bug the face-visibility cue created, and the fix
# =====================================================================
def fig_phase_e():
    # measurement table recorded in STATUS.md
    scen = ["Open face", "Surgical mask", "Full balaclava"]
    eye = [0.92, 0.12, 0.02]
    nose = [0.90, 0.05, 0.02]
    v_vis = [+1.00, -0.48, -0.64]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6),
                                  gridspec_kw={"width_ratios": [1, 1.15]})

    x = np.arange(3)
    ax.bar(x - 0.19, eye, width=0.36, color=BLUE, label="eye confidence")
    ax.bar(x + 0.19, nose, width=0.36, color=PURPLE, label="nose confidence")
    ax.set_xticks(x)
    ax.set_xticklabels(scen, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("keypoint confidence", fontsize=10)
    ax.set_title("The face-visibility cue collapses under cover", fontsize=11.5, pad=10)
    ax.legend(frameon=False, fontsize=9.5)
    ax.grid(axis="y", alpha=0.3, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    cols = [GREEN if v > 0 else RED for v in v_vis]
    ax2.barh(x[::-1], v_vis, height=0.5, color=cols)
    ax2.axvline(0, color=MUTED, lw=1)
    ax2.axvline(-0.35, color=ORANGE, lw=1.6, ls="--")
    ax2.text(-0.34, 2.45, "ORI_BACK_ENTER = -0.35", fontsize=8.5, color=ORANGE)
    for xi, v in zip(x[::-1], v_vis):
        ax2.text(v + (0.04 if v > 0 else -0.04), xi, f"{v:+.2f}",
                 va="center", ha="left" if v > 0 else "right",
                 fontsize=10, color=FG, fontweight="bold")
    ax2.set_yticks(x[::-1])
    ax2.set_yticklabels(scen, fontsize=10)
    ax2.set_xlim(-0.95, 1.45)
    ax2.set_ylim(-0.6, 2.75)
    ax2.set_xlabel("orientation score   (negative => \"facing away\")", fontsize=10)
    ax2.set_title("...so every covered face was pushed toward \"facing away\"",
                  fontsize=11.5, pad=10, color=RED)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    ax2.text(0.28, 0.55,
             "The fix: the visibility cue became\n"
             "ONE-SIDED - it may only push a\n"
             "subject toward \"facing camera\",\n"
             "never toward \"facing away\".\n\n"
             "The shoulder vote, which needs no\n"
             "face at all, became mandatory.",
             fontsize=8.8, color=FG, va="center", linespacing=1.6,
             bbox=dict(boxstyle="round,pad=0.6", facecolor=PANEL, edgecolor=GRID))

    fig.suptitle("Phase E - why a face-based cue must never decide \"facing away\"",
                 fontsize=14, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "phase_e_hardening.png")


# =====================================================================
if __name__ == "__main__":
    print("building figures into assets/ ...")
    fig_identity_swap()
    fig_attention_budget()
    fig_phase_e()
    print("done.")
