# -*- coding: utf-8 -*-
"""
make_social_cards.py - LinkedIn-ready versions of the two README diagrams.

The mermaid diagrams in the README are correct but shaped for a docs page:
the architecture renders 1:2.2 tall and LinkedIn crops anything past 1:1.25,
and the state diagram's auto-layout overlaps its own edge labels.

These are the same two diagrams, redrawn for a feed: landscape, few words,
large type, and the four decision states painted in the colours the system
actually draws on the video.

    python make_social_cards.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "assets" / "social"
OUT.mkdir(parents=True, exist_ok=True)

BG, FG, MUTED, GRID = "#0d1117", "#e6edf3", "#8b949e", "#30363d"
GREEN, RED, ORANGE, BLUE, PURPLE = "#3fb950", "#f85149", "#d29922", "#58a6ff", "#a371f7"
PANEL, DARK = "#161b22", "#21262d"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "font.family": "DejaVu Sans",
})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.35, dpi=150)
    plt.close(fig)
    print("  ok  assets/social/" + name)


def box(ax, x, y, w, h, text, fc=PANEL, ec=GRID, tc=FG, fs=13, weight="normal",
        lw=1.6, r=0.055):
    ax.add_patch(mp.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight=weight, zorder=4, linespacing=1.5)


def arrow(ax, p0, p1, color=MUTED, lw=1.8, rad=0.0, ls="-"):
    ax.annotate("", xy=p1, xytext=p0, zorder=2,
                arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.45",
                                color=color, lw=lw, linestyle=ls,
                                connectionstyle=f"arc3,rad={rad}",
                                shrinkA=3, shrinkB=3))


def elbow(ax, pts, color=MUTED, lw=1.8):
    """Right-angled route: straight segments, arrowhead on the last one."""
    xs = [p[0] for p in pts[:-1]]
    ys = [p[1] for p in pts[:-1]]
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", zorder=2)
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw)


# =====================================================================
#  1) Architecture - landscape, three bands
# =====================================================================
def card_architecture():
    fig, ax = plt.subplots(figsize=(15.5, 8.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    ax.text(50, 59.4, "How it decides", ha="center", fontsize=26,
            fontweight="bold", color=FG)
    ax.text(50, 56.0, "Two pre-trained models. Everything in colour is a decision, not a model.",
            ha="center", fontsize=13, color=MUTED)

    y1, y2, y3 = 47.5, 30.0, 11.0

    for y, t in ((y1, "SEE"), (y2, "GATE"), (y3, "JUDGE")):
        ax.text(1.2, y, t, ha="left", va="center", fontsize=10.5,
                color=GRID, fontweight="bold", rotation=90)

    # ---- band 1: perception ----
    box(ax, 17, y1, 19, 7.4, "Video  /  RTSP", fs=14)
    box(ax, 42, y1, 21, 7.4, "YOLO26-pose\nperson + 17 keypoints", fs=13)
    box(ax, 68, y1, 19, 7.4, "ByteTrack", fc=GREEN, ec=GREEN, tc="#08240f",
        fs=15, weight="bold")
    ax.text(68, y1 - 5.4, "tuned against identity swap", ha="center",
            fontsize=10.5, color=GREEN)
    box(ax, 90, y1, 14, 7.4, "Identity\nmemory", fs=12)

    arrow(ax, (26.6, y1), (31.4, y1))
    arrow(ax, (52.6, y1), (57.4, y1))
    arrow(ax, (77.6, y1), (82.9, y1))

    # ---- band 2: gating ----
    box(ax, 26, y2, 26, 8.6, "Chirality orientation\nfacing camera, or away?",
        fc=BLUE, ec=BLUE, tc="#04203f", fs=13.5, weight="bold")
    box(ax, 62, y2, 26, 8.6, "Face gate\nsize · sharpness · confidence",
        fc=PURPLE, ec=PURPLE, tc="#1c0838", fs=13.5, weight="bold")
    box(ax, 90, y2, 14, 8.6, "\"Analysing\"\nno verdict", fc=DARK, fs=12, tc=MUTED)

    # identity memory -> chirality, routed as an elbow above band 2
    elbow(ax, [(90, y1 - 3.7), (90, 39.6), (26, 39.6), (26, y2 + 4.3)])

    arrow(ax, (39.2, y2), (48.9, y2))
    arrow(ax, (75.1, y2), (82.9, y2))
    ax.text(79, y2 + 2.4, "fails", ha="center", fontsize=10, color=MUTED,
            style="italic")
    ax.text(44, y2 + 2.4, "faces camera", ha="center", fontsize=10, color=MUTED,
            style="italic")

    ax.text(26, y2 - 7.6, "works through a full balaclava\n— it never looks at the face",
            ha="center", va="top", fontsize=10.5, color=BLUE, linespacing=1.5)
    ax.text(71, y2 - 7.6, "every gate fails toward silence,\nnever toward a false alarm",
            ha="center", va="top", fontsize=10.5, color=PURPLE, linespacing=1.5)

    # ---- band 3: decision ----
    box(ax, 17, y3, 21, 7.4, "SigLIP classifier\ncovered / not covered", fs=12.5)
    box(ax, 44, y3, 22, 7.4, "Skin above the mask\n+ brightness guard",
        fc=RED, ec=RED, tc="#2b0705", fs=13, weight="bold")
    box(ax, 71, y3, 20, 7.4, "State machine\nvoting · hysteresis", fs=12.5)
    box(ax, 92, y3, 13, 7.4, "Alert\n+ evidence", fc=DARK, fs=12)

    # face gate -> classifier, routed down the left
    elbow(ax, [(52.0, y2 - 4.3), (52.0, 17.4), (17.0, 17.4), (17.0, y3 + 3.9)])
    ax.text(53.0, 18.4, "passes", ha="left", fontsize=10, color=MUTED,
            style="italic")

    arrow(ax, (27.6, y3), (32.9, y3))
    arrow(ax, (55.1, y3), (60.9, y3))
    arrow(ax, (81.1, y3), (85.4, y3))

    # feedback loop: attention follows risk
    ax.annotate("", xy=(69, y2 - 4.4), xytext=(73, y3 + 3.8), zorder=2,
                arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.45",
                                color=ORANGE, lw=2.2, linestyle=(0, (4, 2)),
                                connectionstyle="arc3,rad=-0.30", shrinkA=3, shrinkB=3))
    ax.text(88, 19.5, "attention\nfollows risk", ha="center", fontsize=11.5,
            color=ORANGE, fontweight="bold", linespacing=1.5)
    ax.text(88, 15.8, "red 1 · orange 3 · green 60", ha="center", fontsize=10,
            color=ORANGE)

    ax.text(44, y3 - 5.6, "a surgical mask leaves skin above it — a balaclava does not",
            ha="center", va="top", fontsize=10.5, color=RED)

    save(fig, "architecture_card.png")


# =====================================================================
#  2) Decision states - the four verdicts, in their real colours
# =====================================================================
def card_states():
    fig, ax = plt.subplots(figsize=(15.5, 8.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 54)
    ax.axis("off")

    ax.text(50, 51.0, "Four states — and the bias toward silence",
            ha="center", fontsize=25, fontweight="bold", color=FG)
    ax.text(50, 47.6, "When the system is unsure, it says so. It does not guess.",
            ha="center", fontsize=13, color=MUTED)

    W, H = 25, 9.5
    states = [
        (20, 34, "ANALYSING", "no opinion yet", "#484f58", DARK, MUTED),
        (20, 14, "CLEAR", "identity recoverable", GREEN, GREEN, "#08240f"),
        (72, 34, "MEDICAL MASK", "covered, but benign", ORANGE, ORANGE, "#2b1d02"),
        (72, 14, "SUSPICIOUS", "identity NOT recoverable", RED, RED, "#2b0705"),
    ]
    for x, y, title, sub, ec, fc, tc in states:
        box(ax, x, y, W, H, "", fc=fc, ec=ec, lw=2.6, r=0.05)
        ax.text(x, y + 1.5, title, ha="center", va="center", fontsize=16.5,
                fontweight="bold", color=tc if fc != DARK else FG, zorder=5)
        ax.text(x, y - 2.2, sub, ha="center", va="center", fontsize=11,
                color=tc if fc != DARK else MUTED, zorder=5)

    # transitions
    arrow(ax, (20, 29.0), (20, 19.0), color=GREEN, lw=2.2)
    ax.text(21.4, 24.0, "3 votes\nface visible", ha="left", va="center",
            fontsize=10.5, color=GREEN, linespacing=1.5)

    arrow(ax, (32.7, 34), (59.2, 34), color=ORANGE, lw=2.2)
    ax.text(46, 36.4, "covered + skin above", ha="center", fontsize=10.5,
            color=ORANGE)

    arrow(ax, (32.7, 31.5), (59.4, 17.0), color=RED, lw=2.2, rad=-0.16)
    ax.text(45.5, 21.6, "covered + no skin\n+ bright enough", ha="center",
            fontsize=10.5, color=RED, linespacing=1.5)

    arrow(ax, (72, 29.0), (72, 19.0), color=RED, lw=2.0, rad=0.28)
    arrow(ax, (72, 19.0), (72, 29.0), color=ORANGE, lw=2.0, rad=0.28)
    ax.text(79.0, 24.0, "majority in a\n4-frame window", ha="left", va="center",
            fontsize=10.5, color=MUTED, linespacing=1.5)

    # green lock is never permanent
    ax.annotate("", xy=(59.4, 11.0), xytext=(32.7, 11.0), zorder=2,
                arrowprops=dict(arrowstyle="<|-|>,head_width=0.2,head_length=0.4",
                                color=MUTED, lw=1.8, linestyle=(0, (5, 3)),
                                connectionstyle="arc3,rad=-0.30", shrinkA=3, shrinkB=3))
    ax.text(46, 4.6, "a green lock is never permanent — re-checked every 60 frames,\n"
                     "because someone can walk in clear and mask up in aisle three",
            ha="center", fontsize=11, color=MUTED, linespacing=1.6)

    # the hold rule
    ax.add_patch(mp.FancyBboxPatch((3.2, 39.5), 33.6, 5.2,
                                   boxstyle="round,pad=0,rounding_size=0.06",
                                   facecolor=PANEL, edgecolor=GRID, lw=1.4, zorder=3))
    ax.text(20, 42.1, "too far · too blurred · too dark · facing away\n→ stays here, no verdict issued",
            ha="center", va="center", fontsize=10.8, color=MUTED,
            linespacing=1.6, zorder=4)
    arrow(ax, (12, 39.3), (14, 38.9), color=GRID, lw=1.6)

    save(fig, "decision_states_card.png")


if __name__ == "__main__":
    print("building LinkedIn cards into assets/social/ ...")
    card_architecture()
    card_states()
    print("done.")
