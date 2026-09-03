# -*- coding: utf-8 -*-
"""
save_suspect_crops.py — best-shot capture for flagged subjects.

Naively dumping a crop on every RED frame gives you hundreds of blurred,
half-occluded images of the same person. This keeps **one** image per
subject: the sharpest, largest view seen while that subject was flagged.

Scoring is deliberately simple and explainable:

    score = laplacian_variance(crop) * sqrt(crop_area)

Sharpness rejects motion blur; area prefers the moment the subject was
closest to the camera. The two are multiplied so a large blurry crop
cannot beat a small sharp one outright — both have to be decent.

Usage inside the notebook (see README):

    capture = SuspectCapture("runs/suspects")
    ...                                  # inside the per-track loop
    capture.observe(tid, frame, box, st, frame_idx)
    ...                                  # after the loop over frames
    capture.flush()
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np


class SuspectCapture:
    """Keeps the best crop per track id, writes it once at the end."""

    def __init__(self, out_dir="runs/suspects", pad=0.08, min_side=48,
                 states=("red",)):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pad = pad                # margin around the box, as a fraction
        self.min_side = min_side      # ignore crops smaller than this
        self.states = set(states)     # which verdicts are worth capturing
        self.best = {}                # tid -> dict(score, crop, meta)

    # ------------------------------------------------------------------
    @staticmethod
    def _score(crop):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return sharpness * float(np.sqrt(crop.shape[0] * crop.shape[1]))

    # ------------------------------------------------------------------
    def observe(self, tid, frame, box, st, frame_idx):
        """Call once per tracked subject per frame, before anything is drawn."""
        if st["color"] not in self.states:
            return

        H, W = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box]
        px, py = (x2 - x1) * self.pad, (y2 - y1) * self.pad
        x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
        x2, y2 = int(min(W, x2 + px)), int(min(H, y2 + py))
        if (x2 - x1) < self.min_side or (y2 - y1) < self.min_side:
            return

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        s = self._score(crop)
        prev = self.best.get(tid)
        if prev is not None and prev["score"] >= s:
            return

        self.best[tid] = {
            "score": s,
            "crop": crop.copy(),          # copy: `frame` is drawn on later
            "meta": {
                "track_id": int(tid),
                "frame": int(frame_idx),
                "state": st["color"],
                "label": st["label"],
                "confidence": round(float(st.get("conf", 0.0)), 3),
                "checks": int(st.get("checks", 0)),
                "box": [x1, y1, x2, y2],
                "sharpness_score": round(s, 1),
            },
        }

    # ------------------------------------------------------------------
    def flush(self, contact_sheet=True):
        """Write one JPEG (+ JSON sidecar) per subject. Returns the paths."""
        written = []
        for tid, b in sorted(self.best.items()):
            stem = self.dir / f"suspect_id{tid:03d}_f{b['meta']['frame']:06d}"
            cv2.imwrite(str(stem.with_suffix(".jpg")), b["crop"],
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            stem.with_suffix(".json").write_text(
                json.dumps(b["meta"], ensure_ascii=False, indent=2),
                encoding="utf-8")
            written.append(str(stem.with_suffix(".jpg")))

        if contact_sheet and written:
            self._contact_sheet()
        print(f"📸 {len(written)} suspect crop(s) -> {self.dir}/")
        return written

    # ------------------------------------------------------------------
    def _contact_sheet(self, cell=(200, 300), cols=5, name="_contact_sheet.jpg"):
        """One reviewable sheet: every flagged subject, ID and confidence."""
        items = sorted(self.best.items())
        rows = (len(items) + cols - 1) // cols
        cw, ch = cell
        sheet = np.full((rows * (ch + 26), cols * cw, 3), 22, np.uint8)

        for i, (tid, b) in enumerate(items):
            r, c = divmod(i, cols)
            thumb = cv2.resize(b["crop"], (cw - 8, ch - 8),
                               interpolation=cv2.INTER_AREA)
            y, x = r * (ch + 26) + 4, c * cw + 4
            sheet[y:y + ch - 8, x:x + cw - 8] = thumb
            cv2.rectangle(sheet, (x - 2, y - 2), (x + cw - 6, y + ch - 6),
                          (0, 0, 255), 2)
            cv2.putText(sheet, f"ID {tid}  {b['meta']['confidence']:.0%}",
                        (x, y + ch + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(str(self.dir / name), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
