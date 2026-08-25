# Legacy notebooks

Kept for provenance. **Neither is the current system** — see
[`Face_Mask_Demo.ipynb`](../Face_Mask_Demo.ipynb) and
[`face_demo.py`](../face_demo.py) in the repository root.

They are preserved because the failures documented in the
[engineering log](../README.md#engineering-log) are traceable to specific
lines in these files, and because it is useful to see where the project
started.

---

## `v1_original.ipynb`

The first working prototype. Single pipeline:

```
YOLO11-pose → ByteTrack
  └─ gate: mean confidence of (nose, left eye, right eye) ≥ 0.5 ?  → no: skip entirely
      └─ eye-line align + tight crop
          └─ SigLIP binary classifier → mask / no_mask
              └─ if mask: skin_ratio(upper 45% of crop) < 0.12 ? → red : orange
                  └─ 3 weighted votes → green locks permanently
```

**What it got right**, and what the current system inherited:

- A fine-tuned binary classifier as the backbone of the decision.
- A tight, eye-aligned face crop.
- Temporal voting instead of per-frame decisions.
- Batching crops before the classifier.
- The picture-in-picture gallery and the cropped-suspect output — the
  single most persuasive thing to show a stakeholder.

**Why it was rewritten** — the failures are catalogued in the
[engineering log](../README.md#engineering-log). The most consequential:

| Issue | Effect |
|---|---|
| `atan2` on anatomical eye order | crop rotated 180°; the classifier saw upside-down hair |
| Keypoint gate | silently skipped every masked person — the exact target case |
| Hard-coded `{0:"mask", 1:"no_mask"}` | the entire system could invert with no error |
| Permanent green lock | a thief could clear, then mask up, and never be re-checked |
| Crops taken after drawing | saved evidence had boxes and skeleton lines across every face |
| Windows path without a raw string | `"C:\1\..."` — `\1` is a control character; the file never opened |

One thing worth recording honestly: this version *felt* more accurate in
practice. Part of that was real — the strict gate meant profiles and
partial faces were never judged, so it rarely produced a false alarm.
Part of it was not: with the rotated crop, `skin_ratio` was measuring
hair, which almost always falls below the threshold, so the system was
**biased toward red**. Plenty of alarms can look like good detection.

Both halves of that lesson are in the current design: the gate came back,
but applied to the *verdict* rather than to *looking*.

---

## `v4_pipeline.ipynb`

The intermediate modular architecture — multi-camera scheduling, identity
bank, zone filtering, evidence fusion, snapshot engine. Roughly 3,900
lines across 24 modules flattened into one notebook.

It was over-built for the actual goal, which was accuracy on a single
video, and its size caused a concrete failure: a stale `build_classifier`
survived a regeneration and silently returned the zero-shot model instead
of the fine-tuned one. The strongest model in the system never loaded,
and that went unnoticed for several iterations.

The current single-file design is a deliberate response. The full modular
package still exists outside this repository and is the intended target
once the single-camera accuracy is settled.
