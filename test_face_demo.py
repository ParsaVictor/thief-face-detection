"""
آزمونِ سرتاسریِ face_demo — با ویدیو و مدل‌های ساختگی.

پنج چیز را جداگانه می‌سنجد:
  ۱) درستیِ حکم برای سه وضعیتِ از پیش معلوم
  ۲) پشت به دوربین هرگز قضاوت نشود
  ۳) ★ سرعت — چند فراخوانیِ طبقه‌بند حذف می‌شود
  ۴) ★ چسبندگی — حکمِ قرمز با یک فریمِ بد برداشته نشود
  ۵) ★ گم‌نکردن — پس از عوض‌شدنِ شناسه، حکم حفظ شود
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from face_demo import (CLEAR, FULL, LEAR, LEYE, LHIP, LSHO, MEDICAL, NOSE, REAR,
                       REYE, RHIP, RSHO, ST_ANALYZING, ST_CLEAR, ST_SUSPECT,
                       ST_THIEF, Cfg, Decider, body_signature, contact_sheet,
                       head_box, kpt_signal, make_crops, orientation, run,
                       skin_signal)

W, H, FPS = 960, 540, 30
SKIN = (150, 175, 205)
CLOTH = (55, 50, 48)
BLUE = (200, 170, 120)


# ─────────────────────────────────────────────────────────────────────────
def draw_person(img, cx, base_y, scale, kind, facing_front=True, shirt=(70, 60, 90)):
    d = 9.0 * scale
    head_r = int(1.6 * d)
    eye_y = base_y
    sw = 3.25 * d
    sho_y = base_y + 3 * d
    hip_y = base_y + 13 * d

    cv2.rectangle(img, (int(cx - sw), int(sho_y)), (int(cx + sw), int(hip_y)),
                  shirt, -1)
    cv2.ellipse(img, (int(cx), int(base_y + 0.2 * d)), (head_r, int(head_r * 1.35)),
                0, 0, 360, SKIN, -1)

    if kind == "medical":
        cv2.rectangle(img, (int(cx - head_r * 0.9), int(base_y + 0.9 * d)),
                      (int(cx + head_r * 0.9), int(base_y + 3.0 * d)), BLUE, -1)
    elif kind == "full":
        cv2.ellipse(img, (int(cx), int(base_y + 0.2 * d)),
                    (int(head_r * 1.05), int(head_r * 1.4)), 0, 0, 360, CLOTH, -1)
        for s in (-1, 1):
            cv2.ellipse(img, (int(cx + s * d / 2), int(eye_y)),
                        (int(d * 0.42), int(d * 0.26)), 0, 0, 360, SKIN, -1)

    for s in (-1, 1):
        cv2.circle(img, (int(cx + s * d / 2), int(eye_y)), max(1, int(d * 0.16)),
                   (35, 35, 35), -1)

    sgn = 1.0 if facing_front else -1.0
    kxy = np.zeros((17, 2), np.float32)
    kc = np.zeros(17, np.float32)

    def S(i, x, y, c):
        kxy[i] = (x, y)
        kc[i] = c

    S(LSHO, cx + sgn * sw, sho_y, 0.95)
    S(RSHO, cx - sgn * sw, sho_y, 0.95)
    S(LEAR, cx + sgn * 1.3 * d, eye_y + 0.25 * d, 0.80)
    S(REAR, cx - sgn * 1.3 * d, eye_y + 0.25 * d, 0.80)
    S(LHIP, cx + 1.8 * d, hip_y, 0.85)
    S(RHIP, cx - 1.8 * d, hip_y, 0.85)
    if facing_front:
        S(LEYE, cx + d / 2, eye_y, 0.92)
        S(REYE, cx - d / 2, eye_y, 0.92)
        # با ماسک: چشم‌ها پیدا، بینی پوشیده
        S(NOSE, cx, eye_y + 0.9 * d, 0.06 if kind in ("medical", "full") else 0.90)

    bbox = np.array([cx - sw - 4, base_y - head_r * 1.5, cx + sw + 4, hip_y + 4],
                    np.float32)
    return bbox, kxy, kc


class FakePose:
    """جای ultralytics. `id_shift` شبیه‌سازیِ عوض‌شدنِ شناسه در وسطِ ویدیو."""

    def __init__(self, plan, frames, id_shift_at=None, shirts=None):
        self.plan = plan
        self.frames = frames
        self.id_shift_at = id_shift_at
        self.shirts = shirts or [(70, 60, 90)] * len(plan)

    def track(self, **kw):
        for f in range(self.frames):
            img = np.full((H, W, 3), 32, np.uint8)
            cv2.rectangle(img, (0, int(H * 0.75)), (W, H), (48, 44, 40), -1)
            boxes, kxys, kcs = [], [], []
            for i, (kind, front) in enumerate(self.plan):
                cx = W * (i + 1) / (len(self.plan) + 1) + 8 * np.sin(f / 9.0)
                b, kxy, kc = draw_person(img, cx, H * 0.30, 2.6, kind, front,
                                         self.shirts[i])
                boxes.append(b)
                kxys.append(kxy)
                kcs.append(kc)
            base = 100 if (self.id_shift_at and f >= self.id_shift_at) else 0
            yield _Res(img, boxes, kxys, kcs, base)


class _Obj:
    pass


def _Res(img, boxes, kxys, kcs, id_base=0):
    import torch
    r = _Obj()
    r.orig_img = img
    r.boxes = _Obj()
    r.boxes.id = torch.arange(1 + id_base, len(boxes) + 1 + id_base)
    r.boxes.xyxy = torch.from_numpy(np.array(boxes, np.float32))
    r.keypoints = _Obj()
    r.keypoints.xy = torch.from_numpy(np.array(kxys, np.float32))
    r.keypoints.conf = torch.from_numpy(np.array(kcs, np.float32))
    return r


class FakeMask:
    """جای مدلِ فاین‌تیون‌شده — از پوستِ نیمهٔ پایین تصمیم می‌گیرد."""

    def __init__(self):
        self.calls = 0
        self.images = 0

    def __call__(self, crops):
        self.calls += 1
        self.images += len(crops)
        out = []
        for c in crops:
            if c.size == 0:
                out.append((0.5, 0.5))
                continue
            h = c.shape[0]
            hsv = cv2.cvtColor(c[int(h * 0.55):, :], cv2.COLOR_BGR2HSV)
            skin = cv2.inRange(hsv, np.array([0, 15, 40], np.uint8),
                               np.array([28, 190, 255], np.uint8)) > 0
            p_open = float(np.clip(skin.mean() * 2.2, 0.03, 0.97))
            out.append((p_open, 1.0 - p_open))
        return out


class FakeZeroShot:
    def __call__(self, crops):
        out = []
        for c in crops:
            if c.size == 0:
                out.append(0.6)
                continue
            h = c.shape[0]
            v = float(cv2.cvtColor(c[int(h * 0.55):, :], cv2.COLOR_BGR2GRAY).mean())
            out.append(float(np.clip((v - 45) / 90.0, 0.05, 0.95)))
        return out


def make_video(path, n=40):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for _ in range(n):
        vw.write(np.zeros((H, W, 3), np.uint8))
    vw.release()


# ─────────────────────────────────────────────────────────────────────────
def main():
    tmp = Path(tempfile.mkdtemp(prefix="facedemo_"))
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"{'✅' if cond else '❌'} {name}{('  ' + extra) if extra else ''}")

    try:
        real = tmp / "in.mp4"
        make_video(real, 60)

        # ═══ ۱) درستیِ حکم ═══════════════════════════════════════════════
        print("=" * 70)
        print("  ۱) حکم برای سه وضعیتِ از پیش معلوم")
        print("=" * 70)
        plan = [("clear", True), ("medical", True), ("full", True)]
        mm = FakeMask()
        cfg = Cfg(video=str(real), out_video=str(tmp / "out.mp4"),
                  out_dir=str(tmp / "crops"), min_face_px=30, half=False)
        summary, dec, panels = run(cfg, pose_model=FakePose(plan, 60),
                                   mask_model=mm, zeroshot=FakeZeroShot(),
                                   verbose=False)
        expect = {1: ST_CLEAR, 2: ST_SUSPECT, 3: ST_THIEF}
        names = {1: "صورتِ باز", 2: "ماسکِ پزشکی", 3: "★ ماسکِ دزدی"}
        for tid, exp in expect.items():
            tr = dec.tracks.get(tid)
            got = tr.state if tr else "—"
            post = dec.posterior(tid) if tr else {}
            chk(f"{names[tid]:<16} → {got:<12}",
                got == exp,
                f"(C{post.get(CLEAR,0):.2f} M{post.get(MEDICAL,0):.2f} "
                f"F{post.get(FULL,0):.2f})")

        # ═══ ۲) سرعت ═════════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print("  ۲) ★ سرعت — چند فراخوانیِ طبقه‌بند حذف شد؟")
        print("=" * 70)
        skipped = summary["skipped_cadence"]
        obs = summary["observations"]
        total = skipped + obs
        naive = 60 * 3       # ۶۰ فریم × ۳ نفر، اگر همه هر فریم بررسی شوند
        print(f"   بدونِ آهنگِ بازبینی : {naive} فراخوانی")
        print(f"   با آهنگِ بازبینی    : {obs} فراخوانی  "
              f"({skipped} رد شد = {skipped/max(1,total):.0%} صرفه‌جویی)")
        print(f"   تصویرهای پردازش‌شدهٔ مدل: {mm.images}")
        chk("طبقه‌بند کمتر از حالتِ ساده اجرا شد", obs < naive,
            f"({obs} < {naive})")
        chk("صرفه‌جویی محسوس است", skipped / max(1, total) > 0.25)

        # زمانِ رسیدن به حکم
        print("\n   زمانِ رسیدن به حکم (فریم از لحظهٔ ورود):")
        d2 = Decider(cfg)
        p_thief = {CLEAR: 0.05, MEDICAL: 0.15, FULL: 0.80}
        first_susp = first_red = None
        for i in range(30):
            d2.observe(1, p_thief, 0.6, i / 30.0)
            st, _ = d2.decide(1, False, i / 30.0, frontal=1.0)   # روبه‌روی کامل
            if st == ST_SUSPECT and first_susp is None:
                first_susp = i + 1
            if st == ST_THIEF and first_red is None:
                first_red = i + 1
                break
        print(f"     مشکوک (خاکستری) پس از {first_susp} فریم")
        print(f"     دزد   (قرمز)    پس از {first_red} فریم")
        chk("حکمِ «مشکوک» در ≤۳ فریم", first_susp and first_susp <= 3)
        chk("حکمِ «دزد» در ≤۵ فریم", first_red and first_red <= 5)

        # === 2.5) gate test ===
        print()
        print("=" * 70)
        print("  ۲.۵) ★ گیتِ روبه‌رو — نیم‌رخ نباید قرمز شود")
        print("=" * 70)
        for fs_val, label, exp in ((1.00, "روبه‌روی کامل", ST_THIEF),
                                   (0.60, "۴۵ درجه", ST_THIEF),
                                   (0.25, "نیم‌رخ", ST_SUSPECT),
                                   (0.05, "تقریباً از پهلو", ST_SUSPECT)):
            dg = Decider(cfg)
            st = None
            for i in range(12):
                dg.observe(1, p_thief, 0.7, i / 30.0)
                st, _ = dg.decide(1, False, i / 30.0, frontal=fs_val)
            chk(f"{label:<18} frontal={fs_val:.2f} -> {st:<11}", st == exp,
                f"(انتظار: {exp})")

        # ═══ ۳) چسبندگی ══════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print("  ۳) ★ چسبندگی — حکمِ قرمز نباید با چند فریمِ بد برداشته شود")
        print("=" * 70)
        d3 = Decider(cfg)
        t = 0.0
        for i in range(8):
            t = i / 30.0
            d3.observe(1, p_thief, 0.7, t)
            d3.decide(1, False, t, frontal=1.0)
        chk("ابتدا قرمز شد", d3.tracks[1].state == ST_THIEF)
        p_clear = {CLEAR: 0.85, MEDICAL: 0.10, FULL: 0.05}
        for i in range(8, 20):                     # ~۰.۴ ثانیه شواهدِ مخالف
            t = i / 30.0
            d3.observe(1, p_clear, 0.7, t)
            d3.decide(1, False, t, frontal=1.0)
        chk("بعد از ۰.۴ ثانیه شواهدِ مخالف هنوز قرمز است",
            d3.tracks[1].state == ST_THIEF, f"(unlock={cfg.thief_unlock_s}s)")
        for i in range(20, 110):                   # ۳ ثانیه شواهدِ مخالف
            t = i / 30.0
            d3.observe(1, p_clear, 0.7, t)
            d3.decide(1, False, t, frontal=1.0)
        chk("بعد از ۳ ثانیه شواهدِ مخالف، قرمز برداشته شد",
            d3.tracks[1].state != ST_THIEF, f"(→ {d3.tracks[1].state})")

        # سبز باید فوراً شکسته شود
        d4 = Decider(cfg)
        for i in range(6):
            d4.observe(1, p_clear, 0.7, i / 30.0)
            d4.decide(1, False, i / 30.0, frontal=1.0)
        was_clear = d4.tracks[1].state == ST_CLEAR
        for i in range(6, 12):
            d4.observe(1, p_thief, 0.7, i / 30.0)
            d4.decide(1, False, i / 30.0, frontal=1.0)
        chk("سبز با نشانهٔ پوشش فوراً شکسته می‌شود",
            was_clear and d4.tracks[1].state != ST_CLEAR,
            f"(→ {d4.tracks[1].state})")

        # ═══ ۴) گم‌نکردنِ فرد ═════════════════════════════════════════════
        print("\n" + "=" * 70)
        print("  ۴) ★ گم‌نکردن — شناسه عوض شود، حکم حفظ شود")
        print("=" * 70)
        # لباس‌های متمایز تا امضای رنگی معنادار باشد
        shirts = [(40, 40, 190), (40, 170, 60), (180, 90, 40)]
        s5, d5, _ = run(Cfg(video=str(real), out_video=str(tmp / "s.mp4"),
                            out_dir=str(tmp / "cs"), min_face_px=30, half=False),
                        pose_model=FakePose(plan, 60, id_shift_at=30, shirts=shirts),
                        mask_model=FakeMask(), zeroshot=FakeZeroShot(), verbose=False)
        print(f"   شناسه‌ها در فریم ۳۰ عوض شدند (۱،۲،۳ → ۱۰۱،۱۰۲،۱۰۳)")
        print(f"   وصلِ مجدد: {s5['reattached']} بار")
        new_thief = [t for t in d5.tracks.values()
                     if t.tid > 100 and t.state == ST_THIEF]
        chk("هویت پس از عوض‌شدنِ شناسه وصل شد", s5["reattached"] >= 1)
        chk("دزد بعد از عوض‌شدنِ شناسه هنوز قرمز است", len(new_thief) >= 1,
            f"(tid={[t.tid for t in new_thief]}, "
            f"ارث از {[t.inherited_from for t in new_thief]})")

        # ═══ ۵) پشت به دوربین ═════════════════════════════════════════════
        print("\n" + "=" * 70)
        print("  ۵) پشت به دوربین هرگز قضاوت نشود")
        print("=" * 70)
        s2, d2b, _ = run(Cfg(video=str(real), out_video=str(tmp / "b.mp4"),
                             out_dir=str(tmp / "cb"), min_face_px=30, half=False),
                         pose_model=FakePose([("full", False)], 60),
                         mask_model=FakeMask(), zeroshot=FakeZeroShot(),
                         verbose=False)
        chk("صفر مشاهده، صفر آلارم", s2["alerts"] == 0 and s2["observations"] == 0,
            f"(رد‌شده: {s2['skipped_back']})")

        # ═══ ۶) خروجی‌ها ══════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print("  ۶) خروجی‌ها")
        print("=" * 70)
        chk("ویدیوی خروجی نوشته شد", Path(cfg.out_video).stat().st_size > 1000)
        chk("صفحهٔ قضاوت ساخته شد",
            (Path(cfg.out_dir) / "_contact_sheet.jpg").exists())
        chk("برشِ پهن برای هر سوژه", len(panels) == 3)
        tr = dec.tracks[3]
        chk("برشِ پهن بزرگ‌تر از تنگ است",
            tr.best_wide is not None and tr.best_tight is not None
            and tr.best_wide.shape[0] > tr.best_tight.shape[0],
            f"({tr.best_tight.shape[:2]} → {tr.best_wide.shape[:2]})")

        sheet = Path(cfg.out_dir) / "_contact_sheet.jpg"
        if sheet.exists():
            dst = Path(__file__).parent / "_test_contact_sheet.jpg"
            shutil.copy(sheet, dst)
            print(f"   صفحهٔ قضاوتِ آزمون: {dst}")

        print("\n" + "=" * 70)
        print("✅ همهٔ آزمون‌ها موفق" if ok else "❌ آزمون شکست خورد")
        print("=" * 70)
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
