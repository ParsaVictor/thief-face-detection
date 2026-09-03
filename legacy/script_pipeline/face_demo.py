"""
face_demo.py — نسخهٔ دموی تک‌ویدیویی، ساده و قوی.

چرا از نو نوشته شد
==================
پکیج بزرگِ security_core برای محصول نهایی درست است، ولی برای رسیدن به
«دقتِ خوب روی یک ویدیو» زیادی بزرگ شده بود و هر بازسازی، خطرِ قاطی‌شدنِ
نسخه‌ها را داشت — چنانکه در آخرین اجرا، یک `build_classifier`ِ قدیمی
باعث شد مدلِ فاین‌تیون‌شده اصلاً بارگذاری نشود و فقط مدلِ ضعیفِ صفر-شات
کار کند.

اینجا همه‌چیز در یک فایل است، خطی و قابلِ دنبال‌کردن.

درسِ اصلی از نسخهٔ اولِ پروژه
============================
نسخهٔ اول با یک مدلِ دوکلاسهٔ فاین‌تیون‌شده روی یک کروپِ تنگِ صورت کار
می‌کرد و خوب جواب می‌داد. هر جا از آن دور شدیم، بدتر شد. پس اینجا:

    مدلِ فاین‌تیون‌شده = ستونِ اصلیِ تصمیم، نه یک گزینه در کنار بقیه.

سه سیگنالِ مکمل
===============
هیچ‌کدام به‌تنهایی کافی نیست؛ با هم بسیار قوی‌اند:

  ۱) مدلِ ماسک (فاین‌تیون‌شده) → «صورت پوشیده هست یا نه؟»
     قوی‌ترین سیگنال. همان چیزی که نسخهٔ اول را کار می‌انداخت.

  ۲) هندسهٔ کی‌پوینت → «چشم‌ها پیدا ولی بینی/دهان نه؟»
     دقیقاً همان نکته‌ای که خودت گفتی: هرکس ماسک زده، چشمانش پیداست
     ولی بینی‌اش نه. این سیگنال به رنگ و نور کاری ندارد، پس شب هم
     کار می‌کند.

  ۳) هندسهٔ پوست → «کدام *نوع* پوشش؟»
     اینجا کلیدِ تفکیکِ ماسکِ پزشکی از ماسکِ دزدی است:

         پیشانی پوست | پایینِ صورت پوست  →  صورتِ باز
         پیشانی پوست | پایینِ صورت پارچه →  ماسکِ پزشکی
         پیشانی پارچه| پایینِ صورت پارچه →  ★ ماسکِ دزدی (بالاکلاوا)

     نسخهٔ اول همین ایده را داشت ولی ناحیه را اشتباه انتخاب کرده بود
     (به‌خاطر باگِ چرخشِ ۱۸۰ درجه، به موی سر نگاه می‌کرد نه پیشانی).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
#  ۱) تنظیمات — همهٔ اعداد قابلِ تیون اینجاست
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class Cfg:
    # ---- ورودی / خروجی ----
    video: str = ""
    out_video: str = "runs/demo.mp4"
    out_dir: str = "runs/demo_crops"
    max_frames: int = 0            # ۰ = کل ویدیو

    # ---- مدل‌ها ----
    pose_weights: str = "yolo11n-pose.pt"
    imgsz: int = 640
    person_conf: float = 0.35
    kpt_conf: float = 0.35
    mask_model: str = "prithivMLmods/Face-Mask-Detection"   # ★ ستونِ اصلی
    use_zeroshot: bool = True      # فقط برای تفکیکِ نوعِ پوشش
    zeroshot_model: str = "google/siglip-base-patch16-224"
    half: bool = True

    # ---- کروپ ----
    # ★ دو کروپِ متفاوت با دو هدفِ متفاوت:
    #   تنگ  → ورودیِ طبقه‌بند. صورت باید بیشترِ کادر را پر کند.
    #   پهن  → برای دیدن و قضاوت‌کردنِ تو. باید سر و شانه و زمینه را
    #          نشان دهد تا بتوانی بگویی تصمیم درست بوده یا نه.
    tight_scale: float = 3.0       # ضریبِ فاصلهٔ دو چشم
    wide_scale: float = 2.1        # چند برابرِ کروپِ تنگ
    min_face_px: int = 40

    # ---- جهت‌گیری ----
    back_threshold: float = -0.25

    # ---- ★ گیتِ ترکیبی برای حکمِ قرمز ---------------------------------
    # نسخهٔ اولِ پروژه یک گیتِ سخت داشت: «میانگینِ اطمینانِ بینی و دو چشم
    # ≥ 0.5، وگرنه اصلاً بررسی نکن». همین باعث می‌شد نیم‌رخ‌ها کاملاً
    # کنار گذاشته شوند — و دقیقاً به همین دلیل آلارمِ کاذبِ نیم‌رخ نداشت.
    #
    # حذفِ کاملِ آن اشتباه بود (دزدِ نقاب‌دار هم چشم و بینی ندارد)، ولی
    # نگه‌داشتنِ کاملش هم اشتباه است.
    #
    # ترکیب: گیت را فقط برای **حکمِ قرمز** می‌گذاریم، نه برای دیدن.
    #   • همه بررسی می‌شوند و می‌توانند «مشکوک» شوند.
    #   • ولی آلارمِ قرمز فقط وقتی مجاز است که مطمئن باشیم فرد
    #     واقعاً رو به دوربین است.
    # نتیجه: نیم‌رخ حداکثر خاکستری می‌شود، هرگز قرمز.
    red_min_frontal: float = 0.45      # کمینهٔ امتیازِ روبه‌رو بودن برای قرمز
    red_min_eye_conf: float = 0.35     # هر دو چشم باید حداقل این اطمینان را داشته باشند

    # ---- تصمیم ----
    # ★ تشدیدِ تدریجی (escalation): اول با ۲ مشاهده یک حکمِ اولیه بده،
    #   بعد اگر مشکوک بود روی همان فرد تمرکز کن تا قطعی شود.
    #   کار به دو مرحله شکسته می‌شود چون دیدنِ سریعِ «مشکوک» ارزش دارد،
    #   ولی آلارمِ قرمز باید مطمئن باشد.
    min_votes: int = 2             # حکمِ اولیه بعد از ۲ مشاهده
    confirm_votes: int = 4         # برای قرمزشدن، این تعداد مشاهده لازم است
    decay: float = 0.80            # محوشدنِ شواهد در هر ثانیه

    watch_enter: float = 0.30      # از این احتمال به بالا → مشکوک + تمرکز
    suspect_enter: float = 0.55    # ورود به قرمز
    suspect_exit: float = 0.30     # خروج از قرمز (هیسترزیس)
    clear_enter: float = 0.60

    # ★ چسبندگیِ حکم — خواستهٔ صریح: بعد از تشخیص، گمش نکن.
    #   قرمز فقط وقتی برداشته می‌شود که این‌قدر ثانیه پشتِ‌سرهم
    #   شواهدِ مخالف بیاید. یک فریمِ بد حکم را عوض نمی‌کند.
    thief_unlock_s: float = 2.0
    clear_unlock_instant: bool = True   # سبز با یک نشانهٔ پوشش فوراً شکسته شود

    # ---- ★ آهنگِ بازبینی (کلیدِ سرعت) --------------------------------
    # گران‌ترین کار، اجرای طبقه‌بند است. لازم نیست برای همه، در هر فریم
    # اجرا شود. کسی که تازه آمده یا مشکوک است هر فریم بررسی می‌شود؛
    # کسی که حکمش قطعی شده به‌ندرت. نتیجه: هم سریع‌تر، هم تمرکزِ
    # محاسبات روی همان کسی که مهم است.
    cadence_analyzing_s: float = 0.0    # هر فریم — باید سریع تصمیم بگیریم
    cadence_focus_s: float = 0.0        # هر فریم — تمرکز روی فردِ مشکوک
    cadence_thief_s: float = 0.50       # قرمز قطعی شده، فقط پایش
    cadence_clear_s: float = 2.00       # سبز — ارزان، ولی هرگز «هیچ‌وقت»

    # ---- ★ حافظهٔ ردیابی — گم نکردن فرد -------------------------------
    # اگر ردیاب شناسهٔ فرد را عوض کند (انسداد، تقاطع، خروج و ورود)،
    # با مقایسهٔ هیستوگرامِ رنگِ بدن او را به هویتِ قبلی وصل می‌کنیم و
    # حکمِ قبلی حفظ می‌شود. بدونِ این، دزدِ قرمز پس از یک انسداد
    # دوباره از صفر شروع می‌کند.
    track_memory_s: float = 12.0        # چند ثانیه هویتِ گم‌شده را نگه داریم
    reid_match_th: float = 0.72         # آستانهٔ شباهتِ هیستوگرام

    # ---- وزنِ سیگنال‌ها ----
    w_model: float = 1.00          # مدلِ فاین‌تیون‌شده — ستونِ اصلی
    w_kpt: float = 0.45            # هندسهٔ کی‌پوینت

    # ★★ سرنخِ پوست پیش‌فرض **خاموش** است. دلیلش یک آزمونِ عینی:
    #
    #   حالت                 پیشانی  پایین   حکمِ سرنخ
    #   بالاکلاوا              0.00   0.00   full_cover  ✔
    #   حجاب + ماسکِ پزشکی     0.12   0.04   full_cover  ✘ آلارمِ کاذب
    #
    #   این سرنخ فرض می‌کند «پیشانیِ پوشیده = نیتِ مجرمانه». روسری هم
    #   پیشانی را می‌پوشاند و هیچ راهی ندارد پارچهٔ روسری را از پارچهٔ
    #   بالاکلاوا تشخیص دهد. در ایران این یعنی سیستمی که مدام به زنانِ
    #   محجبه آلارم می‌دهد — عملاً غیرقابل استفاده.
    #
    #   کدش حذف نشده و برای گزارش در کارتِ قضاوت محاسبه می‌شود (مفید
    #   است بدانی مدل چه دیده)، ولی در تصمیم دخالت نمی‌کند.
    #   تفکیکِ نوعِ پوشش حالا کارِ مدلِ صفر-شات است که معنا می‌فهمد.
    w_skin: float = 0.00
    w_zeroshot: float = 0.50

    # ---- نمایش ----
    draw_skeleton: bool = True
    save_crops: bool = True


# کلاس‌های داخلی — خروجیِ سیگنال‌ها روی این سه‌تا حساب می‌شود
CLEAR, MEDICAL, FULL = "clear", "medical_mask", "full_cover"
CLASSES = [CLEAR, MEDICAL, FULL]

# ═══ وضعیت‌های *نمایشی* — دقیقاً چهار رنگ ═══════════════════════════════
#
# عمداً از کلاس‌های داخلی جدا شده‌اند. مدل با سه کلاس کار می‌کند، ولی
# آنچه اپراتور می‌بیند باید ساده و بی‌ابهام باشد: چهار رنگ، چهار معنی.
# «ماسک پزشکی»، «پشت به دوربین» و «هنوز مطمئن نیستم» همگی از دیدِ
# اپراتور یک چیزند: مشکوک/نامعلوم → خاکستری.
ST_ANALYZING = "analyzing"     # در حال بررسی
ST_CLEAR     = "clear"         # صورتِ باز
ST_SUSPECT   = "suspicious"    # مشکوک یا نامعلوم (شاملِ ماسکِ پزشکی)
ST_THIEF     = "thief"         # پوششِ کاملِ صورت → آلارم

# ★ پالتِ نسخهٔ اولِ پروژه — به انتخابِ خودت. کادر، اسکلت و برچسب
#   همگی با همین رنگ کشیده می‌شوند.
COLORS = {                      # BGR
    ST_ANALYZING: (160, 160, 160),   # خاکستری — در حال بررسی
    ST_CLEAR:     (0, 200, 0),       # سبز     — صورتِ باز
    ST_SUSPECT:   (0, 140, 255),     # نارنجی  — پوششِ بی‌خطر / نامعلوم
    ST_THIEF:     (0, 0, 255),       # قرمز    — پوششِ کاملِ صورت
}
LABELS = {
    ST_ANALYZING: "Analyzing...",
    ST_CLEAR:     "Clear",
    ST_SUSPECT:   "Covered / Unclear",
    ST_THIEF:     "SUSPICIOUS - ALERT",
}
# شدت — برای مرتب‌کردنِ صفحهٔ قضاوت
SEVERITY = {ST_THIEF: 0, ST_SUSPECT: 1, ST_ANALYZING: 2, ST_CLEAR: 3}

# اندیس‌های COCO-17
NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4
LSHO, RSHO, LELB, RELB, LWRI, RWRI = 5, 6, 7, 8, 9, 10
LHIP, RHIP = 11, 12

SKELETON = [(LEYE, REYE), (NOSE, LEYE), (NOSE, REYE), (LEAR, LEYE), (REAR, REYE),
            (LSHO, RSHO), (LSHO, LELB), (LELB, LWRI), (RSHO, RELB), (RELB, RWRI),
            (NOSE, LSHO), (NOSE, RSHO)]


def norm3(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, v) for v in d.values())
    if s <= 1e-9:
        return {c: 1.0 / len(CLASSES) for c in CLASSES}
    return {k: max(0.0, v) / s for k, v in d.items()}


# ═══════════════════════════════════════════════════════════════════════
#  ۲) جهتِ سر — چیرالیتهٔ کی‌پوینت‌ها
# ═══════════════════════════════════════════════════════════════════════
#
# کی‌پوینت‌های COCO برچسبِ آناتومیک دارند: «شانهٔ چپِ شخص».
# پس علامتِ  x(چپ) − x(راست)  می‌گوید فرد رو به دوربین است یا پشت.
# این سیگنال به دیده‌شدنِ صورت کاری ندارد، پس برای نقاب‌دار هم کار
# می‌کند — و همین، تفاوتش با گیتِ کی‌پوینتیِ نسخهٔ اول است.

def _pair(kxy, kc, il, ir, th, expect=None):
    if kc[il] < th or kc[ir] < th:
        return None
    dx = float(kxy[il][0] - kxy[ir][0])
    sep = float(np.hypot(kxy[il][0] - kxy[ir][0], kxy[il][1] - kxy[ir][1]))
    if sep < 2.0:
        return None
    v = float(np.clip(dx / (0.70 * sep), -1, 1))
    w = float(min(kc[il], kc[ir]))
    if expect and expect > 1e-3:
        # کوتاه‌شدگی: در نیم‌رخ دو نقطه روی هم می‌افتند → رأی کم‌وزن می‌شود
        w *= float(np.clip(sep / expect, 0.0, 1.0))
    return v, w


def orientation(kxy, kc, th=0.35) -> Tuple[float, float]:
    """(facing از ‎−۱‎ تا ‎+۱‎ ، اطمینان ۰..۱)"""
    torso = None
    if kc[LSHO] >= th and kc[RSHO] >= th and (kc[LHIP] >= th or kc[RHIP] >= th):
        smid = (kxy[LSHO] + kxy[RSHO]) / 2
        hips = [kxy[i] for i in (LHIP, RHIP) if kc[i] >= th]
        hmid = np.mean(hips, axis=0)
        d = float(np.hypot(*(smid - hmid)))
        torso = d if d > 5 else None
    body = 0.65 * torso if torso else None

    votes = []
    for (il, ir, wt, ex) in ((LSHO, RSHO, 1.00, body),
                             (LEAR, REAR, 0.85, 0.40 * body if body else None),
                             (LEYE, REYE, 0.70, 0.16 * body if body else None)):
        r = _pair(kxy, kc, il, ir, th, ex)
        if r:
            votes.append((r[0], wt * r[1]))

    # سرنخِ دیده‌شدن — عمداً کم‌وزن، وگرنه نقاب‌دار «پشت» تشخیص داده می‌شود
    eyes = (kc[LEYE] + kc[REYE]) / 2
    ears = (kc[LEAR] + kc[REAR]) / 2
    if eyes > 0.05 or kc[NOSE] > 0.05 or ears > 0.05:
        votes.append((float(np.clip((0.6 * eyes + 0.6 * kc[NOSE] - 0.5 * ears) * 1.6,
                                    -1, 1)), 0.30))
    if not votes:
        return 0.0, 0.0

    ws = sum(w for _, w in votes)
    facing = sum(v * w for v, w in votes) / max(1e-6, ws)
    decisive = sum(abs(v) * w for v, w in votes) / max(1e-6, ws)
    conf = float(np.clip(min(1.0, ws / 1.6) * (0.35 + 0.65 * decisive), 0, 1))
    return float(facing), conf


# ═══════════════════════════════════════════════════════════════════════
#  ۳) کروپِ صورت — تنگ برای مدل، پهن برای چشمِ تو
# ═══════════════════════════════════════════════════════════════════════

def head_box(kxy, kc, bbox, W, H, cfg: Cfg) -> Tuple[np.ndarray, float, str]:
    """(جعبهٔ سر xyxy ، اعتبار ۰..۱ ، منبع)"""
    th = cfg.kpt_conf

    # بهترین حالت: هر دو چشم پیدا → مقیاسِ بسیار پایدار
    if kc[LEYE] >= th and kc[REYE] >= th:
        le, re = kxy[LEYE], kxy[REYE]
        d = float(np.hypot(*(le - re)))
        # ★ جبرانِ کوتاه‌شدگی.
        #   فاصلهٔ دو چشم در تصویر با cos(زاویهٔ چرخشِ سر) کوچک می‌شود.
        #   اندازه‌گیری کردم: از روبه‌رو تا ۵۵ درجه، عرضِ کادر از ۶۰ به
        #   ۲۱ پیکسل می‌افتاد — کادر از صورت بیرون می‌زد و مو و
        #   پس‌زمینه واردش می‌شد. همان چیزی که سرنخ‌ها را گمراه می‌کرد.
        #   عرضِ شانه یک مقیاسِ مستقل است؛ کفِ اندازه را از آن می‌گیریم.
        if kc[LSHO] >= th and kc[RSHO] >= th:
            sw = float(np.hypot(*(kxy[LSHO] - kxy[RSHO])))
            if sw > 8:
                d = max(d, 0.135 * sw)      # نسبتِ انسانی: فاصلهٔ چشم ≈ ۰.۱۶ عرضِ شانه
        if d >= 3:
            w = cfg.tight_scale * d
            h = w * 1.25
            cx = (le[0] + re[0]) / 2
            cy = (le[1] + re[1]) / 2 + 0.10 * h   # چشم‌ها ≈ ۴۲٪ ارتفاعِ سر از بالا
            return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2]), 0.95, "eyes"

    # حالتِ نقاب‌دار: چشم پیدا نیست ولی شانه‌ها هستند
    if kc[LSHO] >= th and kc[RSHO] >= th:
        ls, rs = kxy[LSHO], kxy[RSHO]
        sw = float(np.hypot(*(ls - rs)))
        if sw >= 8:
            mid = (ls + rs) / 2
            vx, vy = rs[0] - ls[0], rs[1] - ls[1]
            n = np.array([vy, -vx], np.float32)
            n /= max(1e-6, np.linalg.norm(n))
            if kc[NOSE] >= th:
                if float(np.dot(kxy[NOSE] - mid, n)) < 0:
                    n = -n
            elif n[1] > 0:
                n = -n
            w = 0.50 * sw
            h = w * 1.25
            c = mid + n * 0.42 * sw
            return np.array([c[0]-w/2, c[1]-h/2, c[0]+w/2, c[1]+h/2]), 0.75, "shoulders"

    # آخرین راه‌حل: ۲۰٪ بالای جعبهٔ بدن
    x1, y1, x2, y2 = bbox
    h = 0.20 * (y2 - y1)
    w = min((x2 - x1) * 0.85, h / 1.25)
    return np.array([(x1+x2)/2 - w/2, y1, (x1+x2)/2 + w/2, y1 + h]), 0.35, "bbox"


def _crop(img, box):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return np.zeros((0, 0, 3), np.uint8)
    return img[y1:y2, x1:x2].copy()


def eye_roll(kxy, kc, th=0.40) -> float:
    """
    زاویهٔ کج‌شدگیِ سر.

    ★ باگِ نسخهٔ اول اینجا بود: از (چشمِ راستِ شخص − چشمِ چپِ شخص)
    استفاده می‌کرد. چون چشمِ چپِ شخص در سمتِ راستِ تصویر است، این
    اختلاف برای صورتِ عادیِ روبه‌رو منفی می‌شد و زاویه ۱۸۰ درجه
    درمی‌آمد → تصویر وارونه می‌شد و کروپ به‌جای صورت، موی سر را
    برمی‌داشت. اینجا بر اساس مختصاتِ x در *تصویر* مرتب می‌کنیم.
    """
    if kc[LEYE] < th or kc[REYE] < th:
        return 0.0
    a, b = kxy[LEYE], kxy[REYE]
    l, r = (a, b) if a[0] <= b[0] else (b, a)
    ang = math.degrees(math.atan2(r[1] - l[1], r[0] - l[0]))
    return ang if abs(ang) <= 45 else 0.0


def make_crops(frame, hbox, kxy, kc, cfg: Cfg):
    """(کروپِ تنگِ ترازشده ، کروپِ پهن) — هر دو از فریمِ تمیز."""
    H, W = frame.shape[:2]
    ang = eye_roll(kxy, kc)

    src = frame
    box = hbox.copy()
    if abs(ang) >= 2.0:
        cx, cy = (hbox[0] + hbox[2]) / 2, (hbox[1] + hbox[3]) / 2
        M = cv2.getRotationMatrix2D((float(cx), float(cy)), ang, 1.0)
        src = cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    tight = _crop(src, box)

    # کروپِ پهن: هم‌مرکز، بزرگ‌تر، بدونِ چرخش (طبیعی‌تر دیده می‌شود)
    cx, cy = (hbox[0] + hbox[2]) / 2, (hbox[1] + hbox[3]) / 2
    ww = (hbox[2] - hbox[0]) * cfg.wide_scale
    wh = (hbox[3] - hbox[1]) * cfg.wide_scale
    wide = _crop(frame, [cx - ww/2, cy - wh/2 * 0.9, cx + ww/2, cy + wh/2 * 1.1])
    return tight, wide


def sharpness(img) -> float:
    if img.size == 0:
        return 0.0
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if min(g.shape) > 160:
        g = cv2.resize(g, (160, 160), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


# ═══════════════════════════════════════════════════════════════════════
#  ۴) سیگنالِ پوست — تفکیکِ ماسکِ پزشکی از ماسکِ دزدی
# ═══════════════════════════════════════════════════════════════════════

def skin_mask(bgr) -> np.ndarray:
    """
    ترکیبِ دو فضای رنگی. HSV به‌تنهایی روی پوستِ تیره ضعیف است؛
    YCrCb مکملِ خوبی است و اشتراکشان خطای کاذب را کم می‌کند.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 15, 40], np.uint8),
                     np.array([28, 190, 255], np.uint8))
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    m2 = cv2.inRange(ycc, np.array([0, 133, 77], np.uint8),
                     np.array([255, 180, 130], np.uint8))
    return cv2.bitwise_and(m1, m2)


def is_grayscale(bgr, th: float = 12.0) -> bool:
    """تصویرِ مادون‌قرمزِ شبانه عملاً خاکستری است → سرنخِ پوست بی‌معنی می‌شود."""
    if bgr.size == 0:
        return True
    s = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1]
    return float(s.mean()) < th


def skin_signal(tight) -> Tuple[Optional[Dict[str, float]], float, Dict[str, float]]:
    """
    مدلِ فیزیکیِ مسئله:

        پیشانی پوست | پایین پوست  →  صورتِ باز
        پیشانی پوست | پایین پارچه →  ماسکِ پزشکی
        پیشانی پارچه| پایین پارچه →  ★ ماسکِ دزدی

    خروجی: (توزیع روی سه کلاس یا None ، وزن ، جزئیات برای عیب‌یابی)
    """
    if tight is None or tight.size == 0 or min(tight.shape[:2]) < 16:
        return None, 0.0, {}
    if is_grayscale(tight):
        # شب / IR — این سرنخ ساکت می‌ماند به‌جای اینکه اشتباه بگوید
        return None, 0.0, {"gray": 1.0}

    h, w = tight.shape[:2]
    m = skin_mask(tight) > 0

    # نوارها نسبت به کادرِ سر. کناره‌ها بریده می‌شود تا مو و زمینه
    # کمتر وارد محاسبه شود.
    x0, x1 = int(0.20 * w), int(0.80 * w)
    fore = m[int(0.10 * h):int(0.34 * h), x0:x1]      # پیشانی
    low = m[int(0.58 * h):int(0.95 * h), x0:x1]       # بینی، دهان، چانه

    if fore.size < 12 or low.size < 12:
        return None, 0.0, {}
    f = float(fore.mean())
    l = float(low.mean())

    # توابعِ نرم به‌جای آستانهٔ سخت — خروجی پیوسته و کم‌نوسان می‌شود
    def hi(v, a=0.16, b=0.42):
        return float(np.clip((v - a) / (b - a), 0, 1))

    fore_skin, low_skin = hi(f), hi(l)
    p = {
        CLEAR:   fore_skin * low_skin,
        MEDICAL: fore_skin * (1 - low_skin),
        FULL:    (1 - fore_skin) * (1 - low_skin),
    }
    # اگر پیشانی پوشیده ولی پایین باز است (کلاه/هودی ساده)، شاهدِ دزدی نیست
    p[CLEAR] += (1 - fore_skin) * low_skin * 0.7
    p[MEDICAL] += (1 - fore_skin) * low_skin * 0.3

    # هرچه دو ناحیه قاطع‌تر باشند، سرنخ قابل‌اتکاتر است
    weight = float(np.clip(0.35 + 0.65 * max(abs(fore_skin - 0.5),
                                             abs(low_skin - 0.5)) * 2, 0, 1))
    return norm3(p), weight, {"forehead": round(f, 3), "lower": round(l, 3),
                              "fore_skin": round(fore_skin, 2),
                              "low_skin": round(low_skin, 2)}


# ═══════════════════════════════════════════════════════════════════════
#  ۵) سیگنالِ کی‌پوینت — «چشم‌ها پیدا، بینی نه»
# ═══════════════════════════════════════════════════════════════════════

def frontal_score(kxy, kc, facing: float, ocf: float, th: float = 0.35) -> float:
    """
    ★ گیتِ ترکیبی — «چقدر مطمئنیم این فرد واقعاً رو به دوربین است؟»

    دو منبعِ مستقل، و عمداً هر دو لازم‌اند:

      الف) سبکِ نسخهٔ اول — تقارنِ چشم‌ها.
           در نیم‌رخ، چشمِ دور محو می‌شود. پس *کمینهٔ* اطمینانِ دو چشم
           (نه میانگینشان) یک سنجهٔ مستقیم از روبه‌رو بودن است.
           میانگین فریب می‌خورد: چشمِ نزدیک ۰.۹ و دور ۰.۱ میانگینش ۰.۵
           می‌شود که به‌نظر قابل قبول است، ولی کمینه‌اش ۰.۱ است.

      ب) سبکِ نسخهٔ دوم — چیرالیته.
           برای نقاب‌دار هم کار می‌کند، چون به صورت کاری ندارد.

    ضرب می‌شوند نه جمع: اگر هرکدام بگوید «مطمئن نیستم»، نتیجه پایین
    می‌آید. برای آلارمِ قرمز، محافظه‌کاری درست است.

    خروجی ۰..۱
    """
    eye_min = float(min(kc[LEYE], kc[REYE]))          # (الف) کمینه، نه میانگین
    a = float(np.clip((eye_min - 0.15) / 0.45, 0.0, 1.0))

    # تقارنِ افقی: در نیم‌رخ، فاصلهٔ چشم‌ها نسبت به عرضِ شانه فرو می‌ریزد
    sym = 1.0
    if kc[LSHO] >= th and kc[RSHO] >= th and eye_min >= 0.15:
        sw = float(np.hypot(*(kxy[LSHO] - kxy[RSHO])))
        ed = float(np.hypot(*(kxy[LEYE] - kxy[REYE])))
        if sw > 8:
            sym = float(np.clip((ed / sw) / 0.16, 0.0, 1.0))

    # (ب) چیرالیته: فقط بخشِ «رو به دوربین» را می‌خواهیم
    b = float(np.clip((facing - 0.10) / 0.60, 0.0, 1.0)) * float(np.clip(ocf / 0.5, 0, 1))

    return float(np.clip((0.55 * a + 0.45 * sym) * (0.45 + 0.55 * b), 0.0, 1.0))


def kpt_signal(kc, facing: float) -> Tuple[Optional[Dict[str, float]], float]:
    """
    همان نکته‌ای که در عمل دیدی: هرکس ماسک زده، چشمانش پیداست ولی
    بینی و دهانش نه. مدلِ ژست دقیقاً همین را گزارش می‌کند.

    این سیگنال به رنگ و نور کاری ندارد، پس شب و در تصویرِ IR هم
    کار می‌کند — مکملِ کاملِ سرنخِ پوست.

    نکته: این سیگنال «پوشیده بودن» را می‌گوید ولی *نوعِ* پوشش را نه
    (بالاکلاوا هم شکافِ چشم دارد). تفکیکِ نوع کارِ سرنخِ پوست است.
    """
    if facing < 0.1:
        return None, 0.0                       # نیم‌رخ: نبودِ بینی طبیعی است
    eyes = (kc[LEYE] + kc[REYE]) / 2.0
    nose = float(kc[NOSE])
    if eyes < 0.35:
        return None, 0.0                       # چشم‌ها را نداریم → حرفی نداریم

    if nose < 0.25:
        # چشم پیدا، بینی پوشیده → قطعاً چیزی روی صورت هست
        p = {CLEAR: 0.06, MEDICAL: 0.52, FULL: 0.42}
        w = float(np.clip((eyes - 0.35) / 0.5, 0, 1)) * 0.9
    elif nose >= 0.55:
        p = {CLEAR: 0.78, MEDICAL: 0.15, FULL: 0.07}
        w = float(np.clip((nose - 0.55) / 0.35, 0, 1)) * 0.8
    else:
        p = {CLEAR: 0.45, MEDICAL: 0.35, FULL: 0.20}
        w = 0.3
    return norm3(p), float(w)


# ═══════════════════════════════════════════════════════════════════════
#  ۶) انباشتِ زمانی — رأی‌گیریِ وزن‌دار با محوشدن
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Track:
    tid: int
    scores: Dict[str, float] = field(default_factory=lambda: {c: 0.0 for c in CLASSES})
    n: int = 0
    last_t: float = 0.0
    last_analyzed: float = -1e9
    state: str = ST_ANALYZING
    conf: float = 0.0
    focus: bool = False            # ★ حالتِ تمرکز — هر فریم بررسی می‌شود
    locked_t: float = 0.0          # از کِی در وضعیتِ قفل‌شده است
    contrary_since: float = -1.0   # از کِی شواهدِ مخالفِ حکم می‌آید
    best_q: float = -1.0
    best_wide: Optional[np.ndarray] = None
    best_tight: Optional[np.ndarray] = None
    best_detail: dict = field(default_factory=dict)
    first_t: float = 0.0
    alerted: bool = False
    # ★ بهترین دیدِ روبه‌رویی که تا حالا از این فرد داشته‌ایم.
    #   حکمِ قرمز به این وابسته است، نه به فریمِ جاری: کسی که یک بار
    #   واضح روبه‌رو دیده شده و نقاب داشته، با چرخاندنِ سر از آلارم
    #   فرار نمی‌کند.
    best_frontal: float = 0.0
    hist: Optional[np.ndarray] = None    # امضای رنگیِ بدن، برای وصلِ مجدد
    inherited_from: Optional[int] = None


def body_signature(crop: np.ndarray) -> Optional[np.ndarray]:
    """
    امضای رنگیِ بدن — هیستوگرامِ HS در سه نوارِ افقی.

    چرا سه نوار: لباسِ بالاتنه، پایین‌تنه و کفش معمولاً رنگ‌های
    متفاوتی دارند. یک هیستوگرامِ واحد این ساختار را از دست می‌دهد و
    دو نفر با رنگ‌های مشابه ولی چیدمانِ متفاوت را یکی می‌بیند.

    عمداً ارزان است (~۰.۱ms). هدف، تشخیصِ چهره نیست؛ فقط «همان کسی
    است که چند لحظه پیش گم شد؟».
    """
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 12:
        return None
    hsv = cv2.cvtColor(cv2.resize(crop, (48, 96)), cv2.COLOR_BGR2HSV)
    feats = []
    for i in range(3):
        band = hsv[i * 32:(i + 1) * 32]
        h = cv2.calcHist([band], [0, 1], None, [12, 6], [0, 180, 0, 256])
        h = cv2.normalize(h, h).flatten()
        feats.append(h)
    v = np.concatenate(feats).astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else None


class Decider:
    """
    موتورِ تصمیم با «تشدیدِ تدریجی».

    مسیرِ یک فرد:

        تازه وارد  ──(۲ مشاهده)──►  حکمِ اولیه
                                      │
                        ┌─────────────┼──────────────┐
                        ▼             ▼              ▼
                     سبز          مشکوک           سبز/خاکستری
                  (بازبینیِ      ★ تمرکز:
                   هر ۲ ثانیه)   هر فریم بررسی
                                      │
                              (confirm_votes مشاهده)
                                      ▼
                                 قرمز — چسبنده

    دو خاصیتِ مهم:

    ۱) **سرعت.** فردِ تازه‌وارد و فردِ مشکوک هر فریم بررسی می‌شوند، ولی
       کسی که حکمش قطعی شده به‌ندرت. در صحنه‌ای که بیشتر افراد عادی‌اند،
       این یعنی طبقه‌بند چند برابر کمتر اجرا می‌شود و کلِ خط لوله
       سریع‌تر می‌شود — یعنی سرعت از *حذفِ کارِ بی‌فایده* می‌آید، نه
       از کم‌کردنِ دقت.

    ۲) **چسبندگی.** حکمِ قرمز با یک فریمِ بد برداشته نمی‌شود؛ باید
       چند ثانیه پشتِ‌سرهم شواهدِ مخالف بیاید. برعکس، سبز با اولین
       نشانهٔ پوشش فوراً شکسته می‌شود — چون خطای «دزد را سبز دیدیم»
       بسیار پرهزینه‌تر از خطای «بی‌گناه را دوباره بررسی کردیم» است.
    """
    LOG_FLOOR = -4.5

    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.tracks: Dict[int, Track] = {}
        self.memory: List[Track] = []     # هویت‌های اخیراً گم‌شده
        self.reattached = 0

    # ---------------------------------------------------------------- #
    def get(self, tid: int, t: float) -> Track:
        tr = self.tracks.get(tid)
        if tr is None:
            tr = Track(tid=tid, last_t=t, first_t=t)
            self.tracks[tid] = tr
        else:
            dt = max(0.0, t - tr.last_t)
            if dt > 0:
                f = self.cfg.decay ** dt
                for c in CLASSES:
                    tr.scores[c] *= f
            tr.last_t = t
        return tr

    # ---------------------------------------------------------------- #
    def retire_lost(self, t: float, alive: set):
        """ردیابی‌هایی که این فریم دیده نشدند به حافظهٔ کوتاه‌مدت می‌روند."""
        for tid in list(self.tracks):
            if tid in alive:
                continue
            tr = self.tracks[tid]
            if (t - tr.last_t) > 0.7:               # چند فریم مهلت
                self.memory.append(tr)
                del self.tracks[tid]
        cutoff = t - self.cfg.track_memory_s
        self.memory = [m for m in self.memory if m.last_t >= cutoff][-40:]

    def try_reattach(self, tr: Track, sig: Optional[np.ndarray], t: float) -> bool:
        """
        ★ گم‌نکردنِ فرد.

        وقتی ردیاب یک شناسهٔ تازه می‌سازد (که در انسداد و تقاطع مدام
        اتفاق می‌افتد)، اینجا می‌پرسیم: آیا این همان کسی است که چند
        لحظه پیش گم شد؟ اگر بله، حکم و شواهدِ قبلی را به ارث می‌برد و
        از صفر شروع نمی‌کند.

        نکتهٔ زمان‌بندی که اول از قلم افتاده بود: ردیاب معمولاً شناسهٔ
        جدید را در *همان فریمی* می‌سازد که شناسهٔ قبلی ناپدید می‌شود.
        اگر فقط در «حافظهٔ گم‌شده‌ها» بگردیم، آن هویت هنوز آنجا نرفته
        و هیچ‌وقت وصل نمی‌شود. پس ردیابی‌هایی را هم که *در این فریم
        دیده نشده‌اند* نامزد می‌گیریم.
        """
        if sig is None or tr.n > 0:
            return False
        cands = list(self.memory)
        for o in self.tracks.values():
            if o.tid != tr.tid and o.last_t < t - 1e-6 and o.n > 0:
                cands.append(o)
        if not cands:
            return False
        best, best_s = None, 0.0
        for m in cands:
            if m.hist is None:
                continue
            sim = float(np.dot(sig, m.hist))
            if sim > best_s:
                best, best_s = m, sim
        if best is None or best_s < self.cfg.reid_match_th:
            return False
        # وراثت: شواهد، حکم، بهترین شات
        tr.scores = dict(best.scores)
        tr.n = best.n
        tr.state, tr.conf = best.state, best.conf
        tr.focus, tr.locked_t = best.focus, best.locked_t
        tr.alerted = best.alerted
        tr.best_q, tr.best_wide = best.best_q, best.best_wide
        tr.best_tight, tr.best_detail = best.best_tight, best.best_detail
        tr.inherited_from = best.tid
        tr.hist = best.hist
        if best in self.memory:
            self.memory.remove(best)
        else:
            # هویتِ قبلی هنوز در جدولِ فعال بود — حالا جایش را به این داد
            self.tracks.pop(best.tid, None)
        self.reattached += 1
        return True

    # ---------------------------------------------------------------- #
    def due(self, tid: int, t: float) -> bool:
        """آیا این فرد در این فریم باید به طبقه‌بند برود؟"""
        tr = self.tracks.get(tid)
        if tr is None:
            return True
        c = self.cfg
        if tr.state == ST_THIEF:
            period = c.cadence_thief_s
        elif tr.focus or tr.state == ST_ANALYZING:
            period = c.cadence_focus_s          # ★ تمرکز: هر فریم
        elif tr.state == ST_CLEAR:
            period = c.cadence_clear_s
        else:
            period = c.cadence_focus_s          # مشکوک هم هر فریم
        return (t - tr.last_analyzed) >= period

    # ---------------------------------------------------------------- #
    def observe(self, tid: int, p: Dict[str, float], weight: float, t: float):
        if weight <= 0:
            return
        tr = self.get(tid, t)
        for c in CLASSES:
            tr.scores[c] += weight * max(self.LOG_FLOOR,
                                         math.log(max(p.get(c, 0.0), 1e-9)))
        mx = max(tr.scores.values())
        for c in CLASSES:
            tr.scores[c] -= mx
        tr.n += 1
        tr.last_analyzed = t

    def posterior(self, tid: int) -> Dict[str, float]:
        tr = self.tracks.get(tid)
        if tr is None:
            return {c: 1.0 / 3 for c in CLASSES}
        e = {c: math.exp(tr.scores[c]) for c in CLASSES}
        s = sum(e.values()) or 1.0
        return {c: e[c] / s for c in CLASSES}

    # ---------------------------------------------------------------- #
    def decide(self, tid: int, is_back: bool, t: float,
               frontal: Optional[float] = None) -> Tuple[str, float]:
        tr = self.tracks[tid]
        c = self.cfg
        if frontal is not None:
            tr.best_frontal = max(tr.best_frontal, float(frontal))

        # پشت به دوربین → «نامعلوم». حکمِ قبلی حفظ می‌شود اگر قطعی بوده،
        # چون کسی که قرمز بود و برگشت، هنوز همان آدم است.
        if is_back:
            if tr.state not in (ST_THIEF, ST_CLEAR):
                tr.state, tr.conf = ST_SUSPECT, 0.0
            return tr.state, tr.conf

        if tr.n < c.min_votes:
            if tr.state == ST_ANALYZING:
                tr.conf = 0.0
            return tr.state, tr.conf

        p = self.posterior(tid)
        pf, pc, pm = p[FULL], p[CLEAR], p[MEDICAL]

        # ── حکمِ قرمزِ چسبنده ────────────────────────────────────────────
        if tr.state == ST_THIEF:
            if pf < c.suspect_exit:
                if tr.contrary_since < 0:
                    tr.contrary_since = t
                elif (t - tr.contrary_since) >= c.thief_unlock_s:
                    tr.state, tr.focus, tr.contrary_since = ST_SUSPECT, True, -1.0
            else:
                tr.contrary_since = -1.0
            tr.conf = pf
            return tr.state, tr.conf

        # ── ★ ارتقا به قرمز: سه شرط، هر سه لازم ────────────────────────
        #   ۱) احتمالِ پوششِ کامل از آستانه بیشتر باشد
        #   ۲) به‌اندازهٔ کافی مشاهده جمع شده باشد
        #   ۳) ★ حداقل یک بار او را به‌قدرِ کافی «روبه‌رو» دیده باشیم
        #
        # شرط سوم همان چیزی است که آلارمِ کاذبِ نیم‌رخ را حذف می‌کند.
        # در نیم‌رخ، کادرِ سر روی مو و پس‌زمینه می‌افتد و سرنخ‌ها به‌غلط
        # «پوشیده» می‌گویند. حالا هرچقدر هم مطمئن باشند، تا وقتی صورت
        # واقعاً رو به دوربین نباشد حکمِ قرمز صادر نمی‌شود — فرد در
        # حالتِ «نامعلوم» می‌ماند تا لحظه‌ای که رویش را برگرداند.
        if pf >= c.suspect_enter and tr.n >= c.confirm_votes:
            if tr.best_frontal >= c.red_min_frontal:
                tr.state, tr.conf = ST_THIEF, pf
                tr.focus, tr.locked_t, tr.contrary_since = False, t, -1.0
                return tr.state, tr.conf
            # شواهد هست ولی زاویه اجازه نمی‌دهد → مشکوک بمان و تمرکز کن
            tr.state, tr.conf, tr.focus = ST_SUSPECT, pf, True
            return tr.state, tr.conf

        # ── مشکوک → تمرکز روی همین فرد تا قطعی شود ─────────────────────
        if pf >= c.watch_enter:
            tr.state, tr.conf, tr.focus = ST_SUSPECT, pf, True
            return tr.state, tr.conf

        # ── سبزِ چسبنده، ولی با شکستِ فوری ──────────────────────────────
        if tr.state == ST_CLEAR:
            if c.clear_unlock_instant and pf >= c.watch_enter:
                tr.state, tr.focus = ST_SUSPECT, True
            else:
                tr.conf = pc
            return tr.state, tr.conf

        if pc >= c.clear_enter:
            tr.state, tr.conf, tr.focus = ST_CLEAR, pc, False
            return tr.state, tr.conf

        # ماسکِ پزشکی یا هر پوششِ بی‌خطر → خاکستری
        if pm >= 0.45:
            tr.state, tr.conf, tr.focus = ST_SUSPECT, pm, False
            return tr.state, tr.conf

        tr.state, tr.conf = ST_ANALYZING, max(p.values())
        return tr.state, tr.conf


# ═══════════════════════════════════════════════════════════════════════
#  ۷) مدل‌ها
# ═══════════════════════════════════════════════════════════════════════

def _to_batch(crops, size, mean, std, device, half):
    """پیش‌پردازشِ سریع: resize با OpenCV، نرمال‌سازی روی GPU."""
    import torch
    arr = np.empty((len(crops), size, size, 3), np.float32)
    for i, c in enumerate(crops):
        if c is None or c.size == 0:
            arr[i] = 0.0
            continue
        interp = cv2.INTER_AREA if max(c.shape[:2]) > size else cv2.INTER_LINEAR
        r = cv2.resize(c, (size, size), interpolation=interp)
        arr[i] = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).to(device).permute(0, 3, 1, 2).contiguous()
    m = torch.as_tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.as_tensor(std, device=device).view(1, 3, 1, 1)
    t = (t - m) / s
    return t.half() if half else t


class MaskModel:
    """
    ★ ستونِ اصلیِ تصمیم — همان مدلی که نسخهٔ اولِ پروژه را کار می‌انداخت.

    نگاشتِ لیبل‌ها از خودِ چک‌پوینت خوانده می‌شود. نسخهٔ اول این را
    hard-code کرده بود ({0:"mask",1:"no_mask"}) و اگر ترتیب برعکس بود،
    کلِ سیستم وارونه کار می‌کرد بدون اینکه هیچ خطایی بدهد.
    """

    def __init__(self, cfg: Cfg, device: str):
        import torch
        from transformers import SiglipForImageClassification
        self.device, self.half = device, cfg.half and device.startswith("cuda")
        self.model = SiglipForImageClassification.from_pretrained(cfg.mask_model)
        self.model = self.model.to(device).eval()
        if self.half:
            self.model = self.model.half()
        self.mean = np.array([0.5, 0.5, 0.5], np.float32)
        self.std = np.array([0.5, 0.5, 0.5], np.float32)
        self.size = 224

        id2label = dict(getattr(self.model.config, "id2label", {}) or {})
        self.i_no, self.i_mask = None, None
        for i, lab in id2label.items():
            s = str(lab).lower()
            if any(k in s for k in ("no_mask", "no mask", "nomask", "without", "unmask")):
                self.i_no = int(i)
            elif "mask" in s or "cover" in s:
                self.i_mask = int(i)
        if self.i_no is None or self.i_mask is None:
            self.i_mask, self.i_no = 0, 1
            print(f"[mask] ⚠️ لیبل‌ها گویا نبودند {id2label} — فرض: 0=mask, 1=no_mask")
        else:
            print(f"[mask] نگاشتِ لیبل از چک‌پوینت: {id2label} "
                  f"→ mask={self.i_mask}, no_mask={self.i_no}")

    def __call__(self, crops) -> List[Tuple[float, float]]:
        """[(p_باز, p_پوشیده), ...]"""
        import torch
        if not crops:
            return []
        with torch.inference_mode():
            px = _to_batch(crops, self.size, self.mean, self.std, self.device, self.half)
            logits = self.model(pixel_values=px).logits.float()
            pr = torch.softmax(logits, 1).cpu().numpy()
        return [(float(r[self.i_no]), float(r[self.i_mask])) for r in pr]


# ═══ پرامپت‌ها — چهار کلاس، دو گروه ══════════════════════════════════
#
# ★ کلاسِ «حجاب» عمداً اضافه شده و مهم‌ترین تغییرِ ایمنیِ این نسخه است.
#
# سرنخِ رنگیِ پوست نمی‌توانست پارچهٔ روسری را از پارچهٔ بالاکلاوا جدا
# کند، چون هر دو «پیشانیِ پوشیده»اند. ولی مدلِ صفر-شات *معنا* می‌فهمد:
# «زنی با روسری و ماسک» و «سارقی با بالاکلاوا» دو مفهومِ کاملاً متفاوت‌اند
# و مدل این تفاوت را از میلیون‌ها تصویر آموخته است.
#
# این دقیقاً همان کاری است که صفر-شات در آن قوی است و سرنخِ دست‌ساز
# هرگز نمی‌تواند انجام دهد.
BENIGN = "benign_cover"      # پوششِ بی‌خطر: ماسکِ پزشکی، روسری، حجاب
THREAT = "threat_cover"      # پوششِ مشکوک: بالاکلاوا، اسکی‌ماسک، کلاه‌کاسکت

PROMPTS = {
    BENIGN: [
        # ماسکِ پزشکی
        "a person wearing a surgical medical face mask",
        "a face with a light blue disposable mask, forehead and eyes uncovered",
        "a person wearing a white N95 respirator, eyes and eyebrows visible",
        # ★ حجاب / روسری — با و بدون ماسک
        "a woman wearing a headscarf, her face open and visible",
        "a woman wearing a hijab covering her hair, with a surgical mask on",
        "a person wearing a hood or a winter scarf, face still visible",
        "a woman in a black chador with an uncovered visible face",
    ],
    THREAT: [
        "a person wearing a black balaclava covering the entire face",
        "a robber in a ski mask, only the eyes showing through a narrow slit",
        "a criminal whose whole face is wrapped in dark fabric, no skin visible",
        "a person wearing a full-face motorcycle helmet with the visor down",
        "a masked burglar with the face completely concealed",
    ],
}


class ZeroShot:
    """
    نقشِ محدود و عمدی: فقط پاسخ به «کدام *نوع* پوشش؟»

    مدل‌های صفر-شات در مقایسهٔ **دوتایی** به‌مراتب قابل‌اتکاتر از
    انتخاب یکی از پنج کلاس‌اند. تجربهٔ اجرای قبلی این را نشان داد:
    وقتی همین مدل مجبور بود بین پنج کلاس (شامل «تصویرِ نامشخص») انتخاب
    کند، روی کروپ‌های واقعیِ دوربین مداربسته عملاً بی‌فایده بود.
    """

    def __init__(self, cfg: Cfg, device: str):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.device, self.half = device, cfg.half and device.startswith("cuda")
        self.model = AutoModel.from_pretrained(cfg.zeroshot_model).to(device).eval()
        if self.half:
            self.model = self.model.half()
        tok = AutoTokenizer.from_pretrained(cfg.zeroshot_model)
        self.mean = np.array([0.5, 0.5, 0.5], np.float32)
        self.std = np.array([0.5, 0.5, 0.5], np.float32)
        self.size = 224

        texts, self.slices = [], []
        for c in (BENIGN, THREAT):
            a = len(texts)
            texts.extend(PROMPTS[c])
            self.slices.append((a, len(texts)))
        with torch.no_grad():
            enc = tok(texts, padding="max_length", max_length=64,
                      truncation=True, return_tensors="pt").to(device)
            emb = self._emb(self.model.get_text_features(**enc))
        self.temb = emb / emb.norm(dim=-1, keepdim=True)
        print(f"[zeroshot] آماده — {len(texts)} پرامپت "
              f"({len(PROMPTS[BENIGN])} بی‌خطر شاملِ حجاب / "
              f"{len(PROMPTS[THREAT])} مشکوک)")

    @staticmethod
    def _emb(out):
        """سازگاری با transformers ۴ و ۵ — در نسخهٔ ۵ خروجی شیء است نه تنسور."""
        import torch
        if isinstance(out, torch.Tensor):
            return out.float()
        for a in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
            v = getattr(out, a, None)
            if v is not None:
                return (v.mean(1) if v.dim() == 3 else v).float()
        raise TypeError(f"خروجی ناشناخته: {type(out)}")

    def __call__(self, crops) -> List[float]:
        """نسبتِ «ماسکِ پزشکی» به کلِ پوشش — عددی در ۰..۱"""
        import torch
        if not crops:
            return []
        with torch.inference_mode():
            px = _to_batch(crops, self.size, self.mean, self.std, self.device, self.half)
            ie = self._emb(self.model.get_image_features(pixel_values=px))
            ie = ie / ie.norm(dim=-1, keepdim=True)
            lg = ie @ self.temb.T
            per = torch.stack([lg[:, a:b].mean(1) for a, b in self.slices], 1)
            pr = torch.softmax(per * 20.0, 1).cpu().numpy()
        return [float(r[0]) for r in pr]      # سهمِ MEDICAL


# ═══════════════════════════════════════════════════════════════════════
#  ۸) ترکیبِ سیگنال‌ها
# ═══════════════════════════════════════════════════════════════════════

def combine(p_open: float, p_cov: float, r_med: Optional[float],
            kpt: Optional[Dict[str, float]], kpt_w: float,
            skin: Optional[Dict[str, float]], skin_w: float,
            cfg: Cfg) -> Tuple[Dict[str, float], dict]:
    """
    ترکیبِ لگاریتمی (رأی‌گیریِ لگاریتم-بخت).

    چرا لگاریتمی و نه میانگینِ ساده: اگر دو سرنخِ *مستقل* هر دو
    بگویند «پوشیده»، اطمینان باید از هر دو بیشتر شود، نه اینکه
    میانگینشان گرفته شود. میانگینِ ساده شواهد را رقیق می‌کند.
    """
    acc = {c: 0.0 for c in CLASSES}
    detail = {}

    # ---- ۱) مدلِ فاین‌تیون‌شده: محورِ «باز / پوشیده» ----
    # نسبتِ پزشکی به دزدی از مدلِ صفر-شات می‌آید؛ اگر نبود، نرخِ پایه.
    rm = 0.62 if r_med is None else float(np.clip(r_med, 0.05, 0.95))
    pm = {CLEAR: p_open, MEDICAL: p_cov * rm, FULL: p_cov * (1 - rm)}
    pm = norm3(pm)
    for c in CLASSES:
        acc[c] += cfg.w_model * math.log(max(pm[c], 1e-6))
    detail["model"] = {k: round(v, 3) for k, v in pm.items()}
    detail["p_covered"] = round(p_cov, 3)
    if r_med is not None:
        detail["zs_medical_ratio"] = round(r_med, 3)

    # ---- ۲) هندسهٔ کی‌پوینت ----
    if kpt and kpt_w > 0:
        for c in CLASSES:
            acc[c] += cfg.w_kpt * kpt_w * math.log(max(kpt[c], 1e-6))
        detail["kpt"] = {k: round(v, 2) for k, v in kpt.items()}
        detail["kpt_w"] = round(kpt_w, 2)

    # ---- ۳) هندسهٔ پوست — پیش‌فرض خاموش (w_skin = 0) ----
    # محاسبه‌اش برای گزارش در کارتِ قضاوت مفید است، ولی با وزنِ صفر
    # هیچ اثری روی تصمیم ندارد. دلیلِ خاموشی: حجاب. توضیح کامل در Cfg.
    if skin and skin_w > 0 and cfg.w_skin > 0:
        for c in CLASSES:
            acc[c] += cfg.w_skin * skin_w * math.log(max(skin[c], 1e-6))
        detail["skin"] = {k: round(v, 2) for k, v in skin.items()}
        detail["skin_w"] = round(skin_w, 2)

    mx = max(acc.values())
    e = {c: math.exp(acc[c] - mx) for c in CLASSES}
    s = sum(e.values()) or 1.0
    out = {c: e[c] / s for c in CLASSES}
    detail["fused"] = {k: round(v, 3) for k, v in out.items()}
    return out, detail


# ═══════════════════════════════════════════════════════════════════════
#  ۹) رسم
# ═══════════════════════════════════════════════════════════════════════

def draw_label(img, text, org, color, scale=0.55, th=1, pad=4):
    (tw, t_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th + 1)
    x, y = int(org[0]), max(int(org[1]), t_h + pad * 2)
    cv2.rectangle(img, (x, y - t_h - pad * 2), (x + tw + pad * 2, y), color, -1)
    cv2.putText(img, text, (x + pad, y - pad), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (20, 20, 20), th, cv2.LINE_AA)


def draw_corner_box(img, box, color, th=2, ratio=0.22):
    x1, y1, x2, y2 = [int(v) for v in box]
    c = max(6, int(min(x2 - x1, y2 - y1) * ratio))
    for px, py, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                           (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (px, py), (px + dx * c, py), color, th, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * c), color, th, cv2.LINE_AA)


def draw_person(vis, bbox, hbox, tid, state, conf, kxy, kc, cfg: Cfg):
    col = COLORS[state]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    draw_corner_box(vis, bbox, col, 2)
    cv2.rectangle(vis, (int(hbox[0]), int(hbox[1])),
                  (int(hbox[2]), int(hbox[3])), col, 1, cv2.LINE_AA)
    draw_label(vis, f"#{tid} {LABELS[state]} {conf:.0%}", (x1, y1 - 4), col)

    if cfg.draw_skeleton:
        # اسکلت دقیقاً با همان رنگِ حکم کشیده می‌شود، ولی ضخامتش به
        # اهمیت بستگی دارد: دزد پررنگ و ضخیم، فردِ عادی نازک و کم‌رنگ.
        # این‌طور چشم بلافاصله می‌رود سراغِ کسی که مهم است، به‌جای
        # اینکه با انبوهِ خط‌های هم‌وزن گیج شود.
        thick = 3 if state == ST_THIEF else (2 if state == ST_SUSPECT else 1)
        for a, b in SKELETON:
            if kc[a] >= cfg.kpt_conf and kc[b] >= cfg.kpt_conf:
                cv2.line(vis, tuple(map(int, kxy[a])), tuple(map(int, kxy[b])),
                         col, thick, cv2.LINE_AA)
        for i in (NOSE, LEYE, REYE, LEAR, REAR, LSHO, RSHO, LELB, RELB, LWRI, RWRI):
            if kc[i] >= cfg.kpt_conf:
                # مچ‌ها بزرگ‌تر — ورودیِ ماژولِ اسلحه در فازهای بعدی
                cv2.circle(vis, tuple(map(int, kxy[i])),
                           (5 if i in (LWRI, RWRI) else 3) if state == ST_THIEF
                           else (4 if i in (LWRI, RWRI) else 2),
                           col, -1, cv2.LINE_AA)

    if state == ST_THIEF:
        ov = vis.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), col, -1)
        cv2.addWeighted(ov, 0.14, vis, 0.86, 0, vis)
        draw_label(vis, "ALERT", (x1, y2 + 26), col, 0.65, 2)


def annotate_crop(wide, state, conf, tid, detail) -> np.ndarray:
    """
    ★ برشِ «قابلِ قضاوت».

    هدف این تصویر نمایشِ زیبا نیست؛ این است که تو با یک نگاه بگویی
    «تصمیم درست بود یا نه». پس علاوه بر خودِ تصویر، *دلیلِ* تصمیم
    هم رویش نوشته می‌شود: مدل چه گفت، پوست چه گفت، کی‌پوینت چه گفت.
    """
    if wide is None or wide.size == 0:
        return np.zeros((10, 10, 3), np.uint8)
    img = wide.copy()
    h, w = img.shape[:2]
    scale = 280 / max(1, w)
    if scale != 1:
        img = cv2.resize(img, (280, max(40, int(h * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    h, w = img.shape[:2]

    panel = np.full((h + 78, w, 3), 28, np.uint8)
    panel[:h] = img
    col = COLORS[state]
    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), col, 3)
    cv2.rectangle(panel, (0, h), (w, h + 22), col, -1)
    cv2.putText(panel, f"#{tid} {LABELS[state]}", (5, h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)

    f = detail.get("fused", {})
    lines = [f"fused C{f.get(CLEAR,0):.2f} M{f.get(MEDICAL,0):.2f} F{f.get(FULL,0):.2f}"]
    sk = detail.get("skin")
    if sk:
        lines.append(f"skin  forehead={detail.get('fore',0):.2f} lower={detail.get('low',0):.2f}")
    else:
        lines.append("skin  (silent - grayscale/small)")
    lines.append(f"model p_covered={detail.get('p_covered',0):.2f}"
                 + (f"  kpt_w={detail['kpt_w']:.2f}" if "kpt_w" in detail else ""))
    for i, ln in enumerate(lines):
        cv2.putText(panel, ln[:46], (5, h + 38 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (210, 210, 210), 1, cv2.LINE_AA)
    return panel


def contact_sheet(panels, cols=4, pad=8) -> Optional[np.ndarray]:
    """صفحهٔ تماس: همهٔ سوژه‌ها کنار هم، برای قضاوتِ یکجا."""
    if not panels:
        return None
    ph = max(p.shape[0] for p in panels)
    pw = max(p.shape[1] for p in panels)
    rows = (len(panels) + cols - 1) // cols
    sheet = np.full((rows * (ph + pad) + pad, cols * (pw + pad) + pad, 3), 18, np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        y, x = pad + r * (ph + pad), pad + c * (pw + pad)
        sheet[y:y + p.shape[0], x:x + p.shape[1]] = p
    return sheet


# ═══════════════════════════════════════════════════════════════════════
#  ۱۰) حلقهٔ اصلی — یک ویدیو، از اول تا آخر
# ═══════════════════════════════════════════════════════════════════════

def run(cfg: Cfg, pose_model=None, mask_model=None, zeroshot=None, verbose=True):
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.half = cfg.half and device.startswith("cuda")
    if verbose:
        print(f"🔧 دستگاه: {device} | FP16: {cfg.half}")

    if pose_model is None:
        from ultralytics import YOLO
        pose_model = YOLO(cfg.pose_weights)
    if mask_model is None:
        mask_model = MaskModel(cfg, device)
    if zeroshot is None and cfg.use_zeroshot:
        zeroshot = ZeroShot(cfg, device)

    cap = cv2.VideoCapture(cfg.video)
    if not cap.isOpened():
        raise RuntimeError(f"ویدیو باز نشد: {cfg.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if verbose:
        print(f"📹 {W}×{H} @ {fps:.0f}fps | {total} فریم")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.out_video).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(cfg.out_video, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"نوشتنِ ویدیو ممکن نشد: {cfg.out_video}")

    dec = Decider(cfg)
    stat = {"frames": 0, "obs": 0, "back": 0, "small": 0, "alerts": 0,
            "reattached": 0, "skipped_cadence": 0}
    events: List[dict] = []
    last_pct = -1

    for r in pose_model.track(source=cfg.video, classes=[0], conf=cfg.person_conf,
                              imgsz=cfg.imgsz, tracker="bytetrack.yaml",
                              stream=True, persist=True, verbose=False):
        # ── فریمِ تمیز جدا از فریمِ نمایشی ──────────────────────────────
        # همهٔ برش‌ها از clean گرفته می‌شوند تا خط و متنِ روی تصویر واردِ
        # برشِ ذخیره‌شده نشود — اشکالی که نسخهٔ اول داشت و برش‌ها را
        # برای دیتاستِ آینده بی‌ارزش می‌کرد.
        clean = r.orig_img
        vis = clean.copy()
        t = stat["frames"] / fps
        stat["frames"] += 1
        if cfg.max_frames and stat["frames"] > cfg.max_frames:
            break

        if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
            ids = r.boxes.id.int().cpu().tolist()
            boxes = r.boxes.xyxy.cpu().numpy()
            kxy_all = r.keypoints.xy.cpu().numpy()
            kc_all = (r.keypoints.conf.cpu().numpy()
                      if r.keypoints.conf is not None
                      else np.ones(kxy_all.shape[:2], np.float32))

            batch_tight, batch_meta = [], []

            alive = set(ids)
            for tid, box, kxy, kc in zip(ids, boxes, kxy_all, kc_all):
                tr = dec.get(tid, t)

                # ★ امضای رنگیِ بدن — هم برای وصلِ مجدد، هم برای حافظه.
                #   ارزان است، پس برای همه و در هر فریم به‌روز می‌شود.
                bx1, by1, bx2, by2 = [int(v) for v in box]
                body = clean[max(0, by1):max(1, by2), max(0, bx1):max(1, bx2)]
                sig = body_signature(body)
                if sig is not None:
                    if tr.n == 0 and tr.inherited_from is None:
                        if dec.try_reattach(tr, sig, t):
                            stat["reattached"] += 1
                    tr.hist = sig if tr.hist is None else (
                        0.85 * tr.hist + 0.15 * sig)
                    nrm = float(np.linalg.norm(tr.hist))
                    if nrm > 1e-6:
                        tr.hist /= nrm

                # ── جهت: پشت به دوربین؟ ────────────────────────────────
                facing, ocf = orientation(kxy, kc, cfg.kpt_conf)
                is_back = (ocf >= 0.22) and (facing <= cfg.back_threshold)
                if is_back:
                    stat["back"] += 1
                    st, cf = dec.decide(tid, True, t)
                    draw_person(vis, box, head_box(kxy, kc, box, W, H, cfg)[0],
                                tid, st, cf, kxy, kc, cfg)
                    continue

                fs = frontal_score(kxy, kc, facing, ocf, cfg.kpt_conf)
                hbox, rel, src = head_box(kxy, kc, box, W, H, cfg)
                if min(hbox[2] - hbox[0], hbox[3] - hbox[1]) < cfg.min_face_px:
                    stat["small"] += 1
                    st, cf = dec.decide(tid, False, t)
                    draw_person(vis, box, hbox, tid, st, cf, kxy, kc, cfg)
                    continue

                if not dec.due(tid, t):
                    # ★ همین‌جا سرعت به دست می‌آید: کسی که حکمش قطعی
                    #   شده، این فریم اصلاً به طبقه‌بند نمی‌رود.
                    stat["skipped_cadence"] += 1
                    draw_person(vis, box, hbox, tid, tr.state, tr.conf, kxy, kc, cfg)
                    continue

                tight, wide = make_crops(clean, hbox, kxy, kc, cfg)
                if tight.size == 0 or min(tight.shape[:2]) < 16:
                    draw_person(vis, box, hbox, tid, tr.state, tr.conf, kxy, kc, cfg)
                    continue

                batch_tight.append(tight)
                batch_meta.append((tid, box, hbox, kxy, kc, wide, facing, rel, src, fs))

            # ── اجرای مدل‌ها به‌صورت دسته‌ای ────────────────────────────
            if batch_tight:
                pairs = mask_model(batch_tight)

                # صفر-شات فقط روی کروپ‌هایی که مدلِ اول «پوشیده» دیده.
                # در صحنهٔ عادی که بیشتر صورت‌ها بازند، هزینه‌اش ناچیز است.
                ratios: Dict[int, float] = {}
                if zeroshot is not None:
                    idx = [i for i, pr in enumerate(pairs) if pr[1] >= 0.25]
                    if idx:
                        rr = zeroshot([batch_tight[i] for i in idx])
                        ratios = dict(zip(idx, rr))

                for i, meta in enumerate(batch_meta):
                    tid, box, hbox, kxy, kc, wide, facing, rel, src, fs = meta
                    tight = batch_tight[i]
                    p_open, p_cov = pairs[i]

                    sk, sk_w, sk_d = skin_signal(tight)
                    kp, kp_w = kpt_signal(kc, facing)
                    fused, detail = combine(p_open, p_cov, ratios.get(i),
                                            kp, kp_w, sk, sk_w, cfg)
                    detail["fore"] = sk_d.get("forehead", 0.0)
                    detail["low"] = sk_d.get("lower", 0.0)
                    detail["roi"] = src

                    # وزنِ مشاهده: اندازه × وضوح × اعتبارِ جعبهٔ سر
                    # ★ کفِ کیفیت بالا برده شد. قبلاً حاصل‌ضربِ سه عاملِ
                    #   کوچک می‌توانست به ۰.۰۳ برسد و آن مشاهده عملاً
                    #   بی‌اثر می‌شد — یعنی برای یک حکم ده‌ها فریم لازم بود.
                    #   حالا یک مشاهدهٔ قابل‌قبول، وزنِ قابل‌قبول می‌گیرد و
                    #   حکم در ۲ تا ۴ فریم صادر می‌شود.
                    px = float(min(tight.shape[:2]))
                    q = (float(np.clip((px - cfg.min_face_px) / 70.0, 0.40, 1.0))
                         * float(np.clip(sharpness(tight) / 70.0, 0.55, 1.0))
                         * float(np.clip(rel, 0.55, 1.0)))
                    dec.observe(tid, fused, q, t)
                    stat["obs"] += 1

                    st, cf = dec.decide(tid, False, t, frontal=fs)
                    tr = dec.tracks[tid]

                    # ── بهترین شات: فقط وقتی کیفیت بهتر شد جایگزین کن ──
                    # نسخهٔ اول اولین برش را نگه می‌داشت که معمولاً
                    # لحظهٔ ورود و تارترین حالت است.
                    if q > tr.best_q:
                        tr.best_q, tr.best_wide, tr.best_tight = q, wide, tight
                        tr.best_detail = detail

                    if st == ST_THIEF and not tr.alerted:
                        tr.alerted = True
                        stat["alerts"] += 1
                        events.append({"tid": tid, "t": round(t, 2),
                                       "frame": stat["frames"], "conf": round(cf, 3),
                                       "detail": detail})

                    draw_person(vis, box, hbox, tid, st, cf, kxy, kc, cfg)

            dec.retire_lost(t, alive)

        # ── نوار وضعیت ──────────────────────────────────────────────────
        live = sum(1 for x in dec.tracks.values() if abs(x.last_t - t) < 0.5)
        bar = vis.copy()
        cv2.rectangle(bar, (0, 0), (W, 32), (18, 18, 18), -1)
        cv2.addWeighted(bar, 0.6, vis, 0.4, 0, vis)
        txt = "frame {}  people {}  alerts {}".format(stat["frames"], live, stat["alerts"])
        cv2.putText(vis, txt, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (240, 240, 240), 1, cv2.LINE_AA)
        writer.write(vis)

        if verbose and total:
            pct = int(stat["frames"] / total * 100)
            if pct >= last_pct + 10:
                last_pct = pct
                print("⏳ {:3d}%  |  مشاهده {}  |  هشدار {}".format(
                    pct, stat["obs"], stat["alerts"]))

    writer.release()

    # ── ذخیرهٔ برش‌ها و صفحهٔ تماس ──────────────────────────────────────
    panels = []
    if cfg.save_crops:
        order = sorted(dec.tracks.values(),
                       key=lambda x: (SEVERITY.get(x.state, 9), -x.best_q))
        for tr in order:
            if tr.best_wide is None:
                continue
            p = annotate_crop(tr.best_wide, tr.state, tr.conf, tr.tid, tr.best_detail)
            panels.append(p)
            cv2.imwrite(str(out_dir / "id{:03d}_{}_q{:.2f}.jpg".format(
                tr.tid, tr.state, tr.best_q)), p)
            if tr.best_tight is not None and tr.best_tight.size:
                cv2.imwrite(str(out_dir / "id{:03d}_{}_tight.jpg".format(
                    tr.tid, tr.state)), tr.best_tight)
        sheet = contact_sheet(panels)
        if sheet is not None:
            cv2.imwrite(str(out_dir / "_contact_sheet.jpg"), sheet)

    summary = {
        "frames": stat["frames"], "observations": stat["obs"],
        "skipped_back": stat["back"], "skipped_small": stat["small"],
        "alerts": stat["alerts"], "tracks": len(dec.tracks),
        "skipped_cadence": stat["skipped_cadence"],
        "states": {st: sum(1 for x in dec.tracks.values() if x.state == st)
                   for st in (ST_ANALYZING, ST_CLEAR, ST_SUSPECT, ST_THIEF)},
        "reattached": dec.reattached,
        "events": events,
        "video": cfg.out_video, "crops": str(out_dir),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if verbose:
        print("\n" + "=" * 62)
        print("  فریم {} | مشاهده {} | ردیابی {} | هشدار {}".format(
            stat["frames"], stat["obs"], len(dec.tracks), stat["alerts"]))
        print("  رد‌شده: پشت‌به‌دوربین {} | خیلی کوچک {} | آهنگِ بازبینی {}".format(
            stat["back"], stat["small"], stat["skipped_cadence"]))
        saved = stat["skipped_cadence"]
        total_cand = saved + stat["obs"]
        if total_cand:
            print("  ★ صرفه‌جویی: {:.0%} از فراخوانی‌های طبقه‌بند حذف شد".format(
                saved / total_cand))
        print("  وصلِ مجددِ هویت پس از گم‌شدن: {} بار".format(stat["reattached"]))
        print("  وضعیتِ نهایی: {}".format(summary["states"]))
        print("  ویدیو: {}".format(cfg.out_video))
        print("  برش‌ها: {}   ← _contact_sheet.jpg را ببین".format(out_dir))
        print("=" * 62)
    return summary, dec, panels
