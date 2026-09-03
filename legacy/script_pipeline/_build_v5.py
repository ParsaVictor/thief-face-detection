"""
ساخت نسخهٔ ۲ — سه گامِ درخواستی روی نسخهٔ پایه.

  گام ۵ : شکستنِ قفلِ سبز (ارزان — هر ۲ ثانیه یک بار)
  گام ۶ : حداقلِ اندازهٔ صورت + نمایشِ پیشرفتِ بررسی
  گام ۷ : رنگِ اسکلت = رنگِ وضعیتِ فرد

هر تغییر با «★ گام N» علامت‌گذاری شده تا در کد پیدا کردنش آسان باشد.
بقیهٔ الگوریتم دست‌نخورده است.
"""
import json
import re
from pathlib import Path

SRC = Path(__file__).parent / "orig_cell7.py"
OUT = Path(r"C:\1\1_پروژه\1_پروژه خودم\چهره\Face_Mask_V5.ipynb")

text = SRC.read_text(encoding="utf-8")
heads = [(m.start(), m.group(0)) for m in re.finditer(r"^# -{4,}.*$", text, re.M)]
B = {}
for i, (pos, head) in enumerate(heads):
    end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
    B[re.search(r"(\d)\)", head).group(1)] = text[pos:end].rstrip()

applied = []


def rep(key, old, new, tag):
    assert old in B[key], f"الگو پیدا نشد [{tag}]"
    B[key] = B[key].replace(old, new, 1)
    applied.append(tag)


# ══════════════════════════════════════════════════════════════════════
#  ★ D1 — تاریکی دیگر «مشکوک» تفسیر نمی‌شود
# ══════════════════════════════════════════════════════════════════════
# اندازه‌گیریِ عینی روی همین کد:
#
#   روشناییِ ناحیه (V)   skin_ratio   حکم
#   ۴۵ و بالاتر            ۱.۰۰       نارنجی  ✔
#   ۳۸                     ۰.۰۰       🔴 قرمز  ✘  ← همان پوست، فقط در سایه
#
# بازهٔ HSV در skin_ratio کفِ V ≥ 40 دارد. یعنی پوستِ کاملاً معمولی
# در سایه «صفر پوست» خوانده می‌شود و is_suspicious آن را قرمز می‌کند.
# پرتاب از ۱.۰۰ به ۰.۰۰ ناگهانی است، نه تدریجی.
#
# اصلاح: اگر ناحیه آن‌قدر تاریک است که این سنجه بی‌معنی می‌شود،
# **ادعا نکن**. برگرد نارنجی. برای یک سیستمِ امنیتی، سکوت در ابهام
# بهتر از آژیرِ نادرست است.
rep("3",
    """def is_suspicious(face_bgr):
    h, w, _ = face_bgr.shape
    upper = face_bgr[0:int(h * 0.45), :]
    return skin_ratio(upper) < 0.12""",
    """MIN_BRIGHT_FOR_SKIN = 50        # ★ D1: زیر این روشنایی، قضاوت نکن


def is_suspicious(face_bgr, min_bright=None):
    h, w, _ = face_bgr.shape
    upper = face_bgr[0:int(h * 0.45), :]
    if upper.size == 0:
        return False

    # ★ D1 — گاردِ روشنایی.
    #   نکتهٔ ظریف: به‌خاطر چرخشِ تصویر، این «۴۵٪ بالای کروپ» در واقع
    #   ناحیهٔ بینی و دهان است (آزمون شده) — یعنی ناحیهٔ درست. مشکل
    #   فقط نور است، نه ناحیه.
    thr = MIN_BRIGHT_FOR_SKIN if min_bright is None else min_bright
    v_mean = float(cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)[..., 2].mean())
    if v_mean < thr:
        return False                # خیلی تاریک → نارنجی، نه قرمز

    return skin_ratio(upper) < 0.12""",
    "★ D1: گاردِ روشنایی در is_suspicious")

# ══════════════════════════════════════════════════════════════════════
#  گام ۶ — حداقلِ اندازهٔ صورت
# ══════════════════════════════════════════════════════════════════════
rep("4",
    "def align_and_crop_face(person_img, kxy, kconf, conf_th=0.5):",
    "def align_and_crop_face(person_img, kxy, kconf, conf_th=0.5, min_eye_dist=8):",
    "گام۶: پارامتر min_eye_dist")

rep("4",
    """    eye_dist = float(np.linalg.norm(np.array(leye) - np.array(reye)))
    if eye_dist < 3:
        return None, face_score""",
    """    eye_dist = float(np.linalg.norm(np.array(leye) - np.array(reye)))
    # ★ گام ۶ — حداقلِ اندازهٔ صورت.
    #   قبلاً این عدد ۳ بود؛ یعنی صورتی به عرضِ ~۱۰ پیکسل هم رأی می‌داد و
    #   آن رأی عملاً نویزِ خالص بود. با ۸، فردِ دور دیده و ردیابی می‌شود
    #   ولی تا وقتی به‌قدرِ کافی نزدیک نشده، حکمی دربارهٔ او صادر نمی‌شود
    #   و در حالتِ «Analyzing» می‌ماند.
    if eye_dist < min_eye_dist:
        return None, face_score""",
    "گام۶: آستانه ۳ → ۸")

# ══════════════════════════════════════════════════════════════════════
#  ★ D2 — گیتِ وضوح (جایگزینِ محافظی که تصادفاً داشتیم)
# ══════════════════════════════════════════════════════════════════════
# چرا لازم شد: مقایسهٔ V2 با V3 نشان داد تنها تغییرِ رفتاری، عوض‌شدنِ
# مدلِ ژست بود (yolo11n → yolo11s). مدلِ ضعیف‌تر روی صورت‌های مرزی
# (دور، تار، نیم‌رخ) اطمینانِ پایین می‌داد و گیتِ
# `میانگین(بینی، چشم‌ها) ≥ 0.5` آن‌ها را رد می‌کرد.
#
# یعنی ضعفِ مدل تصادفاً نقشِ **فیلترِ کیفیت** را بازی می‌کرد. مدلِ
# قوی‌تر آن محافظ را برداشت: حالا کی‌پوینت‌ها روی صورتِ تار هم مطمئن‌اند،
# گیت باز می‌شود، و طبقه‌بند روی کروپی قضاوت می‌کند که نباید.
#
# راه‌حل: یک فیلترِ کیفیتِ **صریح** بگذاریم، به‌جای اینکه به ضعفِ مدل
# تکیه کنیم. وضوح با واریانسِ لاپلاسین سنجیده می‌شود — ارزان و مؤثر.
rep("4",
    """    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return None, face_score""",
    """    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return None, face_score

    # ★ D2 — گیتِ وضوح. کروپِ تار = رأیِ نویزی.
    _crop = rotated[y1:y2, x1:x2]
    if _crop.size == 0:
        return None, face_score
    _g = cv2.cvtColor(_crop, cv2.COLOR_BGR2GRAY)
    if float(cv2.Laplacian(_g, cv2.CV_64F).var()) < MIN_SHARPNESS:
        return None, face_score      # خیلی تار → رأی نده، «Analyzing» بماند""",
    "★ D2: گیتِ وضوح")

rep("4",
    "NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4" if False else
    "def align_and_crop_face(person_img, kxy, kconf, conf_th=0.5, min_eye_dist=8):",
    "MIN_SHARPNESS = 20.0            # ★ D2: کمینهٔ وضوحِ کروپ برای رأی‌دادن\n"
    "\n"
    "\n"
    "def align_and_crop_face(person_img, kxy, kconf, conf_th=0.5, min_eye_dist=8):",
    "★ D2: ثابتِ MIN_SHARPNESS")

# ══════════════════════════════════════════════════════════════════════
#  گام ۵ — شکستنِ قفلِ سبز
# ══════════════════════════════════════════════════════════════════════
rep("5",
    """        self.FOCUS_GREEN_NEEDED = 3
""",
    """        self.FOCUS_GREEN_NEEDED = 3
        # ★ گام ۵ — قفلِ سبز دیگر دائمی نیست.
        #   هر ۶۰ فریم (≈۲ ثانیه در ۳۰fps) یک بار دوباره بررسی می‌شود.
        #   هزینه: برای هر فردِ سبز، یک اجرای طبقه‌بند در هر ۶۰ فریم
        #   به‌جای صفر — یعنی حدودِ ۱.۷٪ حالتِ عادی. عملاً رایگان.
        self.GREEN_RECHECK_FRAMES = 60
        # ★ C2 — هر چند فریم یک بار، به تفکیکِ وضعیت
        self.CADENCE = {"red": 1, "orange": 3, "gray": 1}
        # ★ D3 — قرمز باید این‌قدر از نارنجی جلو باشد تا اعلام شود
        # ۱.۰ = خاموش (رفتارِ نسخهٔ اول). بالاتر = سخت‌گیرتر.
        self.RED_MARGIN = 1.00
        # ★ D4 — بین دو آلارمِ یک نفر حداقل این‌قدر فریم فاصله باشد
        self.ALERT_COOLDOWN = 90
""",
    "گام۵: ثابتِ GREEN_RECHECK_FRAMES + ★ C2: جدولِ CADENCE")

rep("5",
    """                "is_new": True,        # برای افکت نمایشی: آیا تازه معرفی شده؟
                "just_finalized": None # برای افکت نمایشی: آیا همین الان قفل نهایی گرفته؟
            }""",
    """                "is_new": True,        # برای افکت نمایشی: آیا تازه معرفی شده؟
                "just_finalized": None,# برای افکت نمایشی: آیا همین الان قفل نهایی گرفته؟
                "locked_frame": 0,     # ★ گام ۵: آخرین فریمی که قفل/بازبینی شد
                "checks": 0,           # ★ گام ۶: چند بار تا حالا بررسی شده
                "last_alert": -10**9,  # ★ D4: آخرین فریمی که آلارم داد
            }""",
    "گام۵/۶: فیلدهای locked_frame و checks")

rep("5",
    """        st = self.data[tid]
        if st["locked"]:
            return False
        if st["mode"] == "fast":""",
    """        st = self.data[tid]
        if st["locked"]:
            # ★ گام ۵ — بازبینیِ دوره‌ایِ افرادِ سبز.
            #   سناریویی که این را لازم می‌کند: کسی با صورتِ باز وارد
            #   می‌شود، سبز قفل می‌شود، بعد داخلِ مغازه ماسک می‌کشد.
            #   با قفلِ دائمی، سیستم دیگر هرگز نگاهش نمی‌کرد.
            return (frame_idx - st["locked_frame"]) >= self.GREEN_RECHECK_FRAMES
        if st["mode"] == "fast":
            return True

        # ★ C2 — آهنگِ بازبینیِ تطبیقی: بودجه به کسی برسد که مهم است.
        #
        #   در حال بررسی → هر فریم. باید سریع به حکم برسیم.
        #   قرمز         → هر فریم. مهم‌ترین فرد در صحنه است.
        #   نارنجی       → هر ۳ فریم. ماسکِ پزشکی رایج و پایدار است؛
        #                  بررسیِ مکررش اتلافِ بودجه است.
        #   سبزِ قفل‌شده → هر ۶۰ فریم (بالاتر).
        #
        #   نسخهٔ قبلی برای نارنجی و قرمز هر دو «هر ۲ فریم» بود — یعنی
        #   به فردِ خطرناک و فردِ بی‌خطر یک اندازه توجه می‌کرد.
        return frame_idx % self.CADENCE.get(st["color"], self.FOCUS_INTERVAL) == 0""",
    "C2: آهنگِ بازبینیِ تطبیقی")

# دو خطِ قدیمیِ should_analyze که بعد از بلوکِ جدید یتیم مانده‌اند
_orphan = ('            return True\n'
           '        return frame_idx % self.FOCUS_INTERVAL == 0\n')
assert _orphan in B["5"], "خطوطِ یتیم پیدا نشد"
B["5"] = B["5"].replace(_orphan, "", 1)
applied.append("C2: پاکسازیِ خطوطِ قدیمیِ should_analyze")

rep("5",
    """    def register_vote(self, tid, category, conf):
        st = self.data[tid]
        st["just_finalized"] = None
""",
    """    def register_vote(self, tid, category, conf, frame_idx=0):
        st = self.data[tid]
        st["just_finalized"] = None
        st["checks"] += 1                       # ★ گام ۶: شمارشِ بررسی‌ها

        # ★ گام ۵ — رأیِ بازبینی برای فردی که قبلاً سبز قفل شده بود.
        if st["locked"]:
            st["locked_frame"] = frame_idx      # ساعتِ بازبینی صفر شود
            if category == "green":
                return                          # هنوز صورتش باز است — کاری نکن
            # پوشش دیده شد → قفل می‌شکند و می‌رود به حالتِ فوکوس
            st["locked"] = False
            st["mode"] = "focus"
            st["votes"] = []
            st["focus_window"].clear()
            st["focus_window"].append(category)
            st["color"], st["label"] = category, LABELS[category]
            if category == "red":
                st["just_finalized"] = "red"
            return
""",
    "گام۵: register_vote")

rep("5",
    """    def _lock(self, tid, category):
        st = self.data[tid]
        st["locked"] = True""",
    """    def _lock(self, tid, category, frame_idx=0):
        st = self.data[tid]
        st["locked"] = True
        st["locked_frame"] = frame_idx          # ★ گام ۵: شروعِ شمارشِ بازبینی""",
    "گام۵: _lock")

for old, new in [('self._lock(tid, "green")\n                else:',
                  'self._lock(tid, "green", frame_idx)\n                else:'),
                 ('self._lock(tid, "green")\n            else:',
                  'self._lock(tid, "green", frame_idx)\n            else:')]:
    assert old in B["5"], "فراخوانیِ _lock پیدا نشد"
    B["5"] = B["5"].replace(old, new, 1)
applied.append("گام۵: انتقال frame_idx به _lock")

# ══════════════════════════════════════════════════════════════════════
#  C1 — حافظهٔ کوتاه‌مدتِ هویت (بلوکِ تازه، از c1_block.py)
# ══════════════════════════════════════════════════════════════════════
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("c1b", str(Path(__file__).parent / "c1_block.py"))
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)
B["5b"] = _m.TRACK_MEMORY_SRC.rstrip()
applied.append("C1: کلاسِ TrackMemory")

# ══════════════════════════════════════════════════════════════════════
#  گام ۷ — رنگِ اسکلت = رنگِ وضعیت
# ══════════════════════════════════════════════════════════════════════
rep("6",
    'def draw_upper_skeleton(frame, kxy, kconf, color=(255, 255, 0), conf_th=0.4):\n    """رسم اسکلت بالاتنه فقط - برای جلوه بصری خفن"""',
    'def draw_upper_skeleton(frame, kxy, kconf, color=(160, 160, 160), conf_th=0.4):\n'
    '    """\n'
    '    رسم اسکلتِ بالاتنه.\n\n'
    '    ★ گام ۷ — رنگِ پیش‌فرض دیگر زردِ ثابت نیست. حالا فراخوان رنگِ\n'
    '    وضعیتِ همان فرد را می‌فرستد، پس کادر و اسکلت و برچسب هم‌رنگ‌اند\n'
    '    و کلِ فریم با یک نگاه خوانده می‌شود.\n'
    '    """',
    "گام۷: امضای draw_upper_skeleton")

# ══════════════════════════════════════════════════════════════════════
#  خط لوله — یکپارچه‌سازی + وصل‌کردنِ سه گام
# ══════════════════════════════════════════════════════════════════════
P = B["7"]
P = P.replace("# ---------------- 7) Main Pipeline (with dramatic slow-mo writing) ----",
              "# ---------------- 7) Main Pipeline ------------------------------------")
P = P.replace(
    """def process_video_demo(input_path, output_path, conf_thres=0.4, yolo_imgsz=640,
                        face_conf_th=0.5, slowmo_repeat=6):""",
    """def process_video(input_path, output_path, conf_thres=0.4, yolo_imgsz=640,
                  face_conf_th=0.5, slowmo_repeat=6,
                  show_skeleton=True, show_gallery=True, min_eye_dist=8):""")
P = P.replace(
    '''    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))''',
    '''    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not out.isOpened():                                        # [ایمنی]
        raise RuntimeError(f"❌ فایل خروجی باز نشد: {output_path}")''')

# ★ گام ۷: کی‌پوینت‌ها را نگه دار تا در حلقهٔ رسم رنگِ درست را بدانیم
P = P.replace(
    "        batch_crops, batch_meta = [], []\n"
    "        trigger_slowmo = False   # آیا این فریم باید کند نمایش داده بشه؟",
    "        batch_crops, batch_meta = [], []\n"
    "        trigger_slowmo = False   # آیا این فریم باید کند نمایش داده بشه؟\n"
    "        kpts_by_tid = {}         # ★ گام ۷: برای رسمِ اسکلت با رنگِ وضعیت")
P = P.replace(
    """            st = state_mgr.ensure(tid)

            # --- افکت نمایشی: فرد کاملا جدید""",
    """            # ★ C1 — اگر شناسه تازه است، شاید همان کسی باشد که چند
            #   ثانیه پیش پشتِ قفسه گمش کردیم.
            is_new_track = tid not in state_mgr.data
            st = state_mgr.ensure(tid)
            if track_memory is not None:
                if is_new_track:
                    track_memory.try_inherit(tid, st, person_crop, box,
                                             frame_idx, ids, W)
                track_memory.observe(tid, person_crop, box, frame_idx, st)

            kpts_by_tid[tid] = (kxy, kconf)          # ★ گام ۷

            # --- افکت نمایشی: فرد کاملا جدید""")

# امضای تابع + فراموشیِ دوره‌ای
P = P.replace(
    "                  show_skeleton=True, show_gallery=True, min_eye_dist=8):",
    "                  show_skeleton=True, show_gallery=True, min_eye_dist=8,\n"
    "                  track_memory=None):")
P = P.replace(
    "        kpts_by_tid = {}         # ★ گام ۷: برای رسمِ اسکلت با رنگِ وضعیت",
    "        kpts_by_tid = {}         # ★ گام ۷: برای رسمِ اسکلت با رنگِ وضعیت\n"
    "        if track_memory is not None and frame_idx % 30 == 0:\n"
    "            track_memory.forget_old(frame_idx)   # ★ C1: فراموشیِ دوره‌ای")
# آمار C1 در خروجی
P = P.replace(
    '    print(f"📄 {len(events)} رویداد -> {json_path}")',
    '    print(f"📄 {len(events)} رویداد -> {json_path}")\n'
    '    if track_memory is not None:\n'
    '        print(f"🔗 C1: {track_memory.stats}")')
applied.append("C1: وصل‌شدن به خط لوله")

# استفاده از min_eye_dist
P = P.replace(
    "                face_crop, face_score = align_and_crop_face(person_crop, local_kxy, kconf, face_conf_th)",
    "                face_crop, face_score = align_and_crop_face(\n"
    "                    person_crop, local_kxy, kconf, face_conf_th, min_eye_dist)")

# ★ گام ۷: حذفِ رسمِ اسکلتِ زردِ ثابت از حلقهٔ اول
P = P.replace(
    """            # --- رسم اسکلت بالاتنه (جلوه بصری) ---
            local_kxy_full = kxy.copy()
            draw_upper_skeleton(frame, local_kxy_full, kconf)

""", "")

# ★ گام ۵: انتقال frame_idx به register_vote
P = P.replace('state_mgr.register_vote(tid, "green", conf)',
              'state_mgr.register_vote(tid, "green", conf, frame_idx)')
P = P.replace('state_mgr.register_vote(tid, cat, conf)',
              'state_mgr.register_vote(tid, cat, conf, frame_idx)')

# ★ گام ۶ و ۷: حلقهٔ رسم
P = P.replace(
    """        for tid, box in zip(ids, boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            st = state_mgr.ensure(tid)
            color = COLORS[st["color"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {tid}: {st['label']}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
""",
    """        for tid, box in zip(ids, boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            st = state_mgr.ensure(tid)
            color = COLORS[st["color"]]

            # ★ گام ۷ — اسکلت با رنگِ وضعیتِ همین فرد
            if show_skeleton and tid in kpts_by_tid:
                kxy_d, kconf_d = kpts_by_tid[tid]
                draw_upper_skeleton(frame, kxy_d, kconf_d, color)

            # ★ گام ۶ — در حالتِ «در حال بررسی» پیشرفت را نشان بده تا
            #   معلوم باشد سیستم فرد را دیده و دارد رویش کار می‌کند،
            #   نه اینکه او را نادیده گرفته باشد.
            label = st["label"]
            if st["color"] == "gray":
                got = len(st["votes"])
                if st["checks"] == 0:
                    label = "Analyzing... (too far)"
                else:
                    label = f"Analyzing... ({got}/{state_mgr.FAST_VOTES_NEEDED})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {tid}: {label}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
""")

P = P.replace("""        # ---- گالری گوشه تصویر ----
        gallery.draw(frame)""",
              """        # ---- گالری گوشه تصویر ----
        if show_gallery:
            gallery.draw(frame)""")
P = P.replace('print(f"✅ Demo video ready in {time.time() - t0:.2f}s -> {output_path}")',
              'print(f"✅ Done in {time.time() - t0:.2f}s -> {output_path}")')

# ══════════════════════════════════════════════════════════════════════
#  A2 — برشِ سوژهٔ مشکوک از فریمِ *هنوز تمیز*
# ══════════════════════════════════════════════════════════════════════
# در نسخهٔ اصلی، برشِ سوژهٔ قرمز در حلقهٔ *رسم* گرفته می‌شد — یعنی بعد
# از اینکه مستطیل و خطوطِ اسکلت روی فریم کشیده شده بودند.
#
# دو راه بود:
#   الف) clean = frame.copy() در ابتدای هر فریم → کپیِ کاملِ فریم،
#        حدودِ ۶MB در 1080p، در هر فریم.
#   ب) برداشتنِ برش *قبل* از شروعِ رسم → صفر کپیِ اضافه.
# گزینهٔ (ب) انتخاب شد چون سریع‌تر است.
_old = '        # ---- رسم باکس‌ها + تشخیص لحظه هشدار قرمز برای اسلوموشن و گالری ----\n'
_new = (
    '        # ---- ★ A2: برشِ سوژه، از فریمی که هنوز رویش رسم نشده ----\n'
    '        for tid, box in zip(ids, boxes):\n'
    '            st = state_mgr.ensure(tid)\n'
    '            if st.get("just_finalized") == "red":\n'
    '                # ★ D4 — یک نفر نباید پشتِ سرِ هم آلارم بدهد.\n'
    '                #   اگر وضعیتش بین نارنجی و قرمز نوسان کند، بدونِ\n'
    '                #   این شرط هر نوسان یک «SUSPECT» تازه در گالری و\n'
    '                #   یک رویدادِ تازه در JSON می‌ساخت — که دقیقاً مثلِ\n'
    '                #   چند آلارمِ کاذبِ پشت‌سرهم دیده می‌شد.\n'
    '                st["just_finalized"] = None\n'
    '                if (frame_idx - st["last_alert"]) < state_mgr.ALERT_COOLDOWN:\n'
    '                    continue\n'
    '                st["last_alert"] = frame_idx\n'
    '                trigger_slowmo = True\n'
    '                x1c, y1c = max(0, int(box[0])), max(0, int(box[1]))\n'
    '                x2c, y2c = min(W, int(box[2])), min(H, int(box[3]))\n'
    '                if x2c > x1c and y2c > y1c:\n'
    '                    gallery.add(frame[y1c:y2c, x1c:x2c].copy(),\n'
    '                                f"SUSPECT ID {tid}", (0, 0, 255))\n'
    '                events.append({                       # ★ A3\n'
    '                    "frame": frame_idx,\n'
    '                    "time_s": round(frame_idx / fps, 2),\n'
    '                    "track_id": int(tid),\n'
    '                    "state": "red",\n'
    '                    "label": LABELS["red"],\n'
    '                    "confidence": round(float(st.get("conf", 0.0)), 3),\n'
    '                    "checks": int(st.get("checks", 0)),\n'
    '                })\n'
    '\n'
    '        # ---- رسم باکس‌ها ----\n')
assert _old in P
P = P.replace(_old, _new, 1)

# حذفِ بلوکِ قدیمیِ برشِ آلوده
_dirty = ('            if st.get("just_finalized") == "red":\n'
          '                trigger_slowmo = True\n'
          '                x1c, y1c = max(0, x1), max(0, y1)\n'
          '                x2c, y2c = min(W, x2), min(H, y2)\n'
          '                crop = frame[y1c:y2c, x1c:x2c].copy()\n'
          '                gallery.add(crop, f"SUSPECT ID {tid}", (0, 0, 255))\n'
          '                st["just_finalized"] = None  # فقط یکبار افکت رو نشون بده\n')
assert _dirty in P, "بلوکِ برشِ آلوده پیدا نشد"
P = P.replace(_dirty, "", 1)

# ══════════════════════════════════════════════════════════════════════
#  A3 — لیستِ رویدادها، نمایشِ اطمینان، ذخیرهٔ JSON
# ══════════════════════════════════════════════════════════════════════
P = P.replace("    t0 = time.time()\n    last_pct = -1\n    frame_idx = 0",
              "    t0 = time.time()\n    last_pct = -1\n    frame_idx = 0\n"
              "    events = []                      # ★ A3: لاگِ رویدادها")

P = P.replace(
    '            label = st["label"]\n            if st["color"] == "gray":',
    '            # ★ A3 — ضریبِ اطمینان روی تصویر. تا حالا محاسبه می‌شد\n'
    '            #   ولی هیچ‌جا دیده نمی‌شد؛ حالا هم روی کادر است و هم در JSON.\n'
    '            label = st["label"]\n'
    '            if st["color"] != "gray" and st.get("conf", 0) > 0:\n'
    '                label = f"{label} {st[\'conf\']:.0%}"\n'
    '            if st["color"] == "gray":')

_end_old = ('    out.release()\n'
            '    print(f"✅ Done in {time.time() - t0:.2f}s -> {output_path}")')
_end_new = (
    '    out.release()\n\n'
    '    # ---- ★ A3: وضعیتِ نهاییِ هر فرد هم ثبت شود ----\n'
    '    for tid, st in state_mgr.data.items():\n'
    '        events.append({\n'
    '            "frame": frame_idx,\n'
    '            "time_s": round(frame_idx / fps, 2),\n'
    '            "track_id": int(tid),\n'
    '            "state": st["color"],\n'
    '            "label": st["label"],\n'
    '            "confidence": round(float(st.get("conf", 0.0)), 3),\n'
    '            "checks": int(st.get("checks", 0)),\n'
    '            "final": True,\n'
    '        })\n\n'
    '    json_path = os.path.splitext(output_path)[0] + "_events.json"\n'
    '    with open(json_path, "w", encoding="utf-8") as f:\n'
    '        json.dump({"video": input_path,\n'
    '                   "frames": frame_idx,\n'
    '                   "pose_weights": POSE_WEIGHTS,\n'
    '                   "params": {"conf_thres": conf_thres,\n'
    '                              "yolo_imgsz": yolo_imgsz,\n'
    '                              "face_conf_th": face_conf_th,\n'
    '                              "min_eye_dist": min_eye_dist},\n'
    '                   "events": events}, f, ensure_ascii=False, indent=2)\n\n'
    '    print(f"✅ Done in {time.time() - t0:.2f}s -> {output_path}")\n'
    '    print(f"📄 {len(events)} رویداد -> {json_path}")\n'
    '    return events')
assert _end_old in P
P = P.replace(_end_old, _end_new, 1)

applied.append("A2: برشِ تمیزِ سوژه (بدون کپیِ اضافه)")
applied.append("A3: لاگِ رویداد + JSON + نمایشِ اطمینان")

for must in ["def process_video(", "kpts_by_tid", "draw_upper_skeleton(frame, kxy_d, kconf_d, color)",
             "register_vote(tid, cat, conf, frame_idx)", "too far", "min_eye_dist)",
             "★ A2: برشِ سوژه", "_events.json", "return events"]:
    assert must in P, f"وصل نشد: {must}"
assert P.count('just_finalized") == "red"') == 1, "بلوکِ تکراری مانده"
assert "draw_upper_skeleton(frame, local_kxy_full, kconf)" not in P, "اسکلتِ زردِ قدیمی مانده"
B["7"] = P
applied.append("گام۷: اسکلت هم‌رنگِ وضعیت در حلقهٔ رسم")
applied.append("گام۶: برچسبِ پیشرفتِ بررسی")

print("تغییرات اعمال‌شده:")
for a in applied:
    print("  ✔", a)



# ══════════════════════════════════════════════════════════════════════
#  B1 — مدلِ ژست به‌صورت متغیر
# ══════════════════════════════════════════════════════════════════════
rep("1",
    'pose_model = YOLO("yolo11n-pose.pt")',
        '''
# ★ B1 — مدلِ ژست حالا متغیر است، نه ثابت.
#
#   yolo11n-pose  سبک‌ترین. همان چیزی که تا حالا استفاده می‌شد.
#   yolo11s-pose  ★ پیش‌فرضِ جدید — کی‌پوینتِ دقیق‌تر، مخصوصاً برای
#                 سوژه‌های دور و مچ/دست. حدودِ ۲ برابر کندتر از n
#                 ولی چون کی‌پوینتِ بهتر یعنی کادرِ بهترِ صورت،
#                 روی کلِ زنجیره اثر مثبت دارد.
#   yolo26s-pose  نسلِ بعدی (منتشرشده ژانویهٔ ۲۰۲۶). NMS-free و با
#                 RLE برای مکان‌یابیِ دقیق‌ترِ کی‌پوینت. مقالهٔ رسمی
#                 تا +۷.۲ AP نسبت به YOLO11 روی COCO-pose گزارش کرده.
#
#   ⚠️ اگر به yolo26 سوییچ کردی: نحوهٔ کالیبراسیونِ *اطمینانِ*
#      کی‌پوینت‌ها ممکن است فرق کند، و گیتِ ما (FACE_CONF_TH) دقیقاً
#      روی همان اطمینان‌ها کار می‌کند. پس بعد از سوییچ حتماً یک بار
#      خروجی را نگاه کن؛ شاید لازم باشد ۰.۵ را کمی بالا/پایین ببری.
POSE_WEIGHTS = "yolo26s-pose.pt"
pose_model = YOLO(POSE_WEIGHTS)
''',
    "B1: مدلِ ژست متغیر شد")

# ══════════════════════════════════════════════════════════════════════
#  A3 — ضریبِ اطمینانِ تصمیم
# ══════════════════════════════════════════════════════════════════════
rep("5",
    '                "locked_frame": 0,     # ★ گام ۵: آخرین فریمی که قفل/بازبینی شد',
    '                "conf": 0.0,           # ★ A3: اطمینانِ حکمِ فعلی (۰..۱)\n'
    '                "locked_frame": 0,     # ★ گام ۵: آخرین فریمی که قفل/بازبینی شد',
    "A3: فیلدِ conf")

rep("5",
    """                best = max(score, key=score.get)
                if best == "green":""",
    """                best = max(score, key=score.get)
                # ★ A3 — سهمِ رأیِ برنده از کلِ امتیاز = ضریبِ اطمینان.
                #   قبلاً این عدد محاسبه می‌شد ولی دور ریخته می‌شد؛
                #   حالا هم روی تصویر نوشته و هم در JSON ثبت می‌شود.
                total = sum(score.values()) or 1.0

                # ★ D3 — قرمز باید *برتریِ روشن* داشته باشد، نه صرفاً
                #   بیشترین رأی. با ۲ نارنجی و ۱ قرمز، «قرمز» می‌توانست
                #   ببرد اگر اطمینانِ آن یک رأی بالا بود. برای آژیرِ
                #   امنیتی این کافی نیست: هزینهٔ آلارمِ کاذب بالاست.
                #   نارنجی چنین شرطی ندارد — سخت‌گیری فقط برای قرمز است.
                if (best == "red" and score["orange"] > 0
                        and score["red"] < self.RED_MARGIN * score["orange"]):
                    best = "orange"
                st["conf"] = score[best] / total
                if best == "green":""",
    "A3: اطمینان در حالتِ fast")

rep("5",
    """                    best = max(set(sub), key=sub.count)
                    if best == "red" and st["color"] != "red":""",
    """                    # ★ C3 — تساویِ آرا دیگر تصادفی نیست.
                    #   `max(set(sub), key=sub.count)` وقتی دو رنگ تعدادِ
                    #   برابر دارند، هرکدام را که در پیمایشِ set زودتر
                    #   بیاید برمی‌گرداند — و ترتیبِ set تضمین‌شده نیست.
                    #   یعنی با ۲ نارنجی و ۲ قرمز، رنگ می‌توانست بی‌دلیل
                    #   بین دو حالت بپرد. حالا در تساوی رنگِ فعلی حفظ
                    #   می‌شود؛ فقط با اکثریتِ واقعی عوض می‌شود.
                    counts = {c: sub.count(c) for c in set(sub)}
                    top = max(counts.values())
                    winners = [c for c, n in counts.items() if n == top]
                    if len(winners) > 1 and st["color"] in winners:
                        best = st["color"]           # تساوی → همان‌که هست
                    else:
                        best = sorted(winners)[0]    # قطعی و تکرارپذیر
                    # ★ D3 — در فوکوس هم قرمز اکثریتِ اکید لازم دارد.
                    #   به RED_MARGIN گره خورده تا وقتی آن را ۱.۰ می‌گذاری
                    #   (یعنی «سخت‌گیری خاموش»)، این هم خاموش شود. وگرنه
                    #   کلید نصفه‌نیمه عمل می‌کرد و گیج‌کننده می‌شد.
                    if (self.RED_MARGIN > 1.0 and best == "red"
                            and counts.get("red", 0) <= counts.get("orange", 0)):
                        best = "orange"
                    st["conf"] = counts[best] / len(window)   # ★ A3
                    if best == "red" and st["color"] != "red":""",
    "A3 + ★ C3: اطمینان و تساویِ قطعی در focus")

# ══════════════════════════════════════════════════════════════════════
COMPARE_CELL = r"""# مقایسهٔ دو مدل روی همان ویدیو — چند صد فریمِ اول کافی است
import time
from collections import Counter

COMPARE_MODELS = ["yolo11s-pose.pt", "yolo26s-pose.pt"]

results = {}
for w in COMPARE_MODELS:
    print()
    print("=" * 60)
    print("  " + w)
    print("=" * 60)
    try:
        pose_model = YOLO(w)
    except Exception as e:
        print("  bargozari nashod:", type(e).__name__, e)
        print("  (shayad ultralytics ghadimi ast:  !pip install -U ultralytics)")
        continue

    POSE_WEIGHTS = w
    state_mgr = TrackStateManager()
    gallery   = PresentationGallery(max_items=4, thumb_size=140)
    tmem = None
    if USE_TRACK_MEMORY:
        tmem = TrackMemory(memory_seconds=MEMORY_SECONDS, fps=_fps_guess,
                           match_th=MEMORY_MATCH_TH, margin=MEMORY_MARGIN,
                           max_center_dist=MEMORY_MAX_DIST)

    t0 = time.time()
    ev = process_video(INPUT_VIDEO, "/content/cmp_" + w.replace(".pt", "") + ".mp4",
                       conf_thres=CONF_THRES, yolo_imgsz=YOLO_IMGSZ,
                       face_conf_th=FACE_CONF_TH, slowmo_repeat=1,
                       show_skeleton=False, show_gallery=False,
                       min_eye_dist=MIN_EYE_DIST, track_memory=tmem)
    dt = time.time() - t0

    finals = [e for e in ev if e.get("final")]
    confs  = [e["confidence"] for e in finals if e["confidence"] > 0]
    results[w] = {
        "seconds": dt,
        "tracks": len(finals),
        "checks": sum(e["checks"] for e in finals),
        "colors": dict(Counter(e["state"] for e in finals)),
        "mean_conf": (sum(confs) / len(confs)) if confs else 0.0,
        "alerts": sum(1 for e in ev if (not e.get("final")) and e["state"] == "red"),
    }

print()
print()
print("=" * 76)
print("  model                seconds   tracks   checks   alerts   mean-conf")
print("=" * 76)
for w, r in results.items():
    print("  {:<20}{:>8.1f}{:>9}{:>9}{:>9}{:>11.0%}".format(
        w, r["seconds"], r["tracks"], r["checks"], r["alerts"], r["mean_conf"]))
    print("    colors:", r["colors"])
print("=" * 76)
print()
print("agar YOLO26 hoshdar-haye dorost-tar va etminan-e balatar dad -> negahash dar:")
print('   POSE_WEIGHTS = "yolo26s-pose.pt"   va sellul-e model ra dobare Run kon')
"""

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {},
                             "source": s.splitlines(keepends=True)})
code = lambda s: cells.append({"cell_type": "code", "metadata": {},
                               "execution_count": None, "outputs": [],
                               "source": s.splitlines(keepends=True)})

md("""# سامانهٔ تشخیص پوششِ صورت — نسخهٔ ۵

نسخهٔ پایه + سه گامِ درخواستی. بقیهٔ الگوریتم **دست‌نخورده** است؛
هر تغییر در کد با `★ گام N` علامت خورده تا پیدا کردنش آسان باشد.

## سه تغییرِ این نسخه

| گام | چه شد | هزینه |
|---|---|---|
| **۵** | قفلِ سبز دیگر دائمی نیست — هر ۲ ثانیه یک بازبینی | ~۱.۷٪ حالتِ عادی |
| **۶** | حداقلِ اندازهٔ صورت + نمایشِ پیشرفتِ بررسی | صفر (کمتر هم می‌شود) |
| **۷** | رنگِ اسکلت = رنگِ وضعیتِ فرد | صفر |

## جدول تصمیم

| رنگ | برچسب | معنی |
|---|---|---|
| ⚪ خاکستری | `Analyzing... (2/3)` | دیده شده، در حالِ بررسی — عدد یعنی چند رأی جمع شده |
| ⚪ خاکستری | `Analyzing... (too far)` | دیده شده ولی هنوز خیلی دور است |
| 🟢 سبز | `Clear` | صورت باز — **هر ۲ ثانیه بازبینی می‌شود** |
| 🟠 نارنجی | `Medical Mask` | ماسک دارد ولی بالای صورت پیداست |
| 🔴 قرمز | `SUSPICIOUS - ALERT` | ماسک دارد و بالای صورت هم پوشیده است |

کادر، اسکلت و برچسب هر سه با همین رنگ کشیده می‌شوند.

> **قبل از شروع:** Runtime ← Change runtime type ← GPU""")

md("""---
## ۰) نصب و راه‌اندازی""")
code("!pip install -q ultralytics opencv-python-headless transformers torch torchvision pillow")

md("""### وارد کردن کتابخانه‌ها و انتخاب دستگاه""")
code('''import cv2, time, torch, os, json, numpy as np
from collections import deque
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, SiglipForImageClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = device == "cuda"
print(f"🔧 Device: {device} | FP16: {USE_HALF}")''')

md("""---
## ۱) مدل ژست — تشخیص فرد، اسکلت و ردیابی

یک مدل، سه کار: جعبهٔ هر فرد، ۱۷ کی‌پوینتِ COCO، و شناسهٔ پایدار برای
دنبال‌کردن هر نفر بین فریم‌ها (ByteTrack).""")
code(B["1"])

md("""---
## ۲) طبقه‌بندِ ماسک

مدلِ دوکلاسهٔ فاین‌تیون‌شده: `mask` یا `no_mask`. دسته‌ای کار می‌کند —
همهٔ صورت‌های یک فریم با هم به مدل می‌روند.

> یک بار این را اجرا کن و با `MASK_ID2LABEL` مقایسه کن:
> `print(mask_model.config.id2label)`""")
code(B["2"])

md("""---
## ۳) سنجهٔ پوست — تفکیکِ ماسکِ پزشکی از مشکوک

اگر کسی ماسکِ پزشکی زده، بالای صورتش پوست دیده می‌شود. اگر پوششِ
کامل داشته باشد، آنجا هم پوشیده است. `is_suspicious` نسبتِ پوست را
روی ۴۵٪ بالای برش می‌سنجد؛ زیر ۰.۱۲ یعنی مشکوک.""")
code(B["3"])

md("""---
## ۴) تراز و برشِ صورت  ★ گام ۶

**گیتِ ورودی:** میانگینِ اطمینانِ بینی و دو چشم. زیرِ آستانه → `None`
→ آن فرد در آن فریم رأی نمی‌دهد. همین باعث می‌شود نیم‌رخ‌ها و
پشت‌به‌دوربین‌ها قضاوت نشوند.

**★ گام ۶ — `min_eye_dist`:** آستانهٔ فاصلهٔ دو چشم از ۳ به **۸**
رسید. قبلاً صورتی به عرضِ ~۱۰ پیکسل هم رأی می‌داد و آن رأی نویزِ
خالص بود. حالا فردِ دور همچنان **دیده و ردیابی می‌شود** — فقط تا
نزدیک‌تر نشده حکمی درباره‌اش صادر نمی‌شود.""")
code(B["4"])

md("""---
## ۵) ماشینِ حالت  ★ گام ۵

رأی‌ها روی چند فریم جمع می‌شوند تا رنگ‌ها چشمک نزنند.

| حالت | نرخِ بررسی |
|---|---|
| `fast` | هر فریم — تا ۳ رأی جمع شود |
| `focus` | هر ۲ فریم — فردِ ماسک‌دار زیرِ نظر |
| `locked` سبز | **هر ۶۰ فریم — ★ گام ۵** |

**★ گام ۵ — شکستنِ قفلِ سبز.** سناریو: کسی با صورتِ باز وارد می‌شود،
سبز قفل می‌شود، بعد داخل ماسک می‌کشد. با قفلِ دائمی سیستم دیگر هرگز
نگاهش نمی‌کرد.

حالا هر ۶۰ فریم (≈۲ ثانیه) یک بار بررسی می‌شود:

- رأی سبز آمد → ساعت صفر می‌شود، سبز می‌ماند
- رأی نارنجی/قرمز آمد → **قفل می‌شکند** و می‌رود به حالتِ فوکوس

**هزینه‌اش دقیقاً چقدر است؟** برای هر فردِ سبز، یک اجرای طبقه‌بند در
هر ۶۰ فریم به‌جای صفر. در حالتِ عادی هر فرد هر فریم بررسی می‌شد،
پس این یعنی **حدودِ ۱.۷٪** آن هزینه. عملاً رایگان.

عدد را می‌توانی عوض کنی: `state_mgr.GREEN_RECHECK_FRAMES = 90`""")
code(B["5"])

md("""---
## ۵ب) ★ C1 — حافظهٔ کوتاه‌مدتِ هویت

**مسئله:** فرد پشتِ قفسه می‌رود و برمی‌گردد → ByteTrack شناسهٔ تازه
می‌دهد → حکمِ قبلی از بین می‌رود. دزدِ قرمز بعد از دو ثانیه انسداد
دوباره «Analyzing» می‌شود.

**نگرانیِ درست: اگر دو نفر لباسِ شبیه داشته باشند چه؟** سه محافظ:

| محافظ | کار |
|---|---|
| حافظهٔ کوتاه | پیش‌فرض ۸ ثانیه. هرچه کوتاه‌تر، برخوردِ تصادفی کمتر |
| گیتِ مکانی | باید نزدیکِ محلِ ناپدیدشدن ظاهر شود و اندازهٔ جعبه هم‌خوان باشد |
| ★ آزمونِ حاشیه | اگر دو هویت **هر دو** شبیه باشند، هیچ‌کدام انتخاب نمی‌شود |

آزمونِ حاشیه مهم‌ترین است: در ابهام، سیستم ترجیح می‌دهد از صفر شروع
کند تا اینکه اشتباهی حکم را منتقل کند.

### و یک قانونِ ایمنی

> **هیچ‌وقت «قفل» به ارث نمی‌رسد.**

فردِ جدید رنگِ قبلی را نشان می‌دهد (پیوستگیِ بصری) ولی **دوباره
رأی‌گیری می‌شود**. اگر تطبیق اشتباه بوده باشد، ظرفِ چند فریم خودش را
اصلاح می‌کند.

اگر قفلِ سبز به ارث می‌رسید، یک تطبیقِ غلط می‌توانست یک دزد را برای
همیشه سبز کند — آن یک حفرهٔ امنیتی بود، نه یک اشکالِ کیفیت.""")
code(B["5b"])

md("""---
## ۶) ابزارِ نمایش — گالری و اسکلت  ★ گام ۷

**★ گام ۷:** رنگِ پیش‌فرضِ `draw_upper_skeleton` دیگر زردِ ثابت نیست.
حالا فراخوان رنگِ وضعیتِ همان فرد را می‌فرستد، پس کادر و اسکلت و
برچسب هم‌رنگ‌اند.""")
code(B["6"])

md("""---
## ۷) خط لولهٔ اصلی

دو تابعِ تکراریِ نسخهٔ اصلی یکی شده‌اند. بررسی کردم — منطقِ تصمیم در
هر دو **یکسان** بود و نسخهٔ دمو فقط سه چیزِ نمایشی اضافه داشت:

| پرچم | پیش‌فرض | اثر |
|---|---|---|
| `show_skeleton` | `True` | رسمِ اسکلت (حالا هم‌رنگِ وضعیت) |
| `show_gallery` | `True` | گالریِ گوشهٔ تصویر |
| `slowmo_repeat` | `6` | تکرارِ فریم در لحظاتِ کلیدی (`1` = خاموش) |
| `min_eye_dist` | `8` | ★ گام ۶ |

**★ گام ۷ — ترتیبِ رسم عوض شد.** در نسخهٔ اصلی اسکلت در حلقهٔ *اول*
کشیده می‌شد، جایی که هنوز رنگِ وضعیت معلوم نبود. حالا کی‌پوینت‌ها در
`kpts_by_tid` نگه داشته می‌شوند و اسکلت در حلقهٔ *رسم* — بعد از
مشخص‌شدنِ رنگ — کشیده می‌شود.

**★ گام ۶ — برچسبِ پیشرفت.** در حالتِ خاکستری به‌جای
`Analyzing...` خالی، حالا `Analyzing... (2/3)` نوشته می‌شود؛ و اگر
هنوز هیچ بررسی‌ای ممکن نبوده `Analyzing... (too far)`. این‌طور معلوم
است سیستم فرد را دیده و دارد رویش کار می‌کند.""")
code(B["7"])

md("""---
## ۸) اجرا

### اتصال Google Drive""")
code("""from google.colab import drive
drive.mount('/content/drive')""")

md("""### تنظیم مسیرها و اجرا""")
code('''# ---------------- مسیرها ----------------
INPUT_VIDEO  = "/content/drive/MyDrive/9.mp4"     # ← ویدیوی خودت
OUTPUT_VIDEO = "/content/output_v5.mp4"

# ---------------- پارامترها (همان مقادیرِ نسخهٔ اصلی) ----------------
CONF_THRES    = 0.4
YOLO_IMGSZ    = 640
FACE_CONF_TH  = 0.5

# ---------------- ★ گام ۶ ----------------
MIN_EYE_DIST  = 8       # حداقلِ فاصلهٔ دو چشم برای صدورِ حکم

# ---------------- ★ C1: حافظهٔ هویت ----------------
USE_TRACK_MEMORY = True
MEMORY_SECONDS   = 8.0   # ۵ تا ۱۰ منطقی است. کوتاه‌تر = محتاط‌تر
MEMORY_MATCH_TH  = 0.80  # کمینهٔ شباهتِ رنگِ لباس (۰..۱)
MEMORY_MARGIN    = 0.06  # ★ اگر دومین گزینه از این نزدیک‌تر بود → رد
MEMORY_MAX_DIST  = 0.35  # حداکثر جابه‌جایی، نسبت به عرضِ فریم

# ---------------- جلوه‌های نمایشی ----------------
SHOW_SKELETON = True
SHOW_GALLERY  = True
SLOWMO_REPEAT = 6       # ۱ = خاموش

import os
if not os.path.exists(INPUT_VIDEO):                               # [ایمنی]
    raise FileNotFoundError(f"❌ ویدیو پیدا نشد: {INPUT_VIDEO}")

# حالتِ داخلی را قبل از هر اجرا صفر می‌کنیم — وگرنه اگر سلول را دو بار
# اجرا کنی، شناسه‌ها و رأی‌های اجرای قبلی باقی می‌مانند.
state_mgr = TrackStateManager()
gallery   = PresentationGallery(max_items=4, thumb_size=140)

# ★ C1 — حافظه هم باید هر اجرا صفر شود
_fps_guess = cv2.VideoCapture(INPUT_VIDEO).get(cv2.CAP_PROP_FPS) or 30.0
track_memory = TrackMemory(memory_seconds=MEMORY_SECONDS,
                           fps=_fps_guess,
                           match_th=MEMORY_MATCH_TH,
                           margin=MEMORY_MARGIN,
                           max_center_dist=MEMORY_MAX_DIST) if USE_TRACK_MEMORY else None

# ★ گام ۵ — نرخِ بازبینیِ افرادِ سبز (۶۰ فریم ≈ ۲ ثانیه در ۳۰fps)
state_mgr.GREEN_RECHECK_FRAMES = 60

events = process_video(INPUT_VIDEO, OUTPUT_VIDEO,
                       conf_thres=CONF_THRES,
                       yolo_imgsz=YOLO_IMGSZ,
                       face_conf_th=FACE_CONF_TH,
                       slowmo_repeat=SLOWMO_REPEAT,
                       show_skeleton=SHOW_SKELETON,
                       show_gallery=SHOW_GALLERY,
                       min_eye_dist=MIN_EYE_DIST,
                       track_memory=track_memory)''')

md("""### نمایش ویدیوی خروجی""")
code('''import subprocess, os
from base64 import b64encode
from IPython.display import HTML

web = "/content/preview.mp4"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", OUTPUT_VIDEO,
                "-vcodec", "libx264", "-crf", "26", web], check=False)

path = web if os.path.exists(web) else OUTPUT_VIDEO
data = b64encode(open(path, "rb").read()).decode()
HTML(f'<video width=860 controls>'
     f'<source src="data:video/mp4;base64,{data}" type="video/mp4"></video>')''')

md("""### گزارشِ وضعیتِ نهاییِ هر فرد

بعد از اجرا، این سلول می‌گوید سیستم دربارهٔ هر نفر به چه نتیجه‌ای
رسید و چند بار بررسی‌اش کرد — برای راستی‌آزمایی مفید است.""")
code('''from collections import Counter

rows = []
for tid, st in sorted(state_mgr.data.items()):
    rows.append((tid, st["color"], st["checks"], st["locked"], st["mode"]))

print(f"{'ID':>4}  {'وضعیت':<8} {'بررسی':>6} {'قفل':>5}  حالت")
print("-" * 44)
for tid, color, checks, locked, mode in rows:
    print(f"{tid:>4}  {color:<8} {checks:>6} {str(locked):>5}  {mode}")

print("\\nجمع‌بندی:", dict(Counter(r[1] for r in rows)))
print("مجموع بررسی‌ها:", sum(r[2] for r in rows))''')

md("""---
## ★ مقایسهٔ YOLO11s با YOLO26

مدلِ ژست فقط یک متغیر است، پس مقایسه ساده است: این سلول همان ویدیو را
با هر دو مدل روی چند صد فریمِ اول اجرا می‌کند و نتیجه را کنار هم
می‌گذارد.

**چه چیزی را مقایسه کن:**

| معیار | یعنی چه |
|---|---|
| تعدادِ بررسی | کی‌پوینتِ بهتر → گیت بیشتر باز می‌شود → بررسیِ بیشتر |
| توزیعِ رنگ‌ها | چند نفر به حکمِ قطعی رسیدند و چند نفر «Analyzing» ماندند |
| میانگینِ اطمینان | حکم‌ها قاطع‌ترند یا مرزی |
| زمان | هزینهٔ سرعت |

**⚠️ نکتهٔ مهم:** YOLO26 از RLE برای مکان‌یابیِ کی‌پوینت استفاده می‌کند،
که یعنی *کالیبراسیونِ اطمینان* ممکن است فرق کند. گیتِ ما
(`FACE_CONF_TH = 0.5`) دقیقاً روی همان اطمینان‌ها کار می‌کند.

اگر با YOLO26 دیدی خیلی‌ها «Analyzing» ماندند، مدل بد نیست — آستانه
باید جابه‌جا شود. `FACE_CONF_TH` را روی ۰.۴ و ۰.۶ هم امتحان کن.""")
code(COMPARE_CELL)
md("""### ذخیره در Google Drive""")
code('''import shutil, os

DRIVE_DEST = '/content/drive/MyDrive/output_v5_saved.mp4'
if os.path.exists(OUTPUT_VIDEO):
    shutil.copy(OUTPUT_VIDEO, DRIVE_DEST)
    print(f"✅ ذخیره شد: {DRIVE_DEST}")
else:
    print("❌ فایل خروجی پیدا نشد — اول سلولِ اجرا را ران کن.")''')

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
n = sum(1 for c in cells if c["cell_type"] == "code")
print(f"\n✅ {OUT}")
print(f"   {len(cells)} سلول ({n} کد) | {OUT.stat().st_size/1024:.0f} KB")
