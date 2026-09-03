# ---------------- 4.5) جهتِ سر — رو به دوربین یا پشت؟ ------------------
#
# ★ فاز E  (نسخهٔ ۲ — مقاوم‌شده)
#
# ایدهٔ اصلی: چیرالیته (دست‌وارگی)
# --------------------------------
# کی‌پوینت‌های COCO برچسبِ **آناتومیک** دارند: اندیس ۵ «شانهٔ چپِ خودِ
# شخص» است، نه «شانهٔ سمتِ چپِ تصویر». پس وقتی کسی برمی‌گردد، برچسب‌ها
# در تصویر جابه‌جا می‌شوند:
#
#     رو به دوربین   →   x(شانهٔ چپ) − x(شانهٔ راست)   مثبت
#     پشت به دوربین  →   همان اختلاف                   منفی
#
# این سیگنال به دیده‌شدنِ صورت کاری ندارد، پس روی نقاب‌دار هم کار می‌کند.
#
#
# ★★ چرا نسخهٔ اول گاهی اشتباه «پشت» می‌گفت — و چه شد
# ═══════════════════════════════════════════════════════════════════
#
# اندازه‌گیریِ عینی روی همان کد نشان داد مقصر «سرنخِ دیده‌شدنِ صورت» بود:
#
#     حالت           چشم  بینی  گوش   v_vis
#     صورتِ باز      0.92  0.90  0.80  +1.00
#     ماسک‌دار       0.12  0.05  0.80  −0.48   ⚠️
#     نقابِ کامل     0.02  0.02  0.85  −0.64   ⚠️
#     دور / تار      0.02  0.02  0.30  −0.20   ⚠️
#
# یعنی **هر کسی که صورتش پوشیده بود، به‌سمتِ «پشت» هل داده می‌شد** —
# با اینکه پوشیدگی هیچ ربطی به جهت ندارد. اگر همان لحظه شانه‌ها هم
# ضعیف دیده می‌شدند (پشتِ پیشخوان، شلوغی، لبهٔ کادر)، حکمِ «پشت»
# صادر می‌شد. دقیقاً همان اشتباهی که در ویدیو دیدی.
#
# پنج سخت‌سازی
# ------------
#   H1  سرنخِ دیده‌شدن **یک‌طرفه** شد. فقط می‌تواند به‌سمتِ «روبه‌رو»
#       هل بدهد، هرگز به‌سمتِ «پشت».
#       استدلال: «صورت را می‌بینم» ⇒ قطعاً روبه‌روست. ولی
#       «صورت را نمی‌بینم» ⇒ **هیچ چیزی** دربارهٔ جهت نمی‌گوید؛
#       می‌تواند ماسک باشد. استنتاجِ اول معتبر است، دومی نیست.
#
#   H2  رأیِ شانه **الزامی** است. بدونِ آن هرگز «پشت» اعلام نمی‌شود.
#       شانه‌ها بزرگ‌ترین فاصله را دارند و کمتر از همه دچارِ
#       جابه‌جاییِ برچسبِ چپ/راست می‌شوند.
#
#   H3  توافقِ دو سرنخ. یا دو رأیِ منفیِ مستقل، یا شانهٔ **قاطع**.
#
#   H4  هیسترزیس. ورود به «پشت» سخت‌تر از ماندن در آن.
#
#   H5  شیرِ اطمینان (در خط لوله). حتی وقتی «پشت» تشخیص داده شد،
#       هر N فریم یک بار به‌هرحال بررسی می‌شود — تا یک خطای پایدارِ
#       جهت‌گیری هرگز نتواند کسی را برای همیشه پنهان کند.

# اندیسِ لگن — در سلولِ مدلِ ژست تعریف نشده و این بلوک به آن نیاز دارد.
LHIP, RHIP = 11, 12

# ---- آستانه‌ها (در سلولِ تنظیماتِ بالا هم قابلِ تغییرند) ----------------
BACK_ENTER      = -0.35    # برای *ورود* به حالتِ پشت  (H4)
BACK_EXIT       = -0.20    # برای *خروج* از حالتِ پشت  (H4)
ORI_MIN_CONF    = 0.35     # زیرِ این اطمینان، اصلاً نظر نده
ORI_HISTORY     = 7        # طولِ تاریخچهٔ هر فرد
ORI_MIN_SAMPLES = 4        # حداقل نمونه تا حکمِ «پشت»
ORI_NEG_RATIO   = 0.60     # چند درصدِ تاریخچه باید منفی باشد
SHOULDER_STRONG = -0.55    # «شانهٔ قاطع» یعنی این‌قدر منفی  (H3)

# سازگاری با نامِ قدیمی
BACK_THRESHOLD = BACK_ENTER


def _chirality(kxy, kconf, i_left, i_right, conf_th, expected_sep=None):
    """
    یک رأیِ چیرالیته از یک جفتِ کی‌پوینتِ چپ/راست.
    خروجی: (مقدار در ‎[−۱,+۱]‎ ، وزن) یا None.

    «کوتاه‌شدگی»: در نیم‌رخ دو نقطه روی هم می‌افتند و علامت نویزی
    می‌شود. فاصلهٔ فعلی را با فاصلهٔ *مورد انتظار* می‌سنجیم و وزن را
    به همان نسبت کم می‌کنیم — پس نیم‌رخ خودش را «نامطمئن» اعلام می‌کند.
    """
    cl, cr = float(kconf[i_left]), float(kconf[i_right])
    if cl < conf_th or cr < conf_th:
        return None
    lx, ly = float(kxy[i_left][0]), float(kxy[i_left][1])
    rx, ry = float(kxy[i_right][0]), float(kxy[i_right][1])
    dx = lx - rx
    sep = math.sqrt(dx * dx + (ly - ry) ** 2)
    if sep < 2.0:
        return None

    v = dx / (0.70 * sep)
    v = 1.0 if v > 1.0 else (-1.0 if v < -1.0 else v)
    w = min(cl, cr)
    if expected_sep and expected_sep > 1e-3:
        f = sep / expected_sep
        w *= 1.0 if f > 1.0 else f
    return v, w


def _torso_len(kxy, kconf, conf_th):
    """طولِ تنه — مقیاسی که با چرخشِ فرد حولِ محورِ عمودی تغییر نمی‌کند."""
    sx = sy = n = 0.0
    for i in (LSHOULDER, RSHOULDER):
        if kconf[i] >= conf_th:
            sx += float(kxy[i][0]); sy += float(kxy[i][1]); n += 1
    if n == 0:
        return None
    sx /= n; sy /= n

    hx = hy = m = 0.0
    for i in (LHIP, RHIP):
        if kconf[i] >= conf_th:
            hx += float(kxy[i][0]); hy += float(kxy[i][1]); m += 1
    if m == 0:
        return None
    hx /= m; hy /= m

    d = math.sqrt((sx - hx) ** 2 + (sy - hy) ** 2)
    return d if d > 5.0 else None


def head_orientation(kxy, kconf, conf_th=0.35):
    """
    خروجی: (facing ، confidence ، info)

        facing      −۱ کاملاً پشت … ۰ نیم‌رخ … +۱ کاملاً رو به دوربین
        confidence  ۰..۱
        info        دیکشنریِ جزئیات — برای گزارش و برای قواعدِ H2/H3

    هزینه: ~۴۱ میکروثانیه. هیچ مدلی اجرا نمی‌شود.
    """
    torso = _torso_len(kxy, kconf, conf_th)
    body = 0.65 * torso if torso else None      # عرضِ شانه در حالتِ روبه‌رو

    votes = []
    info = {"sh": None, "ear": None, "eye": None, "vis": 0.0, "neg": 0}

    r = _chirality(kxy, kconf, LSHOULDER, RSHOULDER, conf_th, body)
    if r is not None:
        votes.append((r[0], 1.00 * r[1])); info["sh"] = r[0]

    r = _chirality(kxy, kconf, LEAR, REAR, conf_th,
                   0.40 * body if body else None)
    if r is not None:
        votes.append((r[0], 0.85 * r[1])); info["ear"] = r[0]

    r = _chirality(kxy, kconf, LEYE, REYE, conf_th,
                   0.16 * body if body else None)
    if r is not None:
        votes.append((r[0], 0.70 * r[1])); info["eye"] = r[0]

    # ★ H1 — سرنخِ دیده‌شدن، حالا **یک‌طرفه**.
    #   «صورت را می‌بینم» ⇒ قطعاً روبه‌رو  (استنتاجِ معتبر)
    #   «صورت را نمی‌بینم» ⇒ هیچ  (می‌تواند ماسک باشد)
    #   پس فقط بخشِ مثبتش را نگه می‌داریم.
    eyes = (float(kconf[LEYE]) + float(kconf[REYE])) / 2.0
    nose = float(kconf[NOSE])
    v_vis = (0.6 * eyes + 0.6 * nose) * 1.6 - 0.30
    v_vis = max(0.0, min(1.0, v_vis))           # ← هرگز منفی نمی‌شود
    if v_vis > 0.0:
        votes.append((v_vis, 0.30))
        info["vis"] = v_vis

    if not votes:
        return 0.0, 0.0, info

    ws = sum(w for _, w in votes)
    if ws < 1e-6:
        return 0.0, 0.0, info
    facing = sum(v * w for v, w in votes) / ws
    info["neg"] = sum(1 for v in (info["sh"], info["ear"], info["eye"])
                      if v is not None and v < -0.15)

    decisive = sum(abs(v) * w for v, w in votes) / ws
    conf = min(1.0, ws / 1.6) * (0.35 + 0.65 * decisive)
    conf = 1.0 if conf > 1.0 else (0.0 if conf < 0.0 else conf)
    return float(facing), float(conf), info


def back_verdict(st, facing, conf, info):
    """
    آیا این فرد «پشت به دوربین» است؟

    این تابع عمداً از خودِ `head_orientation` جداست: آنجا فقط اندازه‌گیری
    می‌شود، اینجا **تصمیم** گرفته می‌شود. قواعدِ سخت‌گیرانه اینجایند تا
    بشود بدونِ دست‌زدن به اندازه‌گیری، محافظه‌کارتر یا آزادترشان کرد.

    برمی‌گرداند: (is_back, reason)
    """
    # اندازه‌گیریِ بی‌کیفیت اصلاً واردِ تاریخچه نمی‌شود
    if conf >= ORI_MIN_CONF:
        st["facing_hist"].append(facing)
        # ★ H2 — رأیِ شانه در همین فریم بود یا نه؟ برای قاعدهٔ الزام.
        st["sh_hist"].append(info["sh"] if info["sh"] is not None else 0.0)

    h = list(st["facing_hist"])
    sh = list(st["sh_hist"])
    was_back = st.get("is_back", False)

    if len(h) < ORI_MIN_SAMPLES:
        return False, "نمونهٔ کافی نیست"

    med = sorted(h)[len(h) // 2]
    neg_ratio = sum(1 for x in h if x < 0) / len(h)
    thr = BACK_EXIT if was_back else BACK_ENTER      # ★ H4 هیسترزیس

    if med > thr:
        return False, f"میانه {med:+.2f} > آستانه {thr:+.2f}"

    # ★ H5 — بیشترِ تاریخچه باید منفی باشد، نه فقط میانه
    if neg_ratio < ORI_NEG_RATIO:
        return False, f"فقط {neg_ratio:.0%} تاریخچه منفی است"

    # ★ H2 — بدونِ شواهدِ شانه هرگز «پشت» اعلام نکن
    sh_med = sorted(sh)[len(sh) // 2] if sh else 0.0
    if sh_med >= 0.0:
        return False, "شانه‌ها «پشت» را تأیید نمی‌کنند"

    # ★ H3 — یا شانهٔ قاطع، یا توافقِ دو سرنخ
    if sh_med > SHOULDER_STRONG and info["neg"] < 2:
        return False, f"شانه ضعیف ({sh_med:+.2f}) و توافق ندارد"

    return True, f"میانه {med:+.2f} | شانه {sh_med:+.2f} | {neg_ratio:.0%} منفی"
