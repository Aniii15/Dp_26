#!/usr/bin/env python3
"""Build the BladeVision 3.0 poster (A2 portrait, 420 x 594 mm).

Everything is drawn here, so the poster is reproducible and every number on it
is traceable to this repository:

    python build_poster.py --dpi 96   --out proof.png     # fast proof
    python build_poster.py --dpi 200                       # print file

Type: Archivo Black (display), Barlow Condensed (heads), Barlow (text),
Space Mono (technical labels).  Palette: navy ground, cyan/teal accents, amber
for numbers that were measured on the rig.
"""
from __future__ import annotations

import argparse
import math
import os

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
FONTS = os.path.join(ASSETS, "fonts")

W_MM, H_MM = 420.0, 594.0
MARGIN_MM = 15.0

NAVY = (7, 17, 30)
NAVY2 = (9, 23, 39)
PANEL = (13, 31, 50)
PANEL2 = (17, 40, 63)
AMBER_PANEL = (36, 34, 22)
BORDER = (34, 68, 100)
BORDER_HI = (48, 96, 138)
WHITE = (239, 246, 253)
BODY = (196, 214, 232)
MUTED = (141, 166, 190)
DIM = (104, 130, 156)
CYAN = (56, 214, 240)
TEAL = (48, 200, 176)
AMBER = (250, 178, 74)

_FONT_FILES = {
    "display": "ArchivoBlack-Regular.ttf",
    "head": "BarlowCondensed-Bold.ttf",
    "semibold": "Barlow-SemiBold.ttf",
    "bold": "Barlow-Bold.ttf",
    "medium": "Barlow-Medium.ttf",
    "body": "Barlow-Regular.ttf",
    "light": "Barlow-Light.ttf",
    "mono": "SpaceMono-Regular.ttf",
}
_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


class Canvas:
    def __init__(self, dpi: int):
        self.dpi = dpi
        self.k = dpi / 25.4
        self.w = int(round(W_MM * self.k))
        self.h = int(round(H_MM * self.k))
        self.img = Image.new("RGB", (self.w, self.h), NAVY)
        self.d = ImageDraw.Draw(self.img)

    def px(self, mm: float) -> int:
        return int(round(mm * self.k))

    def font(self, kind: str, mm: float, scale: int = 1) -> ImageFont.FreeTypeFont:
        size = max(6, int(round(mm * self.k * scale)))
        key = (kind, size)
        if key not in _cache:
            _cache[key] = ImageFont.truetype(
                os.path.join(FONTS, _FONT_FILES[kind]), size)
        return _cache[key]

    # ── text ────────────────────────────────────────────────────────────────
    def text(self, x, y, s, font, color=WHITE, anchor="la", spacing=None,
             leading=1.14):
        if "\n" in s:
            for i, line in enumerate(s.split("\n")):
                self.text(x, y + i * font.size * leading / self.k, line,
                          font, color, anchor, spacing)
            return
        if spacing:
            self._tracked(x, y, s, font, color, spacing, anchor)
            return
        self.d.text((self.px(x), self.px(y)), s, font=font, fill=color,
                    anchor=anchor)

    def _tracked(self, x, y, s, font, color, spacing_mm, anchor):
        extra = spacing_mm * self.k
        widths = [self.d.textlength(ch, font=font) for ch in s]
        total = sum(widths) + extra * (len(s) - 1)
        px = self.px(x)
        if anchor.startswith("m"):
            px -= total / 2.0
        elif anchor.startswith("r"):
            px -= total
        top = font.getbbox("Hg")[1]
        py = self.px(y) - top
        for ch, wd in zip(s, widths):
            self.d.text((px, py), ch, font=font, fill=color)
            px += wd + extra

    def wrap(self, s: str, font, width_mm: float) -> list[str]:
        limit = self.px(width_mm)
        out, cur = [], ""
        for word in s.split():
            trial = (cur + " " + word).strip()
            if not cur or self.d.textlength(trial, font=font) <= limit:
                cur = trial
            else:
                out.append(cur)
                cur = word
        if cur:
            out.append(cur)
        return out

    def para(self, x, y, w, s, font, color=BODY, leading=1.28) -> float:
        """Draw wrapped text; returns the y of the next free line."""
        step = font.size * leading / self.k
        for ln in self.wrap(s, font, w):
            self.d.text((self.px(x), self.px(y)), ln, font=font, fill=color)
            y += step
        return y

    def chip(self, x, y, label, font, fg=CYAN, bg=None, border=None,
             pad_x=2.0, pad_y=1.0):
        w = self.d.textlength(label, font=font) / self.k + 2 * pad_x
        h = font.size * 1.35 / self.k + 2 * pad_y
        self.d.rounded_rectangle(
            [self.px(x), self.px(y), self.px(x + w), self.px(y + h)],
            radius=self.px(1.6), fill=bg, outline=border, width=1)
        self.d.text((self.px(x + pad_x),
                     self.px(y) + (self.px(y + h) - self.px(y) - font.size) / 2
                     - font.size * 0.10), label, font=font, fill=fg)
        return w, h


def load_cover(path: str, w_px: int, h_px: int, focus_y=0.35) -> Image.Image:
    im = Image.open(path).convert("RGB")
    sw, sh = im.size
    target = w_px / h_px
    if sw / sh > target:
        nw = int(sh * target)
        x0 = (sw - nw) // 2
        im = im.crop((x0, 0, x0 + nw, sh))
    else:
        nh = int(sw / target)
        y0 = int((sh - nh) * focus_y)
        im = im.crop((0, y0, sw, y0 + nh))
    return im.resize((w_px, h_px), Image.LANCZOS)


def load_fit(path: str, w_px: int, h_px: int) -> tuple[Image.Image, int]:
    """Scale to fit *inside* the box, never crop.  Returns (image, y offset)."""
    im = Image.open(path).convert("RGB")
    sw, sh = im.size
    k = min(w_px / sw, h_px / sh)
    nw, nh = max(1, int(sw * k)), max(1, int(sh * k))
    return im.resize((nw, nh), Image.LANCZOS), (h_px - nh) // 2


def rounded_mask(size, radius):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1],
                                        radius=radius, fill=255)
    return m


def fade(img, side="up", color=NAVY, strength=200):
    w, h = img.size
    grad = Image.new("L", (1, h) if side in ("up", "down") else (w, 1))
    px = grad.load()
    n = h if side in ("up", "down") else w
    for i in range(n):
        v = strength * (i / max(1, n - 1)) ** 1.4
        if side == "up":
            v = strength * (1 - i / max(1, n - 1)) ** 1.2
        if side in ("up", "down"):
            px[0, i] = int(v)
        else:
            px[i, 0] = int(v)
    grad = grad.resize((w, h))
    tint = Image.new("RGB", (w, h), color)
    out = img.copy()
    out.paste(tint, (0, 0), grad)
    return out


# ─────────────────────────────────────────────────────────────────────────────
def build(dpi: int, out: str) -> str:
    c = Canvas(dpi)
    X0, X1 = MARGIN_MM, W_MM - MARGIN_MM
    CW = X1 - X0
    d = c.d

    # ── ground ──────────────────────────────────────────────────────────────
    c.img.paste(NAVY2, (0, 0, c.w, c.h))
    glow = Image.new("RGB", (c.w, c.h), NAVY2)
    gd = ImageDraw.Draw(glow)
    for i in range(16):
        r = c.px(150 + i * 14)
        gd.ellipse([c.px(300) - r, c.px(-30) - r, c.px(300) + r, c.px(-30) + r],
                   fill=(10 + i, 30 + i * 2, 52 + i * 2))
    glow = glow.filter(ImageFilter.GaussianBlur(c.px(20)))
    c.img.paste(Image.blend(c.img, glow, 0.78), (0, 0))
    d = c.d = ImageDraw.Draw(c.img)
    yy = 0.0
    while yy < H_MM:
        d.line([(0, c.px(yy)), (c.w, c.px(yy))], fill=(11, 27, 44), width=1)
        yy += 6.0
    xx = 0.0
    while xx < W_MM:
        d.line([(c.px(xx), 0), (c.px(xx), c.h)], fill=(11, 27, 44), width=1)
        xx += 6.0

    # ══ MASTHEAD ════════════════════════════════════════════════════════════
    BAND = 150.0
    # The hero keeps its full height: the blade is 768x1376 (0.56 aspect), so a
    # ~84 mm wide plate at 150 mm tall fits it almost exactly instead of
    # cropping the tip or the stage.  Everything else gets the space back.
    hero_w = 84.0
    hero_x = X1 - hero_w
    hero, oy = load_fit(os.path.join(ASSETS, "blade_hero.png"),
                        c.px(hero_w), c.px(BAND))
    hero = fade(hero, "up", NAVY, 120)
    d.rounded_rectangle([c.px(hero_x) - 1, 0, c.px(hero_x + hero_w),
                         c.px(BAND)], radius=c.px(1.5), fill=(12, 26, 42),
                         outline=BORDER, width=1)
    hw, hh = hero.size
    hero_rgba = hero.convert("RGBA")
    veil = Image.new("RGBA", (hw, hh), (0, 0, 0, 0))
    vd = ImageDraw.Draw(veil)
    for i in range(hw):
        a = int(150 * (1 - i / hw) ** 1.6)
        vd.line([(i, 0), (i, hh)], fill=NAVY + (a,))
    hero_rgba = Image.alpha_composite(hero_rgba, veil).convert("RGB")
    c.img.paste(hero_rgba, (c.px(hero_x) + (c.px(hero_w) - hw) // 2, oy),
                rounded_mask(hero_rgba.size, c.px(1.2)))

    # ── institute mark, on a white plate (the logo is a white-ground original)
    LW, LH = 56.0, 30.0
    d.rounded_rectangle([c.px(X0), c.px(9), c.px(X0 + LW), c.px(9 + LH)],
                        radius=c.px(1.8), fill=(255, 255, 255))
    logo = Image.open(os.path.join(ASSETS, "iit_mandi_logo.png")).convert("RGB")
    lw = c.px(LW - 4.0)
    lh = int(round(lw * logo.size[1] / logo.size[0]))
    if lh > c.px(LH - 4.0):
        lh = c.px(LH - 4.0)
        lw = int(round(lh * logo.size[0] / logo.size[1]))
    logo = logo.resize((lw, lh), Image.LANCZOS)
    c.img.paste(logo, (c.px(X0 + (LW - lw / c.k) / 2),
                       c.px(9 + (LH - lh / c.k) / 2) + 3))

    kx = X0 + LW + 6.0
    c.text(kx, 11.0, "INDIAN INSTITUTE OF TECHNOLOGY MANDI", c.font("head", 4.4),
           CYAN, spacing=0.45)
    c.text(kx, 16.6, "Group 62  ·  Precision. Detection. Reliability.",
           c.font("body", 3.9), MUTED)

    # ── title, sized to the space that is actually free
    gap_r = hero_x - 8.0
    title_src = "BLADEVISION"
    size = 30.0
    while size > 16.0:
        f = c.font("display", size)
        t = d.textlength(title_src, font=f) / c.k
        t2 = d.textlength("3.0", font=c.font("display", size * 0.62)) / c.k
        if X0 + t + 4.0 + t2 <= gap_r:
            break
        size -= 0.4
    title_f = c.font("display", size)
    c.text(X0, 40.0, title_src, title_f, WHITE, spacing=-0.3)
    tw = d.textlength(title_src, font=title_f) / c.k
    c.text(X0 + tw + 4.0, 40.0 + size * 0.34, "3.0",
           c.font("display", size * 0.62), CYAN, spacing=-0.3)

    d.line([(c.px(X0), c.px(76)), (c.px(X0 + 86), c.px(76))], fill=CYAN,
           width=max(2, c.px(0.5)))
    c.text(X0, 81.0, "From a stepper pulse to a stitched mosaic.",
           c.font("semibold", 8.4), WHITE)
    c.para(X0, 92.0, min(226.0, gap_r - X0),
           "A closed-loop, boundary-guided scanner that photographs a turbine "
           "blade at full coverage and rebuilds it as one continuous, "
           "defect-ready image — with no position sensor on any axis.",
           c.font("body", 4.1), BODY)

    for i, (big, small) in enumerate([
            ("1.8 mm", "field of view\nper 1080p frame"),
            ("68 %", "measured frame\noverlap (target 65)"),
            ("43 %", "of the captures\nof the old fixed steps"),
            ("13.3 k", "lines of Python\nacross 14 modules")]):
        sx = X0 + i * 56.0
        c.text(sx, 114.0, big, c.font("display", 10.8), CYAN)
        c.text(sx, 126.0, small, c.font("body", 3.7), MUTED)

    # ══ SPEC BAR ════════════════════════════════════════════════════════════
    spec_y = BAND + 3.0
    d.line([(c.px(X0), c.px(spec_y)), (c.px(X1), c.px(spec_y))], fill=BORDER_HI,
           width=max(1, c.px(0.3)))
    specs = [("STAGE", "X 800 · Z 1600 steps/mm"),
             ("Y AXIS", "BTS7960 PWM actuator"),
             ("CAMERA", "Canon EOS 80D, 1080p HDMI"),
             ("LENS", "Questar QM-100 catadioptric"),
             ("TESTS", "151 checks, no rig needed")]
    sw = CW / len(specs)
    for i, (k, v) in enumerate(specs):
        x = X0 + i * sw
        c.text(x, spec_y + 4.0, k, c.font("head", 3.6), TEAL, spacing=0.5)
        c.text(x, spec_y + 8.2, v, c.font("medium", 3.9), BODY)
        if i:
            d.line([(c.px(x - 2.5), c.px(spec_y + 3.0)),
                    (c.px(x - 2.5), c.px(spec_y + 13.0))], fill=BORDER, width=1)
    d.line([(c.px(X0), c.px(spec_y + 16.0)), (c.px(X1), c.px(spec_y + 16.0))],
           fill=(22, 48, 74), width=1)

    TOP = spec_y + 22.0

    # ══ CARD FRAMEWORK ══════════════════════════════════════════════════════
    def card(x, y, w, kicker, title, accent=CYAN, pad=5.0, head=None):
        nlines = title.count("\n") + 1
        if head is None:
            head = 16.0 + (nlines - 1) * 7.8
        d.rounded_rectangle([c.px(x), c.px(y), c.px(x + w), c.px(y + head)],
                            radius=c.px(3), fill=PANEL2)
        d.rounded_rectangle([c.px(x), c.px(y), c.px(x + w), c.px(y + head)],
                            radius=c.px(3), outline=BORDER, width=1)
        c.text(x + pad, y + 2.6, kicker, c.font("head", 3.9), accent,
               spacing=0.45)
        c.text(x + pad, y + 7.0, title, c.font("head", 7.6), WHITE, leading=1.02)
        d.rectangle([c.px(x), c.px(y + head - 1.2), c.px(x + w),
                     c.px(y + head)], fill=accent)
        return y + head

    def close(x, y, w, bottom, debug=""):
        d.rounded_rectangle([c.px(x), c.px(y), c.px(x + w), c.px(bottom)],
                            radius=c.px(3), fill=None, outline=BORDER, width=1)
        if os.environ.get("POSTER_DEBUG"):
            print(f"  card {debug:12s} {y:6.1f} -> {bottom:6.1f} "
                  f"(h {bottom - y:5.1f})")
        return bottom

    COLGAP = 6.0
    col_w = (CW - 2 * COLGAP) / 3.0
    cx = [X0 + i * (col_w + COLGAP) for i in range(3)]
    GAP = 6.0
    BD = 4.1                     # body type size (mm) - one step up from v1

    # ══ COLUMN 1 ════════════════════════════════════════════════════════════
    x, y = cx[0], TOP
    tx = 5.0
    ty = card(x, y, col_w, "01 — THE PROBLEM",
              "Microns of damage, hundreds of\nmillimetres of part")
    p = ty + 3.2
    p = c.para(x + tx, p, col_w - 2 * tx,
               "A blade fails at its edges and around its film-cooling holes — "
               "defects tens of microns across, on a part hundreds of "
               "millimetres long. The field of view is 1.8 mm, so one blade is "
               "thousands of overlapping frames.",
               c.font("body", BD), BODY)
    p += 3.4
    for head, txt in [
            ("Coverage", "A manual XY stage makes the operator the positioning "
                         "system: slow, and never the same twice."),
            ("Focus", "Depth of field is 0.09 mm on a tilted surface, so most "
                      "frames are partly blurred — and a whole-frame sharpness "
                      "score peaks where nothing is sharp."),
            ("Evidence", "Nothing is recorded, so no two inspections can be "
                         "compared.")]:
        d.ellipse([c.px(x + tx), c.px(p + 1.4), c.px(x + tx + 1.5),
                   c.px(p + 2.9)], fill=CYAN)
        c.text(x + tx + 3.0, p, head, c.font("semibold", 4.1), WHITE)
        p = c.para(x + tx + 3.0, p + 4.4, col_w - 2 * tx - 3.0, txt,
                   c.font("body", 3.9), MUTED)
        p += 2.0
    y = close(x, y, col_w, p + 0.4, "01 problem") + GAP

    ty = card(x, y, col_w, "02 — THE PIPELINE", "Five stages, one closed loop",
              accent=TEAL)
    p = ty + 3.2
    for i, (name, txt) in enumerate([
            ("MOTION", "ESP32 over Wi-Fi drives X, Z and the Y actuator; a move "
                       "ends when the camera stops changing, never on a timer."),
            ("CAPTURE", "One 1080p frame per stop, logged to capture_log.csv so "
                        "the true order survives same-second file names."),
            ("FOCUS", "Tile-based autofocus: 200-step coarse walk, bracket, "
                      "50-step refine, then a 3-exposure stack."),
            ("SWEEP + LIVE STITCH", "Boundary-guided serpentine; a worker "
                                    "thread stitches each frame as it lands."),
            ("RECONSTRUCT", "Pose-graph re-stitch of every frame, sharpest-pixel "
                            "blend, exposure solve, lens mask.")], 1):
        d.rounded_rectangle([c.px(x + tx), c.px(p), c.px(x + tx + 5.8),
                             c.px(p + 5.6)], radius=c.px(1.1), fill=TEAL)
        c.text(x + tx + 2.9, p + 0.75, str(i), c.font("display", 3.3), NAVY,
               anchor="ma")
        c.text(x + tx + 9.2, p + 0.4, name, c.font("head", 4.6), WHITE,
               spacing=0.4)
        p = c.para(x + tx + 9.2, p + 6.0, col_w - 2 * tx - 9.2, txt,
                   c.font("body", 3.85), BODY)
        p += 2.8
    col1_bottom = close(x, y, col_w, p - 1.8, "02 pipeline")

    # ══ COLUMN 2 ════════════════════════════════════════════════════════════
    x, y = cx[1], TOP
    ty = card(x, y, col_w, "03 — THE SOFTWARE", "Fourteen modules, four jobs")
    p = ty + 3.2
    for name, size, txt in [
            ("main3.py", "2 965", "Operator GUI (five tabs), ESP32 HTTP layer, "
                                  "sweep state machine."),
            ("focus_engine.py", "1 301", "Tiles, bracketing, stacking, trend "
                                         "prediction, stitch weights."),
            ("scan_geometry.py", "530", "Learns the real step size from the "
                                        "stitcher; writes geometry.json."),
            ("live/global_stitch.py", "1 261", "Pose graph, per-move motion "
                                               "model, sharpest-pixel blend."),
            ("live/pipeline.py", "766", "Streaming stitcher: one decision per "
                                        "frame, non-blocking."),
            ("live/registration.py", "311", "Phase correlation on high-passed "
                                            "frames — no features needed.")]:
        c.text(x + tx, p, name, c.font("mono", 3.4), CYAN)
        nw = d.textlength(name, font=c.font("mono", 3.4)) / c.k
        c.text(x + tx + nw + 2.2, p + 0.5, size, c.font("medium", 3.0), DIM)
        p = c.para(x + tx, p + 4.7, col_w - 2 * tx, txt, c.font("body", 3.8),
                   BODY)
        p += 2.2
    p = c.para(x + tx, p + 0.4, col_w - 2 * tx,
               "+ lens, quality, mosaic, config, purge, run_live,\n"
               "  run_global, __init__  (179-235 lines each)",
               c.font("mono", 2.95), DIM)
    y = close(x, y, col_w, p + 3.0, "03 software") + GAP

    # ── data flow diagram ──
    ty = card(x, y, col_w, "04 — DATA FLOW", "One capture, two loops",
              accent=TEAL)
    p = ty + 2.4
    DIA_H = 74.0
    S = 3
    patch = Image.new("RGB", (int(col_w * c.k) * S, int(DIA_H * c.k) * S), PANEL2)
    pd = ImageDraw.Draw(patch)

    def dfont(kind, mm):
        return c.font(kind, mm, S)

    def P(a, b):
        return (a * c.k * S, b * c.k * S)

    def box(a, b, w, h, label, sub, col=CYAN, fill=(15, 42, 62), font_mm=3.25):
        pd.rounded_rectangle([P(a, b), P(a + w, b + h)], radius=2.5 * S,
                             fill=fill, outline=col, width=max(1, S - 1))
        pd.text(P(a + w / 2, b + 1.9), label, font=dfont("semibold", font_mm),
                fill=WHITE, anchor="ma")
        pd.text(P(a + w / 2, b + h - 3.1), sub, font=dfont("body", 2.6),
                fill=MUTED, anchor="ma")

    def arrow(a1, b1, a2, b2, col=BORDER_HI):
        pd.line([P(a1, b1), P(a2, b2)], fill=col, width=max(1, S - 1))
        ang = math.atan2(b2 - b1, a2 - a1)
        for sgn in (+1, -1):
            pd.line([P(a2, b2), P(a2 - 1.9 * math.cos(ang + sgn * 0.5),
                                  b2 - 1.9 * math.sin(ang + sgn * 0.5))],
                    fill=col, width=max(1, S - 1))

    LX, RX, BWD = 2.8, 31.0, 25.0
    box(LX, 1.4, BWD, 9.2, "MOTOR MOVE", "ESP32 / HTTP GET")
    box(RX, 1.4, BWD, 9.2, "FRAME", "1080p, saved + logged")
    arrow(LX + BWD + 0.6, 6.0, RX - 0.6, 6.0)

    box(LX, 14.8, BWD, 9.2, "LIVE STITCHER", "Decision: dx, dy, ncc", TEAL)
    box(RX, 14.8, BWD, 9.2, "PREVIEW MOSAIC", "grown, half scale", TEAL)
    arrow(LX + BWD / 2, 10.9, LX + BWD / 2, 14.4)
    arrow(LX + BWD + 0.6, 19.4, RX - 0.6, 19.4)

    box(LX, 28.2, BWD, 9.2, "GEOMETRY", "px/mm, real overlap", AMBER,
        fill=AMBER_PANEL)
    box(RX, 28.2, BWD, 9.2, "NEXT MOVE", "Y pulse / X step", AMBER,
        fill=AMBER_PANEL)
    arrow(LX + BWD / 2, 24.3, LX + BWD / 2, 27.8)
    arrow(LX + BWD + 0.6, 32.8, RX - 0.6, 32.8)

    lane = 40.0
    pd.line([P(RX + BWD / 2, 37.7), P(RX + BWD / 2, lane), P(LX - 1.5, lane)],
            fill=AMBER, width=max(1, S - 1))
    pd.line([P(LX - 1.5, lane), P(LX - 1.5, 2.2)], fill=AMBER,
            width=max(1, S - 1))
    pd.polygon([P(LX - 1.5, 0.6), P(LX - 2.5, 2.6), P(LX - 0.5, 2.6)], fill=AMBER)
    pd.text(P((LX + RX + BWD) / 2, lane + 1.5), "measured, never guessed",
            font=dfont("medium", 2.7), fill=AMBER, anchor="ma")

    box(LX, 47.0, BWD, 9.2, "SCAN FOLDER", "frames + log", MUTED,
        fill=(13, 34, 52))
    box(RX, 47.0, BWD, 9.2, "GLOBAL SOLVE", "all frames, IRLS", CYAN,
        fill=(13, 34, 52))
    arrow(LX + BWD + 0.6, 51.6, RX - 0.6, 51.6)
    pd.line([P(RX + BWD / 2, 56.6), P(RX + BWD / 2, 60.6), P(LX + BWD / 2, 60.6),
             P(LX + BWD / 2, 62.6)], fill=CYAN, width=max(1, S - 1))
    pd.polygon([P(LX + BWD / 2, 64.4), P(LX + BWD / 2 - 1.0, 62.4),
                P(LX + BWD / 2 + 1.0, 62.4)], fill=CYAN)
    pd.rounded_rectangle([P(LX, 65.0), P(LX + BWD, 70.8)], radius=2.5 * S,
                         fill=(20, 40, 46), outline=AMBER, width=max(1, S - 1))
    pd.text(P(LX + BWD / 2, 66.6), "DELIVERABLE MOSAIC",
            font=dfont("head", 3.2), fill=WHITE, anchor="ma")
    pd.text(P(LX + BWD / 2, 69.4), "+ report, poses, tiles",
            font=dfont("body", 2.55), fill=MUTED, anchor="ma")

    patch = patch.resize((c.px(col_w), c.px(DIA_H)), Image.LANCZOS)
    c.img.paste(patch, (c.px(x), c.px(p)))
    p += DIA_H + 0.8
    y = close(x, y, col_w, p, "04 data flow") + GAP

    ty = card(x, y, col_w, "05 — MEASURED, NOT GUESSED",
              "What the commands actually do", accent=AMBER)
    p = ty + 3.2
    rows = [("AXIS", "COMMANDED", "MEASURED", "OVERLAP"),
            ("Y", "1200 ms pulse", "344 px / 1080", "68 %"),
            ("X", "0.40 mm = 320 pulses", "426 px / 1920", "78 %"),
            ("SCALE", "800 steps / mm", "1064 px / mm", "1.80 mm")]
    cw = [col_w * 0.125, col_w * 0.375, col_w * 0.285, col_w * 0.215]
    for r, row in enumerate(rows):
        ry = p + r * 5.6
        if r == 0:
            for i, cell in enumerate(row):
                c.text(x + tx + sum(cw[:i]), ry, cell, c.font("head", 3.2),
                       TEAL, spacing=0.25)
            d.line([(c.px(x + tx), c.px(ry + 4.6)),
                    (c.px(x + col_w - tx), c.px(ry + 4.6))], fill=BORDER, width=1)
        else:
            for i, cell in enumerate(row):
                col = AMBER if i == 3 else (WHITE if i == 0 else BODY)
                fn = c.font("semibold", 3.5) if i == 0 else c.font("body", 3.4)
                c.text(x + tx + sum(cw[:i]), ry, cell, fn, col)
    p += 5.6 * len(rows) + 1.0
    p = c.para(x + tx, p, col_w - 2 * tx,
               "X was four times more conservative than the stitcher needs, and "
               "the pixel scale differs between scans — so each scan folder now "
               "records the commands that made it.",
               c.font("body", 3.6), MUTED)
    col2_bottom = close(x, y, col_w, p + 1.2, "05 measured")

    # ══ COLUMN 3 ════════════════════════════════════════════════════════════
    x, y = cx[2], TOP
    ty = card(x, y, col_w, "06 — THE RESULT", "One image, not a collage")
    p = ty + 3.2
    IMG_H = 50.0
    macro = load_cover(os.path.join(ASSETS, "real_detail.png"),
                       c.px(col_w - 2 * tx), c.px(IMG_H), focus_y=0.5)
    mx, my = x + tx, p
    c.img.paste(macro, (c.px(mx), c.px(my)),
                rounded_mask(macro.size, c.px(1.4)))
    cap = "real scan · 1 : 1 pixels · cooling-hole row"
    cap_w = d.textlength(cap, font=c.font("mono", 2.7)) / c.k + 3.2
    d.rounded_rectangle([c.px(mx + 1.2), c.px(my + 1.4),
                         c.px(mx + 1.2 + cap_w), c.px(my + 6.6)],
                        radius=c.px(0.9), fill=(10, 20, 30))
    c.text(mx + 2.8, my + 2.8, cap, c.font("mono", 2.7), (236, 246, 251))
    p = my + IMG_H + 3.0
    p = c.para(x + tx, p, col_w - 2 * tx,
               "Every saved frame contributes: poses solved over all overlaps at "
               "once, each pixel taken from the sharpest frame that saw it, one "
               "global gain solve — and clipped lens corners masked, not blended.",
               c.font("body", BD), BODY)
    y = close(x, y, col_w, p + 0.8, "06 result") + GAP

    ty = card(x, y, col_w, "07 — IMPACT", "Faster, repeatable, auditable",
              accent=TEAL)
    p = ty + 3.6
    for big, txt in [
            ("~43 %", "of the captures for the same coverage once the steps are "
                      "closed-loop — 3 columns instead of 6"),
            ("0", "operator decisions mid-sweep: the boundary estimate turns "
                  "the column and ends the scan"),
            ("151", "automated checks over focus, geometry, GUI and stitching, "
                    "with no camera and no display")]:
        c.text(x + tx, p - 0.4, big, c.font("display", 8.2), CYAN)
        bwx = d.textlength(big, font=c.font("display", 8.2)) / c.k
        pend = c.para(x + tx + bwx + 3.2, p + 0.3,
                      col_w - 2 * tx - bwx - 3.2, txt, c.font("body", 3.9), BODY)
        p = max(p + 9.8, pend + 2.0)
    y = close(x, y, col_w, p - 0.4, "07 impact") + GAP

    ty = card(x, y, col_w, "08 — ENGINEERING RULES",
              "What the design refuses to do", accent=AMBER)
    p = ty + 3.2
    for txt in [
            "Live stitching may never delay a motor move: worker thread, "
            "non-blocking submit, self-disabling on any error.",
            "No worker thread touches a widget; every result crosses back "
            "through one queue.",
            "Overlap is measured, never assumed — and an outlier is refused, "
            "not smoothed."]:
        d.line([(c.px(x + tx), c.px(p + 1.6)), (c.px(x + tx + 3.8),
                                                c.px(p + 1.6))],
               fill=AMBER, width=max(1, c.px(0.3)))
        p = c.para(x + tx + 5.0, p, col_w - 2 * tx - 5.0, txt,
                   c.font("body", 3.9), BODY)
        p += 2.8
    col3_bottom = close(x, y, col_w, p - 1.6, "08 rules")

    col_bottom = max(col1_bottom, col2_bottom, col3_bottom)

    # ══ SERPENTINE (full width) ═════════════════════════════════════════════
    # the band sits in the space that is left, a little closer to the credits
    # than to the columns above it
    by = col_bottom + GAP + 4.0
    serp_w = CW * 0.565
    ty = card(X0, by, serp_w, "09 — HOW A COLUMN RUNS",
              "Serpentine, boundary-triggered", pad=5.4)
    p = ty + 2.4
    DH = 62.0
    patch = Image.new("RGB", (c.px(serp_w) * S, int(DH * c.k) * S), PANEL2)
    pd = ImageDraw.Draw(patch)

    def Q(a, b):
        return (a * c.k * S, b * c.k * S)

    ncol, nrow = 5, 4
    cwid, chei = 28.0, 8.6
    ox, oy = 17.0, 4.6
    for ci in range(ncol):
        for ri in range(nrow):
            x0 = ox + ci * cwid
            y0 = oy + ri * chei
            heat = 0.55 + 0.45 * math.sin((ci * nrow + ri) * 0.7) ** 2
            fill = (int(22 * heat) + 9, int(58 * heat) + 12, int(80 * heat) + 16)
            pd.rounded_rectangle([Q(x0 + 3.2, y0 + 0.9),
                                  Q(x0 + cwid - 3.2, y0 + chei - 0.9)],
                                 radius=1.6 * S, fill=fill,
                                 outline=(40, 84, 118), width=S)
        label = (f"column {ci + 1}" if ci < 3 else
                 ("…" if ci == 3 else "last column"))
        pd.text(Q(ox + ci * cwid + cwid / 2, oy + nrow * chei + 1.6), label,
                font=dfont("head", 3.1),
                fill=CYAN if ci < 3 else MUTED, anchor="ma")

    path = []
    for ci in range(ncol):
        ys = [oy + (r + 0.5) * chei for r in range(nrow)]
        if ci % 2:
            ys.reverse()
        path += [(ox + ci * cwid + cwid / 2, v) for v in ys]
    pd.line([Q(a_, b_) for a_, b_ in path], fill=CYAN, width=int(1.4 * S),
            joint="curve")
    for i in range(1, len(path)):
        x1, y1 = path[i - 1]
        x2, y2 = path[i]
        if abs(y1 - y2) < 0.01:
            continue
        ang = math.atan2(y2 - y1, x2 - x1)
        tip = (x2 - 2.2 * math.cos(ang), y2 - 2.2 * math.sin(ang))
        for sgn in (+1, -1):
            pd.line([Q(tip[0] - 1.5 * math.cos(ang + sgn * 0.55),
                       tip[1] - 1.5 * math.sin(ang + sgn * 0.55)), Q(*tip)],
                    fill=CYAN, width=int(1.3 * S))

    # Y step bracket, on the left
    yb = oy + 2.5 * chei
    pd.line([Q(ox - 4.4, yb - chei / 2), Q(ox - 4.4, yb + chei / 2)],
            fill=AMBER, width=int(1.4 * S))
    for sgn in (-1, 1):
        pd.line([Q(ox - 4.4 + sgn * 1.5, yb - chei / 2 + 1.2),
                 Q(ox - 4.4, yb - chei / 2)], fill=AMBER, width=int(1.4 * S))
        pd.line([Q(ox - 4.4 + sgn * 1.5, yb + chei / 2 - 1.2),
                 Q(ox - 4.4, yb + chei / 2)], fill=AMBER, width=int(1.4 * S))
    pd.line([Q(ox - 4.4, yb), Q(ox - 1.2, yb)], fill=AMBER, width=S)
    pd.line([Q(11.4, yb), Q(ox - 5.0, yb)], fill=(60, 62, 40), width=S)
    pd.text(Q(1.2, yb - 2.7), "Y step", font=dfont("head", 3.0), fill=AMBER,
            anchor="lm")
    pd.text(Q(1.2, yb + 0.3), "344 px", font=dfont("mono", 3.0), fill=WHITE,
            anchor="lm")

    # X step bracket between column 1 and 2
    ybase = oy + nrow * chei - 7.2
    pd.line([Q(ox + cwid, ybase), Q(ox + cwid, ybase + 3.0)], fill=AMBER, width=S)
    pd.line([Q(ox + cwid, ybase + 1.5), Q(ox + 2 * cwid, ybase + 1.5)],
            fill=AMBER, width=S)
    pd.line([Q(ox + 2 * cwid, ybase), Q(ox + 2 * cwid, ybase + 3.0)],
            fill=AMBER, width=S)
    pd.text(Q(ox + 1.5 * cwid, ybase + 4.2), "X step 426 px  ->  0.90 mm",
            font=dfont("head", 3.0), fill=AMBER, anchor="ma")

    # legend: one block on the right, one lane under the grid
    ax = ox + ncol * cwid + 6.0
    pd.line([Q(ax - 2.8, oy - 1.0), Q(ax - 2.8, oy + nrow * chei + 1.0)],
            fill=(34, 76, 108), width=S)
    pd.text(Q(ax, oy - 1.4), "BOUNDARY TRIGGERS", font=dfont("head", 3.1),
            fill=TEAL)
    for i, (k, pct, v) in enumerate([("lower", "10 %", "turns the column"),
                                     ("upper", "20 %", "must see the real edge"),
                                     ("right", "10 %", "scan complete")]):
        yb2 = oy + 3.4 + i * 8.0
        pd.text(Q(ax, yb2), k, font=dfont("semibold", 3.0), fill=WHITE)
        pd.text(Q(ax + 10.6, yb2), pct, font=dfont("mono", 3.0), fill=AMBER)
        pd.text(Q(ax, yb2 + 3.3), v, font=dfont("body", 2.75), fill=MUTED)

    af_y = oy + nrow * chei + 6.6
    pd.line([Q(ox - 4.4, af_y - 1.6), Q(ox + ncol * cwid - 4.0, af_y - 1.6)],
            fill=(34, 76, 108), width=S)
    pd.text(Q(ox - 4.4, af_y + 0.6), "AUTOFOCUS", font=dfont("head", 3.1),
            fill=TEAL)
    pd.text(Q(ox + 20.0, af_y + 0.6), "every 4th row, plus both sides of every "
                                      "column turn", font=dfont("body", 2.85),
            fill=MUTED)

    patch = patch.resize((c.px(serp_w), c.px(DH)), Image.LANCZOS)
    c.img.paste(patch, (c.px(X0), c.px(p)))
    p += DH + 3.0
    p = c.para(X0 + 5.4, p, serp_w - 10.8,
               "The sweep never counts rows. It walks until the boundary "
               "estimate says the edge is in view, turns a column, and walks "
               "back the other way — so a longer part simply takes more moves "
               "rather than a re-parameterised job.", c.font("body", 3.9), BODY)
    band1 = close(X0, by, serp_w, p + 1.0, "09 serpentine")

    # ── the two real images, side by side: the mosaic and its source map ────
    ix = X0 + serp_w + COLGAP
    iw = CW - serp_w - COLGAP
    ty = card(ix, by, iw, "10 — THE REAL OUTPUT",
              "A full blade, and who owns each pixel", accent=TEAL, pad=5.0)
    p = ty + 3.0
    two_w = (iw - 3 * 4.4) / 2.0
    ih = 57.0
    for j, (path, cap_t, cap_s) in enumerate([
            (os.path.join(ASSETS, "real_mosaic.jpg"), "stitched mosaic",
             "5 680 × 12 697 px"),
            (os.path.join(ASSETS, "real_sources.png"), "per-pixel source map",
             "one colour per frame")]):
        bxx = ix + 4.4 + j * (two_w + 4.4)
        im, oy = load_fit(path, c.px(two_w), c.px(ih))
        # a dark plate behind, so both images sit on the same ground
        d.rounded_rectangle([c.px(bxx), c.px(p), c.px(bxx + two_w),
                             c.px(p + ih)], radius=c.px(2), fill=(6, 12, 20),
                            outline=BORDER, width=1)
        c.img.paste(im, (c.px(bxx) + (c.px(two_w) - im.size[0]) // 2,
                         c.px(p) + oy),
                    rounded_mask(im.size, c.px(1.6)))
        c.text(bxx, p + ih + 2.2, cap_t, c.font("semibold", 3.6), WHITE)
        c.text(bxx, p + ih + 6.6, cap_s, c.font("mono", 2.9), MUTED)
    p += ih + 13.0
    p = c.para(ix + 5.0, p, iw - 10.0,
               "Left: the deliverable, from every frame of one scan. Right: the "
               "same scan with each pixel coloured by the frame that won it — "
               "red = dead-reckoned, white = seams.", c.font("body", 3.75), BODY)
    band2 = close(ix, by, iw, p, "10 real output")
    band_bottom = max(band1, band2)

    # ══ CREDITS ═════════════════════════════════════════════════════════════
    cred_h = 32.0
    cy0 = min(band_bottom + 8.0, H_MM - 10.0 - cred_h)
    d.rounded_rectangle([c.px(X0), c.px(cy0), c.px(X1), c.px(cy0 + cred_h)],
                        radius=c.px(3), fill=PANEL, outline=BORDER, width=1)
    d.rectangle([c.px(X0), c.px(cy0 + cred_h - 1.6), c.px(X1),
                 c.px(cy0 + cred_h)], fill=CYAN)

    c.text(X0 + 6.0, cy0 + 5.0, "GROUP 62", c.font("head", 4.6), AMBER,
           spacing=0.6)
    c.text(X0 + 6.0, cy0 + 10.6,
           "Abhigit Gope · Ritik Gupta · Juhi Agrawal\nNitu Chaudhary · Anirudh "
           "Berwal", c.font("body", 3.9), WHITE)

    s1 = X0 + 138.0
    d.line([(c.px(s1), c.px(cy0 + 4.4)), (c.px(s1), c.px(cy0 + cred_h - 5.0))],
           fill=BORDER, width=1)
    c.text(s1 + 7.0, cy0 + 5.0, "FACULTY MENTOR", c.font("head", 3.4), TEAL,
           spacing=0.5)
    c.text(s1 + 7.0, cy0 + 10.0, "Dr. Gajendra Singh", c.font("semibold", 4.7),
           WHITE)

    s2 = X0 + 232.0
    d.line([(c.px(s2), c.px(cy0 + 4.4)), (c.px(s2), c.px(cy0 + cred_h - 5.0))],
           fill=BORDER, width=1)
    c.text(s2 + 7.0, cy0 + 5.0, "STUDENT MENTORS", c.font("head", 3.4), TEAL,
           spacing=0.5)
    c.text(s2 + 7.0, cy0 + 10.0, "Jay Shah  ·  Samagra Singhal",
           c.font("semibold", 4.7), WHITE)

    c.text(X1 - 6.0, cy0 + 5.0, "Indian Institute of Technology Mandi",
           c.font("head", 4.2), CYAN, anchor="ra", spacing=0.3)
    c.text(X1 - 6.0, cy0 + 10.6, "BladeVision 3.0  ·  A2  ·  420 × 594 mm",
           c.font("mono", 2.9), MUTED, anchor="ra")

    # ══ FOOTER ══════════════════════════════════════════════════════════════
    fy = min(cy0 + cred_h + 4.0, H_MM - 8.0)
    c.text(X0, fy, "BLADEVISION 3.0", c.font("head", 4.2), WHITE, spacing=0.45)
    c.text(X0 + 46.0, fy + 0.6,
           "Python · OpenCV · NumPy · Tk · ESP32  —  Aniii15/Dp_26",
           c.font("mono", 2.7), MUTED)
    c.text(X1, fy + 0.6, "Every number here is measured on the rig or asserted "
                         "by a test in the repository.",
           c.font("body", 3.1), DIM, anchor="ra")

    c.img.save(out, optimize=True)
    if os.environ.get("POSTER_DEBUG"):
        print(f"col_bottom={col_bottom:.1f} band={by:.1f}-{band_bottom:.1f} "
              f"credits={cy0:.1f}-{cy0 + cred_h:.1f} footer={fy:.1f} "
              f"H={H_MM}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, f"BladeVision_3.0_A2_{a.dpi}dpi.png")
    print(build(a.dpi, out))
