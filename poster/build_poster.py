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
    BAND = 142.0
    hero_w = 150.0
    hero_x = X1 - hero_w
    hero = load_cover(os.path.join(ASSETS, "blade_hero.png"),
                      c.px(hero_w), c.px(BAND), focus_y=0.0)
    hero = fade(hero, "up", NAVY, 210)
    c.img.paste(hero, (c.px(hero_x), 0), rounded_mask(hero.size, c.px(1.5)))
    veil = Image.new("RGBA", (c.px(hero_w), c.px(BAND)), (0, 0, 0, 0))
    vd = ImageDraw.Draw(veil)
    for i in range(c.px(hero_w)):
        a = int(205 * (1 - i / c.px(hero_w)) ** 1.5)
        vd.line([(i, 0), (i, c.px(BAND))], fill=NAVY + (a,))
    cur = c.img.crop((c.px(hero_x), 0, c.px(hero_x + hero_w),
                      c.px(BAND))).convert("RGBA")
    c.img.paste(Image.alpha_composite(cur, veil).convert("RGB"),
                (c.px(hero_x), 0))

    # brand mark
    m = 8.4
    d.rounded_rectangle([c.px(X0), c.px(10), c.px(X0 + m), c.px(10 + m)],
                        radius=c.px(1.8), outline=CYAN, width=max(2, c.px(0.32)))
    d.line([c.px(X0 + 2.1), c.px(10 + m / 2), c.px(X0 + m - 2.1),
            c.px(10 + m / 2)], fill=CYAN, width=max(2, c.px(0.32)))
    d.ellipse([c.px(X0 + m / 2 - 1.05), c.px(10 + m / 2 - 1.05),
               c.px(X0 + m / 2 + 1.05), c.px(10 + m / 2 + 1.05)], fill=CYAN)
    c.text(X0 + m + 4, 9.6, "AUTONOMOUS MICRO-INSPECTION PLATFORM",
           c.font("head", 4.5), CYAN, spacing=0.45)

    title_f = c.font("display", 29)
    c.text(X0, 21.5, "BLADEVISION", title_f, WHITE, spacing=-0.3)
    tw = d.textlength("BLADEVISION", font=title_f) / c.k
    c.text(X0 + tw + 3.0, 20.8, "3.0", c.font("display", 29), CYAN,
           spacing=-0.3)

    d.line([(c.px(X0), c.px(58)), (c.px(X0 + 90), c.px(58))], fill=CYAN,
           width=max(2, c.px(0.5)))
    c.text(X0, 63.5, "From a stepper pulse to a stitched mosaic.",
           c.font("semibold", 8.4), WHITE)
    c.para(X0, 75.5, 220.0,
           "A closed-loop, boundary-guided scanner that photographs a turbine "
           "blade at full coverage and rebuilds it as one continuous, "
           "defect-ready image — with no position sensor on any axis.",
           c.font("body", 4.2), BODY)

    for i, (big, small) in enumerate([
            ("1.8 mm", "field of view\nper 1080p frame"),
            ("68 %", "measured frame\noverlap (target 65)"),
            ("43 %", "of the captures\nof the old fixed steps"),
            ("13.3 k", "lines of Python\nacross 14 modules")]):
        sx = X0 + i * 55.0
        c.text(sx, 99.5, big, c.font("display", 11.2), CYAN)
        c.text(sx, 112.5, small, c.font("body", 3.6), MUTED)

    bw = 40.0
    d.rounded_rectangle([c.px(X1 - bw), c.px(8), c.px(X1), c.px(21)],
                        radius=c.px(1.5), fill=(18, 25, 31), outline=AMBER,
                        width=max(1, c.px(0.25)))
    c.text(X1 - bw / 2, 10.4, "GROUP", c.font("head", 3.3), AMBER, anchor="ma",
           spacing=0.6)
    c.text(X1 - bw / 2, 13.2, "62", c.font("display", 6.2), WHITE, anchor="ma")

    # ══ KPI STRIP ═══════════════════════════════════════════════════════════
    ky = BAND + 4
    kpis = [("X AXIS", "800", "steps / mm"),
            ("Z AXIS", "1600", "steps / mm · DoF 0.09 mm"),
            ("CAPTURE", "1920×1080", "HDMI card, OpenCV"),
            ("VERIFIED BY", "151", "automated checks · 2.2 k lines"),
            ("CODEBASE", "13.3 k", "lines · 14 modules")]
    kw = CW / len(kpis)
    d.line([(c.px(X0), c.px(ky)), (c.px(X1), c.px(ky))], fill=BORDER_HI,
           width=max(1, c.px(0.3)))
    for i, (lab, val, sub) in enumerate(kpis):
        x = X0 + i * kw
        c.text(x, ky + 3.6, lab, c.font("head", 3.7), TEAL, spacing=0.5)
        c.text(x, ky + 7.8, val, c.font("display", 7.4), WHITE)
        c.text(x, ky + 16.4, sub, c.font("body", 3.3), MUTED)
        if i:
            d.line([(c.px(x - 2), c.px(ky + 2.6)),
                    (c.px(x - 2), c.px(ky + 20.6))], fill=BORDER, width=1)

    TOP = ky + 25.0

    # ══ CARD FRAMEWORK ══════════════════════════════════════════════════════
    def card(x, y, w, kicker, title, accent=CYAN, pad=4.4, head=None):
        nlines = title.count("\n") + 1
        if head is None:
            head = 15.0 + (nlines - 1) * 7.6
        d.rounded_rectangle([c.px(x), c.px(y), c.px(x + w), c.px(y + head)],
                            radius=c.px(3), fill=PANEL2)
        d.rounded_rectangle([c.px(x), c.px(y), c.px(x + w), c.px(y + head)],
                            radius=c.px(3), outline=BORDER, width=1)
        c.text(x + pad, y + 2.3, kicker, c.font("head", 3.6), accent,
               spacing=0.45)
        c.text(x + pad, y + 6.4, title, c.font("head", 7.0), WHITE,
               leading=1.02)
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

    COLGAP = 5.0
    col_w = (CW - 2 * COLGAP) / 3.0
    cx = [X0 + i * (col_w + COLGAP) for i in range(3)]
    GAP = 5.6

    # ══ COLUMN 1 ════════════════════════════════════════════════════════════
    x, y = cx[0], TOP
    ty = card(x, y, col_w, "01 — THE PROBLEM",
              "Microns of damage, hundreds of\nmillimetres of part")
    p = ty + 2.8
    p = c.para(x + 4.4, p, col_w - 8.8,
               "A turbine blade fails at the leading edge, at the tip and around "
               "its film-cooling holes: defects a few tens of microns across, on "
               "a part several hundred millimetres long. Under the microscope the "
               "field of view is about 1.8 mm, so one blade is thousands of "
               "overlapping photographs.", c.font("body", 3.8), BODY)
    p += 2.2
    for head, txt in [
            ("Coverage", "Manual XY stages make the operator the positioning "
                         "system: slow, and quietly different every time."),
            ("Focus", "Depth of field is 0.09 mm on a curved, tilted surface — "
                      "most frames are partly blurred, and a frame-wide "
                      "sharpness score peaks where nothing is sharp."),
            ("Evidence", "Nothing is recorded, so no two inspections can be "
                         "compared or audited.")]:
        d.ellipse([c.px(x + 4.6), c.px(p + 1.2), c.px(x + 6.0), c.px(p + 2.6)],
                  fill=CYAN)
        c.text(x + 7.4, p, head, c.font("semibold", 3.8), WHITE)
        p = c.para(x + 7.4, p + 4.0, col_w - 12.0, txt, c.font("body", 3.6), MUTED)
        p += 1.4
    p += 0.4
    c.chip(x + 4.4, p, "3 axes + 1 actuator", c.font("medium", 3.1), CYAN,
           bg=(14, 44, 60), border=(30, 92, 108))
    c.chip(x + 4.4 + 35.0, p, "no encoders", c.font("medium", 3.1), MUTED,
           border=BORDER)
    c.chip(x + 4.4 + 58.5, p, "operator-bound", c.font("medium", 3.1), MUTED,
           border=BORDER)
    y = close(x, y, col_w, p + 6.6, "01 problem") + GAP

    ty = card(x, y, col_w, "02 — THE PIPELINE", "Five stages, one closed loop",
              accent=TEAL)
    p = ty + 2.8
    for i, (name, txt) in enumerate([
            ("MOTION", "ESP32 over Wi-Fi: X 800 steps/mm, Z 1600, Y on a BTS7960 "
                       "PWM actuator. A move ends when the camera stops changing "
                       "— never on a timer."),
            ("CAPTURE", "One 1080p HDMI frame per stop, saved as "
                        "colNN_phase_HHMMSS.jpg and appended to capture_log.csv "
                        "so the true order survives same-second names."),
            ("FOCUS", "Tile-based autofocus: 200-step coarse walk, bracket the "
                      "peak, 50-step refine, 3-exposure stack. The range cap is "
                      "a hard bound (0 = full travel)."),
            ("SWEEP + LIVE STITCH", "Boundary-guided serpentine; autofocus every "
                                    "4th row. A worker thread stitches each frame "
                                    "as it lands and never delays a move."),
            ("RECONSTRUCT", "Global pose-graph re-stitch of every frame, "
                            "sharpest-pixel blending, exposure solve, lens mask "
                            "— one continuous mosaic plus a testable report.")],
            1):
        d.rounded_rectangle([c.px(x + 4.2), c.px(p), c.px(x + 9.8),
                             c.px(p + 5.2)], radius=c.px(1.1), fill=TEAL)
        c.text(x + 7.0, p + 0.62, str(i), c.font("display", 3.2), NAVY,
               anchor="ma")
        c.text(x + 12.0, p + 0.3, name, c.font("head", 4.4), WHITE, spacing=0.4)
        p = c.para(x + 12.0, p + 5.7, col_w - 16.4, txt, c.font("body", 3.58),
                   BODY)
        p += 2.2
    col1_bottom = close(x, y, col_w, p - 1.6, "02 pipeline")
    y = col1_bottom + GAP

    # ══ COLUMN 2 ════════════════════════════════════════════════════════════
    x, y = cx[1], TOP
    ty = card(x, y, col_w, "03 — THE SOFTWARE", "Fourteen modules, four jobs")
    p = ty + 2.8
    for name, size, txt in [
            ("main3.py", "2 965", "Operator GUI (five tabs), ESP32 HTTP layer, "
                                  "sweep state machine, headless-testable."),
            ("focus_engine.py", "1 301", "Tiles, Tenengrad scores, bracketing, "
                                         "stacking, trend prediction, weights."),
            ("scan_geometry.py", "530", "Learns what the motors really do from "
                                        "the stitcher's registration; writes "
                                        "geometry.json per scan."),
            ("live/global_stitch.py", "1 261", "Pose graph (IRLS), per-move "
                                               "motion model, Potts-MRF "
                                               "sharpest-pixel blend."),
            ("live/pipeline.py", "766", "Streaming stitcher: a decision per "
                                        "frame, blended into a preview."),
            ("live/registration.py", "311", "Phase correlation on high-passed "
                                            "frames — no features required."),
            ("live/lens.py", "179", "Radial validity mask + flat field, fitted "
                                    "from the scan itself."),
            ("live/quality.py", "132", "Sharpness, directional blur, dead-frame "
                                       "detection.")]:
        c.text(x + 4.4, p, name, c.font("mono", 3.3), CYAN)
        nw = d.textlength(name, font=c.font("mono", 3.3)) / c.k
        c.text(x + 4.4 + nw + 2.0, p + 0.5, size, c.font("medium", 2.9), DIM)
        p = c.para(x + 4.4, p + 4.4, col_w - 8.8, txt, c.font("body", 3.45), BODY)
        p += 1.6
    y = close(x, y, col_w, p - 1.2, "03 software") + GAP

    # ── data flow diagram ──
    ty = card(x, y, col_w, "04 — DATA FLOW", "One capture, two loops",
              accent=TEAL)
    p = ty + 2.0
    DIA_H = 71.0
    S = 3
    patch = Image.new("RGB", (int(col_w * c.k) * S, int(DIA_H * c.k) * S), PANEL2)

    def dfont(kind, mm):
        return c.font(kind, mm, S)
    pd = ImageDraw.Draw(patch)

    def P(a, b):
        return (a * c.k * S, b * c.k * S)

    def box(a, b, w, h, label, sub, col=CYAN, fill=(15, 42, 62), font_mm=3.15):
        pd.rounded_rectangle([P(a, b), P(a + w, b + h)], radius=2.5 * S,
                             fill=fill, outline=col, width=max(1, S - 1))
        pd.text(P(a + w / 2, b + 1.7), label, font=dfont("semibold", font_mm),
                fill=WHITE, anchor="ma")
        pd.text(P(a + w / 2, b + h - 2.9), sub, font=dfont("body", 2.45),
                fill=MUTED, anchor="ma")

    def arrow(a1, b1, a2, b2, col=BORDER_HI):
        pd.line([P(a1, b1), P(a2, b2)], fill=col, width=max(1, S - 1))
        ang = math.atan2(b2 - b1, a2 - a1)
        for sgn in (+1, -1):
            pd.line([P(a2, b2), P(a2 - 1.9 * math.cos(ang + sgn * 0.5),
                                  b2 - 1.9 * math.sin(ang + sgn * 0.5))],
                    fill=col, width=max(1, S - 1))

    LX, RX, BW = 2.8, 31.0, 25.0
    box(LX, 1.4, BW, 9.2, "MOTOR MOVE", "ESP32 / HTTP GET")
    box(RX, 1.4, BW, 9.2, "FRAME", "1080p, saved + logged")
    arrow(LX + BW + 0.6, 6.0, RX - 0.6, 6.0)

    box(LX, 14.8, BW, 9.2, "LIVE STITCHER", "Decision: dx, dy, ncc", TEAL)
    box(RX, 14.8, BW, 9.2, "PREVIEW MOSAIC", "grown, half scale", TEAL)
    arrow(LX + BW / 2, 10.9, LX + BW / 2, 14.4)
    arrow(LX + BW + 0.6, 19.4, RX - 0.6, 19.4)

    box(LX, 28.2, BW, 9.2, "GEOMETRY", "px/mm, real overlap", AMBER,
        fill=AMBER_PANEL)
    box(RX, 28.2, BW, 9.2, "NEXT MOVE", "Y pulse / X step", AMBER,
        fill=AMBER_PANEL)
    arrow(LX + BW / 2, 24.3, LX + BW / 2, 27.8)
    arrow(LX + BW + 0.6, 32.8, RX - 0.6, 32.8)

    # feedback lane (clear of every box)
    lane = 40.0
    pd.line([P(RX + BW / 2, 37.7), P(RX + BW / 2, lane), P(LX - 1.5, lane)],
            fill=AMBER, width=max(1, S - 1))
    pd.line([P(LX - 1.5, lane), P(LX - 1.5, 2.2)], fill=AMBER,
            width=max(1, S - 1))
    pd.polygon([P(LX - 1.5, 0.6), P(LX - 2.5, 2.6), P(LX - 0.5, 2.6)], fill=AMBER)
    pd.text(P((LX + RX + BW) / 2, lane + 1.5), "measured, never guessed",
            font=dfont("medium", 2.7), fill=AMBER, anchor="ma")

    box(LX, 47.0, BW, 9.2, "SCAN FOLDER", "frames + log", MUTED,
        fill=(13, 34, 52))
    box(RX, 47.0, BW, 9.2, "GLOBAL SOLVE", "all frames, IRLS", CYAN,
        fill=(13, 34, 52))
    arrow(LX + BW + 0.6, 51.6, RX - 0.6, 51.6)
    pd.line([P(RX + BW / 2, 56.6), P(RX + BW / 2, 60.2), P(LX + BW / 2, 60.2),
             P(LX + BW / 2, 62.0)], fill=CYAN, width=max(1, S - 1))
    pd.polygon([P(LX + BW / 2, 63.6), P(LX + BW / 2 - 1.0, 61.6),
                P(LX + BW / 2 + 1.0, 61.6)], fill=CYAN)
    pd.rounded_rectangle([P(LX, 64.2), P(LX + BW, 69.8)], radius=2.5 * S,
                         fill=(20, 40, 46), outline=AMBER, width=max(1, S - 1))
    pd.text(P(LX + BW / 2, 65.7), "DELIVERABLE MOSAIC",
            font=dfont("head", 3.2), fill=WHITE, anchor="ma")
    pd.text(P(LX + BW / 2, 68.4), "+ report, poses, tiles",
            font=dfont("body", 2.55), fill=MUTED, anchor="ma")

    patch = patch.resize((c.px(col_w), c.px(DIA_H)), Image.LANCZOS)
    c.img.paste(patch, (c.px(x), c.px(p)))
    p += DIA_H + 0.8
    y = close(x, y, col_w, p, "04 data flow") + GAP

    ty = card(x, y, col_w, "05 — MEASURED, NOT GUESSED",
              "What the commands actually do", accent=AMBER)
    p = ty + 2.8
    rows = [("AXIS", "COMMANDED", "MEASURED", "OVERLAP"),
            ("Y", "1200 ms pulse", "344 px / 1080", "68 %"),
            ("X", "0.40 mm = 320 pulses", "426 px / 1920", "78 %"),
            ("SCALE", "800 steps / mm", "1064 px / mm", "1.80 mm")]
    cw = [col_w * 0.125, col_w * 0.375, col_w * 0.285, col_w * 0.215]
    for r, row in enumerate(rows):
        ry = p + r * 5.1
        if r == 0:
            for i, cell in enumerate(row):
                c.text(x + 3.2 + sum(cw[:i]), ry, cell, c.font("head", 3.1),
                       TEAL, spacing=0.25)
            d.line([(c.px(x + 3.2), c.px(ry + 4.3)),
                    (c.px(x + col_w - 3.2), c.px(ry + 4.3))], fill=BORDER, width=1)
        else:
            for i, cell in enumerate(row):
                col = AMBER if i == 3 else (WHITE if i == 0 else BODY)
                fn = c.font("semibold", 3.35) if i == 0 else c.font("body", 3.25)
                c.text(x + 3.2 + sum(cw[:i]), ry, cell, fn, col)
    p += 5.1 * len(rows) + 0.6
    p = c.para(x + 3.2, p, col_w - 6.4,
               "X was four times more conservative than the stitcher needs; the "
               "pixel scale is not portable between scans of the same rig, so "
               "it is now written into every scan folder.",
               c.font("body", 3.3), MUTED)
    col2_bottom = close(x, y, col_w, p + 1.4, "05 measured")

    # ══ COLUMN 3 ════════════════════════════════════════════════════════════
    x, y = cx[2], TOP
    ty = card(x, y, col_w, "06 — THE RESULT", "One image, not a collage")
    p = ty + 2.8
    IMG_H = 41.0
    macro = load_cover(os.path.join(ASSETS, "surface_macro.png"),
                       c.px(col_w - 8.8), c.px(IMG_H), focus_y=0.5)
    mx, my = x + 4.4, p
    c.img.paste(macro, (c.px(mx), c.px(my)),
                rounded_mask(macro.size, c.px(1.4)))
    # call out one hole of the row the mask is meant to catch
    ring_x, ring_y = mx + 44.0, my + 10.6
    d.ellipse([c.px(ring_x - 3.0), c.px(ring_y - 3.0), c.px(ring_x + 3.0),
               c.px(ring_y + 3.0)], outline=AMBER, width=max(2, c.px(0.3)))
    d.line([(c.px(ring_x - 3.2), c.px(ring_y - 1.0)),
            (c.px(mx + 25.0), c.px(my + 3.4))], fill=AMBER, width=1)
    # a plate behind the annotation keeps it legible on the bright metal
    cap = "film-cooling hole row  ·  1 : 1 mosaic tile"
    cap_w = d.textlength(cap, font=c.font("mono", 2.6)) / c.k + 3.0
    d.rounded_rectangle([c.px(mx + 1.0), c.px(my + 1.2),
                         c.px(mx + 1.0 + cap_w), c.px(my + 6.2)],
                        radius=c.px(0.9), fill=(10, 20, 30))
    c.text(mx + 2.4, my + 2.5, cap, c.font("mono", 2.6), (236, 246, 251))
    p = my + IMG_H + 2.2
    p = c.para(x + 4.4, p, col_w - 8.8,
               "Every saved frame contributes: poses are re-solved over all "
               "overlaps at once, each canvas pixel is taken from the sharpest "
               "frame that saw it, and one global gain solve removes the "
               "patchwork. Clipped lens corners are masked out, not blended in.",
               c.font("body", 3.5), BODY)
    y = close(x, y, col_w, p + 1.0, "06 result") + GAP

    ty = card(x, y, col_w, "07 — IMPACT", "Faster, repeatable, auditable",
              accent=TEAL)
    p = ty + 3.0
    for big, txt in [
            ("~43 %", "of the captures for the same coverage once the step sizes "
                      "are closed-loop (3 columns instead of 6 on a 3.6 mm part)"),
            ("0", "operator decisions mid-sweep — the boundary estimate triggers "
                  "every turn and stops the scan at the part edge"),
            ("151", "automated checks over focus, geometry, GUI and stitching "
                    "that run without a camera, motors or display"),
            ("1 file", "per scan: geometry.json records the commands that made "
                       "it, so two scans can finally be compared")]:
        c.text(x + 4.4, p - 0.4, big, c.font("display", 7.8), CYAN)
        bwx = d.textlength(big, font=c.font("display", 7.8)) / c.k
        pend = c.para(x + 4.4 + bwx + 2.8, p + 0.2, col_w - 8.4 - bwx - 2.8, txt,
                      c.font("body", 3.32), BODY)
        p = max(p + 9.0, pend + 1.2)
    y = close(x, y, col_w, p, "07 impact") + GAP

    ty = card(x, y, col_w, "08 — ENGINEERING RULES",
              "What the design refuses to do", accent=AMBER)
    p = ty + 2.8
    for txt in [
            "Live stitching may never delay a motor move: worker thread, "
            "non-blocking submit, self-disabling on any error.",
            "No worker thread ever touches a widget; every result crosses back "
            "through one queue.",
            "No frame is discarded because a preview rejected it — the "
            "deliverable uses every frame saved.",
            "Overlap is measured, never assumed; an outlier is refused, not "
            "smoothed."]:
        d.line([(c.px(x + 4.4), c.px(p + 1.4)), (c.px(x + 8.0), c.px(p + 1.4))],
               fill=AMBER, width=max(1, c.px(0.3)))
        p = c.para(x + 9.4, p, col_w - 13.6, txt, c.font("body", 3.55), BODY)
        p += 2.0
    y = close(x, y, col_w, p - 1.2, "08 rules") + GAP

    ty = card(x, y, col_w, "09 — NEXT", "Where this goes", accent=AMBER)
    p = ty + 2.8
    for txt in [
            "Defect classifier over mosaic tiles — the stitcher already emits "
            "the per-tile sharpness and quality maps it needs.",
            "A position sensor on the Y actuator: the 55–81 % overlap band is "
            "mechanical, and software can only size the mean.",
            "Stride-2 rows, verified per scan rather than hardcoded."]:
        d.rectangle([c.px(x + 4.4), c.px(p + 1.1), c.px(x + 6.0),
                     c.px(p + 2.7)], fill=AMBER)
        p = c.para(x + 7.8, p, col_w - 12.4, txt, c.font("body", 3.45), BODY)
        p += 1.7
    col3_bottom = close(x, y, col_w, p - 1.0, "09 next")

    col_bottom = max(col1_bottom, col2_bottom, col3_bottom)

    # ══ BOTTOM BAND ═════════════════════════════════════════════════════════
    by = col_bottom + 6.0
    wide_w = CW * 0.575
    nx = X0 + wide_w + COLGAP
    nw = CW - wide_w - COLGAP

    ty = card(X0, by, wide_w, "10 — HOW A COLUMN RUNS",
              "Serpentine, boundary-triggered", pad=4.6)
    p = ty + 2.0
    DH = 76.0
    patch = Image.new("RGB", (c.px(wide_w) * S, c.px(DH) * S), PANEL2)
    pd = ImageDraw.Draw(patch)

    def Q(a, b):
        return (a * c.k * S, b * c.k * S)

    ncol, nrow = 4, 6
    cwid, chei = 30.0, 8.2
    ox, oy = 17.0, 6.0
    # one rounded cell per capture, on a serpentine path
    for ci in range(ncol):
        for ri in range(nrow):
            x0 = ox + ci * cwid
            y0 = oy + ri * chei
            heat = 0.55 + 0.45 * math.sin((ci * nrow + ri) * 0.7) ** 2
            fill = (int(22 * heat) + 9, int(58 * heat) + 12, int(80 * heat) + 16)
            pd.rounded_rectangle([Q(x0 + 3.4, y0 + 0.9),
                                  Q(x0 + cwid - 3.4, y0 + chei - 0.9)],
                                 radius=1.6 * S, fill=fill,
                                 outline=(40, 84, 118), width=S)
        pd.text(Q(ox + ci * cwid + cwid / 2, oy + nrow * chei + 1.6),
                f"column {ci + 1}", font=dfont("head", 3.1),
                fill=CYAN if ci == 0 else MUTED, anchor="ma")

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

    # Y step bracket (left)
    yb = oy + 2.5 * chei
    pd.line([Q(ox - 4.0, yb - chei / 2), Q(ox - 4.0, yb + chei / 2)],
            fill=AMBER, width=int(1.4 * S))
    for sgn in (-1, 1):
        pd.line([Q(ox - 4.0 + sgn * 1.5, yb - chei / 2 + 1.2),
                 Q(ox - 4.0, yb - chei / 2)], fill=AMBER, width=int(1.4 * S))
        pd.line([Q(ox - 4.0 + sgn * 1.5, yb + chei / 2 - 1.2),
                 Q(ox - 4.0, yb + chei / 2)], fill=AMBER, width=int(1.4 * S))
    pd.line([Q(ox - 4.0, yb), Q(ox - 1.2, yb)], fill=AMBER, width=S)
    pd.line([Q(8.6, yb), Q(ox - 4.6, yb)], fill=(60, 62, 40), width=S)
    pd.text(Q(1.0, yb - 2.6), "Y step", font=dfont("head", 3.0), fill=AMBER,
            anchor="lm")
    pd.text(Q(1.0, yb + 0.4), "344 px", font=dfont("mono", 3.0), fill=WHITE,
            anchor="lm")

    # X step bracket between column 1 and 2 (on the seam between cells)
    ybase = oy + nrow * chei - 6.2
    pd.line([Q(ox + cwid, ybase), Q(ox + cwid, ybase + 3.0)], fill=AMBER, width=S)
    pd.line([Q(ox + cwid, ybase + 1.5), Q(ox + 2 * cwid, ybase + 1.5)],
            fill=AMBER, width=S)
    pd.line([Q(ox + 2 * cwid, ybase), Q(ox + 2 * cwid, ybase + 3.0)],
            fill=AMBER, width=S)
    pd.text(Q(ox + 1.5 * cwid, ybase + 4.2), "X step 426 px  ->  0.90 mm",
            font=dfont("head", 3.1), fill=AMBER, anchor="ma")

    # legend column at the right
    ax = ox + ncol * cwid + 5.0
    pd.line([Q(ax - 2.4, oy), Q(ax - 2.4, oy + nrow * chei)],
            fill=(34, 76, 108), width=S)
    pd.text(Q(ax, oy - 0.4), "BOUNDARY TRIGGERS", font=dfont("head", 3.0),
            fill=TEAL)
    for i, (k, pct, v) in enumerate([("lower", "10 %", "turns the column"),
                                     ("upper", "20 %", "must see the real edge"),
                                     ("right", "10 %", "scan complete")]):
        yb2 = oy + 4.6 + i * 8.6
        pd.text(Q(ax, yb2), k, font=dfont("semibold", 3.0), fill=WHITE)
        pd.text(Q(ax + 10.6, yb2), pct, font=dfont("mono", 3.0), fill=AMBER)
        pd.text(Q(ax, yb2 + 3.4), v, font=dfont("body", 2.8), fill=MUTED)
    pd.text(Q(ax, oy + 32.0), "AUTOFOCUS", font=dfont("head", 3.0), fill=TEAL)
    pd.text(Q(ax, oy + 36.0), "every 4th row, and both sides",
            font=dfont("body", 2.8), fill=MUTED)
    pd.text(Q(ax, oy + 39.2), "of every column turn", font=dfont("body", 2.8),
            fill=MUTED)
    pd.text(Q(ax, oy + 45.0), "NEVER COUNTS ROWS", font=dfont("head", 3.0),
            fill=TEAL)
    pd.text(Q(ax, oy + 49.0), "a longer part simply takes",
            font=dfont("body", 2.8), fill=MUTED)
    pd.text(Q(ax, oy + 52.2), "more moves, not a new job",
            font=dfont("body", 2.8), fill=MUTED)

    patch = patch.resize((c.px(wide_w), c.px(DH)), Image.LANCZOS)
    c.img.paste(patch, (c.px(X0), c.px(p)))
    p += DH + 2.2
    p = c.para(X0 + 4.6, p, wide_w - 9.2,
               "The sweep never counts rows. It walks until the boundary estimate "
               "says the edge is in view, turns a column, and walks back the "
               "other way — so a shorter or longer part costs proportionally "
               "less or more time instead of a re-parameterised job.",
               c.font("body", 3.45), BODY)
    band1 = close(X0, by, wide_w, p, "10 serpentine")

    ty = card(nx, by, nw, "11 — THE RIG", "Commodity parts, no encoders",
              accent=TEAL, pad=4.0)
    p = ty + 2.6
    for name, txt in [("ESP32", "Wi-Fi HTTP; 3 steppers + 1 PWM actuator"),
                      ("HDMI capture", "1920×1080 into OpenCV at index 0"),
                      ("Optics", "microscope lens · 1.8 mm field · 0.09 mm DoF"),
                      ("Stage", "X 800 / Z 1600 steps per mm, Y actuator"),
                      ("Host", "Python 3.11 · Tk · OpenCV · NumPy")]:
        c.text(nx + 4.0, p, name, c.font("semibold", 3.45), WHITE)
        p = c.para(nx + 4.0, p + 3.8, nw - 8.0, txt, c.font("body", 3.25), MUTED)
        p += 1.3
    p += 1.4
    c.text(nx + 4.0, p, "TO RUN IT", c.font("head", 3.3), AMBER, spacing=0.45)
    p += 4.2
    for i, step in enumerate(["Flash the ESP32 firmware", "python main3.py",
                              "Start sweep -> read the mosaic"], 1):
        d.rounded_rectangle([c.px(nx + 4.0), c.px(p), c.px(nx + 7.4),
                             c.px(p + 3.4)], radius=c.px(0.8), fill=AMBER)
        c.text(nx + 5.7, p + 0.35, str(i), c.font("display", 2.5), NAVY,
               anchor="ma")
        c.text(nx + 9.4, p + 0.5, step, c.font("body", 3.25), BODY)
        p += 4.8
    band2 = close(nx, by, nw, p - 0.6, "11 rig")

    band_bottom = max(band1, band2)
    for bx, bwid in ((X0, wide_w), (nx, nw)):
        d.rounded_rectangle([c.px(bx), c.px(by), c.px(bx + bwid),
                             c.px(band_bottom)], radius=c.px(3), outline=BORDER,
                            width=1)

    # ══ CREDITS ═════════════════════════════════════════════════════════════
    cred_h = 32.0
    cy0 = max(band_bottom + 6.0, H_MM - 12.0 - cred_h)
    d.rounded_rectangle([c.px(X0), c.px(cy0), c.px(X1), c.px(cy0 + cred_h)],
                        radius=c.px(3), fill=PANEL, outline=BORDER, width=1)
    d.rectangle([c.px(X0), c.px(cy0 + cred_h - 1.6), c.px(X1),
                 c.px(cy0 + cred_h)], fill=CYAN)

    c.text(X0 + 6.0, cy0 + 4.6, "GROUP 62", c.font("head", 4.4), AMBER,
           spacing=0.6)
    c.text(X0 + 6.0, cy0 + 10.0,
           "Abhigit Gope · Ritik Gupta · Juhi Agrawal\nNtu Chaudhary · Anirudh "
           "Berwal", c.font("body", 3.7), WHITE)

    s1 = X0 + 138.0
    d.line([(c.px(s1), c.px(cy0 + 4.0)), (c.px(s1), c.px(cy0 + cred_h - 4.6))],
           fill=BORDER, width=1)
    c.text(s1 + 7.0, cy0 + 4.6, "FACULTY MENTOR", c.font("head", 3.3), TEAL,
           spacing=0.5)
    c.text(s1 + 7.0, cy0 + 9.4, "Dr. Gajendra Singh", c.font("semibold", 4.6),
           WHITE)

    s2 = X0 + 232.0
    d.line([(c.px(s2), c.px(cy0 + 4.0)), (c.px(s2), c.px(cy0 + cred_h - 4.6))],
           fill=BORDER, width=1)
    c.text(s2 + 7.0, cy0 + 4.6, "STUDENT MENTORS", c.font("head", 3.3), TEAL,
           spacing=0.5)
    c.text(s2 + 7.0, cy0 + 9.4, "Jay Shah  ·  Samagra Singhal",
           c.font("semibold", 4.6), WHITE)

    c.text(X1 - 6.0, cy0 + 4.6, "Precision. Detection. Reliability.",
           c.font("head", 4.0), CYAN, anchor="ra", spacing=0.4)
    c.text(X1 - 6.0, cy0 + 10.0, "BladeVision 3.0  ·  A2  ·  420 × 594 mm",
           c.font("mono", 2.8), MUTED, anchor="ra")

    # ══ FOOTER ══════════════════════════════════════════════════════════════
    fy = min(cy0 + cred_h + 3.6, H_MM - 8.0)
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
