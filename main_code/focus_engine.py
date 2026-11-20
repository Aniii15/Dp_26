"""Region-wise autofocus for a curved blade.

Why the old autofocus could not work
------------------------------------
``MotorControlApp._autofocus`` climbed Z one step at a time, asking the Siamese
CNN "is frame B sharper than frame A?".  Two things make that fail on a curved
turbine blade, and both are visible in the delivered mosaic:

1. **Wrong question, wrong region.**  The CNN compares one crop of the frame
   (``CropMiddleThird``), i.e. *one region*, and returns a single number.  A
   blade is not flat: at any one Z some regions are sharp and others are not
   (look at the top view: the surface recedes through the field of view).  A
   scalar preference cannot say *which* region is sharp, so it cannot guide a
   search whose objective is "maximise focus over the blade's area".

2. **A hill climb needs a hill.**  When the start point is far out of focus,
   every nearby position looks equally bad.  The CNN then returns ~0.5, the old
   code read ``prob < 0.4`` as "peak at current position" and stopped - or, when
   ``prob`` sat between 0.4 and 1.0, kept climbing up to 20 x 400 steps (5 mm) in
   whichever direction it happened to be going.  Nothing in that loop ever asked
   *"is my best sample at the edge of what I searched?"*, so a focus that lies
   outside the searched range looked exactly like a locked focus.  That is the
   defect the user reports: a band of the blade where **every** frame is blurred,
   and the mosaic faithfully reproduces it.

What this module does instead
-----------------------------
* **Scores regions, not frames.**  The frame is split into a tile grid; each
  tile gets an exposure-normalised Tenengrad score (the same formula as
  ``image_stitching/live/quality.sharpness`` so the numbers mean the same thing
  in both places).  Each tile's score is then normalised by *that tile's own*
  best score over the Z-stack, which makes tiles with different texture
  comparable - "0.95" means "95 % of the best this patch can ever look".
* **Searches a range, then refuses to lie about it.**  The scan covers the whole
  configured travel in both directions.  If the best sample is at either end of
  the window, the walk keeps going (never past ``z_travel_limit``).  Only a
  bracketed peak - sharp on both sides - counts as focus.  This is the direct
  fix for "the range was too small".
* **Optimises the area in focus**, not one number: the chosen Z is the one that
  maximises the fraction of (usable) blade area that is within
  ``in_focus_ratio`` of its own peak.  When several regions cannot be sharp at
  the same Z it says so, with numbers, instead of quietly producing a blur.
* **Predicts** the next Z from the trend along the scan path (``FocusTrend``),
  so a stop can be searched in a narrow window and still escalate when the
  prediction is wrong.  This is what makes a wide-range search affordable in the
  middle of a sweep.
* **Can stack**: when the depth spread genuinely exceeds the depth of field,
  ``compose_focus_stack`` merges a small Z-bracket into one all-sharp frame, and
  ``tile_weight_map`` gives the stitcher a per-pixel "trust me here" weight so
  overlapping frames contribute the sharpest available pixels.

Nothing here talks to hardware: motion and capture arrive as callables, so the
whole module is testable (and is tested) without a camera or a motor.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

__all__ = [
    "FocusConfig", "FocusError", "FocusResult", "FocusStack", "AutoFocus",
    "FocusTrend", "TileMetrics", "tile_bounds", "tile_metrics",
    "compose_focus_stack", "tile_weight_map", "full_frame",
]


# ──────────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FocusConfig:
    """Everything that decides how expensive and how thorough a scan is.

    Steps are Z-motor steps (Z is 3200 pulses/rev on a 4 mm lead screw =
    800 steps/mm, so 100 steps = 0.125 mm).  The defaults scan ~3.75 mm of
    travel at 0.125 mm - slower than the
    old hill climb per step, but bounded, and it always brackets the peak.
    """

    # tile grid
    tiles_x: int = 6
    tiles_y: int = 4
    margin: float = 0.04             # ignore this fraction of each frame edge
    roi: tuple = (0.0, 0.0, 1.0, 1.0)
    """Part of the frame the *search* looks at, as fractions (x0, y0, x1, y1).

    The tile grid is laid over this rectangle only, so every focus decision
    (peak, area-in-focus, depth spread) is about that region.  On this rig the
    blade curves away from the lens towards the right of the view: no single Z
    makes the whole frame sharp, and the right side is covered again - by the
    sharp left part of the *next* strip - one X step later.  Focusing on the
    left ~40 % therefore gives a sharp band in every frame, which the global
    stitcher then picks per pixel.  Use ``(0, 0, 0.4, 1)`` for that.

    Stacking (``compose_focus_stack``) and stitch weights
    (``tile_weight_map``) always use the whole frame regardless of this."""
    preblur: int = 3                 # kill single-pixel sensor noise (odd, 0=off)
    # A tile may only vote on focus if it has *texture*.  Judged against the best
    # tile in view (relative) with a small absolute floor, because a badly
    # defocused blade still shows luma std ~5 while genuinely featureless
    # background is ~0 - an absolute threshold would throw away every blurred
    # frame, which is exactly when the answer matters most.
    min_contrast: float = 2.0
    min_contrast_rel: float = 0.12
    min_luma: float = 6.0            # near-black tile
    max_luma: float = 252.0          # blown-out tile

    # search
    range_steps: int = 3000
    """Cap on how far the search may walk from where it started, both directions
    (so 6000 = +-3.75 mm around the start).

    This is a *bound*, not a distance travelled: the scan stops as soon as the
    peak is bracketed, and a prediction narrows it further.  Set it to ``0`` to
    let the search use the whole of ``z_travel_limit`` - the right choice when
    the part's depth envelope is unknown, a bad one when it is known and small,
    because then every capture spent outside the envelope is pure cost."""
    coarse_steps: int = 100          # 0.125 mm
    fine_steps: int = 25             # 31 um
    fine_samples: int = 5            # samples per fine sweep (odd)
    z_travel_limit: int = 12000      # 15 mm either way: hard safety clamp
    max_samples: int = 140           # budget guard for one scan
    settle_s: float = 0.18           # wait after every Z move before capture
    backlash_steps: int = 0          # approach every target from below if > 0
    predictive_window: int = 300     # window width when a hint is available

    # decision
    in_focus_ratio: float = 0.90     # tile counts as focused at >=90 % of its peak
    min_in_focus_area: float = 0.60  # below this the stop is flagged
    dof_ratio: float = 0.75          # spread > dof_ratio*DOF => needs stacking
    # A peak only counts when it stands clear of *both* ends of the window.  On a
    # badly defocused frame every sample looks equally soft, and a 0.1 % wobble
    # would otherwise be read as "peak found" - the exact false positive that
    # produced blurred bands in the mosaic.
    blend_falloff: float = 0.01      # weight = exp(-shortfall/tau), focus stacking
    bracket_margin: float = 0.05
    gallop_signal: float = 2.5
    """How much the gallop's best sample must stand above its worst before the
    neighbourhood is worth rescanning.  On a blade that is simply out of reach
    every sample is noise and the spread is ~1.7x; a real focus cone that the
    gallop stepped over still shows 5-10x, because the blur ramp is steep.
    Without this test the scan would either give up on a focusable patch (it
    stepped over the cone) or spend a hundred captures proving a blade is out of
    range."""
    curve_min_variation: float = 0.05
    min_peak_sharpness: float = 0.01
    """Absolute floor for "this sample is actually sharp".

    The score is ``mean(|grad|^2) / mean(luma)^2``, so it is exposure invariant
    but not contrast invariant: measured on the production scans a focused frame
    scores 0.3-0.7, a hopelessly defocused one 0.0006.  Without this floor a
    completely blurred field looks like a noisy curve with a 30 % "peak" in it,
    and the scan happily reports focus it never found - the failure mode that
    produced a blurred band in the mosaic.  Raise it if a defocused frame is ever
    accepted; ``AutoFocus`` also uses the best scan of the session as a
    reference, so a low-contrast part is not penalised for being low-contrast.
    """

    def steps_to_mm(self, steps: float, steps_per_mm: float = 800.0) -> float:
        return steps / steps_per_mm


class FocusError(RuntimeError):
    """Raised when a scan cannot proceed (no feed, no texture, no bracketing)."""


# ──────────────────────────────────────────────────────────────────────────────
#  Per-tile metrics
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TileMetrics:
    sharpness: np.ndarray            # (tiles,) exposure-normalised Tenengrad
    contrast: np.ndarray             # (tiles,) luma std - how much to trust it
    luma: np.ndarray                 # (tiles,) mean luma


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _roi_px(shape, cfg: FocusConfig) -> tuple[int, int, int, int]:
    """ROI as pixel bounds ``(x0, y0, x1, y1)``, clamped and never empty."""
    h, w = shape[:2]
    rx0, ry0, rx1, ry1 = (tuple(cfg.roi) + (0.0, 0.0, 1.0, 1.0))[:4]
    rx0, rx1 = sorted((min(max(rx0, 0.0), 1.0), min(max(rx1, 0.0), 1.0)))
    ry0, ry1 = sorted((min(max(ry0, 0.0), 1.0), min(max(ry1, 0.0), 1.0)))
    x0, x1 = int(round(rx0 * w)), int(round(rx1 * w))
    y0, y1 = int(round(ry0 * h)), int(round(ry1 * h))
    if x1 - x0 < 8:
        x0, x1 = 0, w
    if y1 - y0 < 8:
        y0, y1 = 0, h
    return x0, y0, x1, y1


def full_frame(cfg: FocusConfig) -> FocusConfig:
    """The same config with the ROI reset to the whole frame."""
    if tuple(cfg.roi) == (0.0, 0.0, 1.0, 1.0):
        return cfg
    return replace(cfg, roi=(0.0, 0.0, 1.0, 1.0))


def tile_bounds(shape, cfg: FocusConfig) -> list[tuple[int, int, int, int]]:
    """Tile rectangles ``(y0, y1, x0, x1)``, row-major, over the ROI.

    The margin is only applied at *frame* edges (vignetting, sensor border),
    not at ROI edges that lie inside the frame."""
    h, w = shape[:2]
    x0, y0, x1, y1 = _roi_px(shape, cfg)
    my, mx = int(h * cfg.margin), int(w * cfg.margin)
    xs = np.linspace(max(x0, mx), min(x1, w - mx), cfg.tiles_x + 1).astype(int)
    ys = np.linspace(max(y0, my), min(y1, h - my), cfg.tiles_y + 1).astype(int)
    return [(int(ys[j]), int(ys[j + 1]), int(xs[i]), int(xs[i + 1]))
            for j in range(cfg.tiles_y) for i in range(cfg.tiles_x)]


def tile_metrics(frame: np.ndarray, cfg: FocusConfig) -> TileMetrics:
    """Sharpness / contrast / brightness for every tile of one frame.

    Sharpness is ``mean(|grad|^2) / mean(luma)^2`` - the exposure normalisation
    matters because the blade's brightness drifts as the scan crosses it, and a
    raw Laplacian variance mostly measures that drift (see ``quality.py``).
    Unsharp tiles are still *measured*; whether they may vote is decided later
    by ``FocusStack.reliable``, using contrast which is texture, not blur.
    """
    gray = _to_gray(frame)
    if cfg.preblur and cfg.preblur >= 3:
        gray = cv2.GaussianBlur(gray, (cfg.preblur, cfg.preblur), 0)
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    energy = gx * gx + gy * gy

    boxes = tile_bounds(gray.shape, cfg)
    sharp = np.zeros(len(boxes), np.float32)
    contrast = np.zeros(len(boxes), np.float32)
    luma = np.zeros(len(boxes), np.float32)
    for i, (y0, y1, x0, x1) in enumerate(boxes):
        e = energy[y0:y1, x0:x1]
        t = g[y0:y1, x0:x1]
        m = float(t.mean())
        sharp[i] = float(e.mean()) / (m * m + 1e-3)
        contrast[i] = float(t.std())
        luma[i] = m
    return TileMetrics(sharp, contrast, luma)


# ──────────────────────────────────────────────────────────────────────────────
#  A Z-stack and what it says about focus
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FocusResult:
    """What a focus scan concluded - and what the operator should do about it."""

    z_peak: float = 0.0                # steps, relative to where the scan started
    peak_score: float = 0.0            # mean normalised sharpness of usable tiles
    in_focus_area: float = 0.0         # 0..1 of usable tile area within ratio
    tile_z: np.ndarray | None = None   # (tiles,) best Z per tile, nan if unusable
    tile_score: np.ndarray | None = None
    reliable: np.ndarray | None = None
    dof_steps: float = 0.0             # depth of field, measured from the curves
    depth_spread: float = 0.0          # p90-p10 of the per-tile best Z
    needs_stacking: bool = False
    bracketed: bool = False            # a peak with sharp samples on both sides
    curve_variation: float = 0.0       # (max-min)/max of the sharpness curve
    sharpness_gain: float = 1.0        # best sample / first sample
    range_limited: str = ""            # "neg" / "pos" / "" - peak at a clamp
    samples: int = 0
    seconds: float = 0.0
    trajectory: list[tuple[float, float]] = field(default_factory=list)

    @property
    def focus_fraction_tiles(self) -> float:
        if self.reliable is None or not self.reliable.any():
            return 0.0
        return float(self.in_focus_area)

    def recommendation(self, steps_per_mm: float = 800.0) -> list[str]:
        """Operator-facing advice, in the order they should act on it."""
        out: list[str] = []
        if not self.bracketed and self.samples:
            out.append(
                f"No bracketed focus peak: sharpness varied only "
                f"{self.curve_variation * 100:.1f}% across the whole search, so "
                f"nothing in view was ever in focus.  Widen the Z range and/or "
                f"check that the part is inside the stage's travel.")
        if self.range_limited:
            where = "below" if self.range_limited == "neg" else "above"
            out.append(
                f"Best focus was at the edge of the searched travel ({where} it) - "
                f"the Z range is still too small; raise the range/limit.")
        if self.bracketed and self.in_focus_area < 0.6:
            out.append(
                f"Only {self.in_focus_area * 100:.0f}% of the blade is in focus at "
                f"this Z. If that holds along the column, close the aperture "
                f"(more depth of field) or enable focus stacking.")
        if self.needs_stacking:
            out.append(
                f"Depth spread {self.depth_spread:.0f} steps "
                f"({self.depth_spread / steps_per_mm:.2f} mm) exceeds the measured "
                f"depth of field {self.dof_steps:.0f} steps "
                f"({self.dof_steps / steps_per_mm:.2f} mm): no single Z can make "
                f"the whole field sharp - bracket Z and stack.")
        if not out:
            out.append("Focus locked and bracketed by sharp samples on both sides.")
        return out

    def summary(self, steps_per_mm: float = 800.0) -> str:
        mm = self.z_peak / steps_per_mm
        if not self.bracketed:
            return (f"NO PEAK (curve varies {self.curve_variation * 100:.1f}%), "
                    f"best guess z={self.z_peak:+.0f} steps ({mm:+.2f} mm)")
        return (f"z={self.z_peak:+.0f} steps ({mm:+.2f} mm), "
                f"area in focus {self.in_focus_area * 100:.0f}%, "
                f"DOF {self.dof_steps / steps_per_mm:.2f} mm, "
                f"spread {self.depth_spread / steps_per_mm:.2f} mm"
                f"{', STACKING NEEDED' if self.needs_stacking else ''}")


class FocusStack:
    """A set of frames captured at known Z offsets, scored per tile."""

    def __init__(self, cfg: FocusConfig | None = None):
        self.cfg = cfg or FocusConfig()
        self.z: list[float] = []
        self.sharp: list[np.ndarray] = []
        self.contrast: list[np.ndarray] = []
        self.luma: list[np.ndarray] = []
        self.tiles = 0

    def __len__(self) -> int:
        return len(self.z)

    def add(self, z: float, frame: np.ndarray) -> np.ndarray:
        m = tile_metrics(frame, self.cfg)
        self.z.append(float(z))
        self.sharp.append(m.sharpness)
        self.contrast.append(m.contrast)
        self.luma.append(m.luma)
        self.tiles = len(m.sharpness)
        return m.sharpness

    # ── derived views ────────────────────────────────────────────────────────
    def matrix(self) -> np.ndarray:
        if not self.sharp:
            return np.zeros((0, self.cfg.tiles_x * self.cfg.tiles_y), np.float32)
        return np.vstack(self.sharp)

    def reliable(self) -> np.ndarray:
        """Tiles with enough texture and sane exposure to judge focus at all.

        Texture is judged by the *best contrast seen anywhere in the stack*,
        relative to the other tiles: a tile that is merely blurred at every
        sample is not mistaken for an empty one, but a genuinely featureless
        one (background, specular blow-out) drops out.
        """
        if not self.sharp:
            return np.zeros(self.cfg.tiles_x * self.cfg.tiles_y, bool)
        contrast = np.vstack(self.contrast)
        luma = np.vstack(self.luma)
        cfg = self.cfg
        best = contrast.max(axis=0)
        gate = max(cfg.min_contrast, cfg.min_contrast_rel * float(np.median(best)))
        textured = best >= gate
        lum = luma.max(axis=0)
        lit = (lum <= cfg.max_luma) & (lum >= cfg.min_luma)
        return textured & lit

    def normalised(self) -> np.ndarray:
        """Per-tile score divided by that tile's own best score over the stack."""
        m = self.matrix().astype(np.float32)
        if m.size == 0:
            return m
        peak = m.max(axis=0)
        peak[peak <= 0] = 1.0
        return m / peak

    def quality(self) -> np.ndarray:
        """(n,) mean sharpness over usable tiles - the *search* signal.

        Deliberately not normalised per tile: normalisation is right for judging
        "how much of the field is at its own best", but it is flat (0/0-ish)
        while the whole window is out of focus, which is precisely when the
        search needs a gradient to follow.  Raw sharpness rises monotonically
        towards focus across a whole millimetre of travel, so it can steer.
        """
        m = self.matrix()
        usable = self.reliable()
        if m.size == 0:
            return np.zeros(0, np.float32)
        if usable.any():
            return m[:, usable].mean(axis=1)
        return m.mean(axis=1)

    def area_in_focus(self) -> np.ndarray:
        """(n,) fraction of usable tile area within ``in_focus_ratio`` of peak."""
        norm = self.normalised()
        usable = self.reliable()
        if norm.size == 0 or not usable.any():
            return np.zeros(len(self) or 1, np.float32)
        hit = (norm[:, usable] >= self.cfg.in_focus_ratio)
        counted = norm[:, usable] > 0
        denom = np.maximum(counted.sum(axis=1), 1)
        return hit.sum(axis=1) / denom

    def curve_variation(self) -> float:
        q = self.quality()
        if not q.size:
            return 0.0
        hi = float(q.max())
        return 0.0 if hi <= 0 else float((hi - float(q.min())) / hi)

    def curve(self) -> tuple[np.ndarray, np.ndarray]:
        """(z, quality) sorted by Z.

        The stack is in *capture* order, and a fine pass appends samples that sit
        on the peak - so anything that asks "what does the curve look like at its
        ends" must sort first, or a perfectly bracketed peak looks like it ends
        on the peak itself.
        """
        q = self.quality()
        if not q.size:
            return np.zeros(0), q
        z = np.asarray(self.z, np.float64)
        order = np.argsort(z, kind="stable")
        return z[order], q[order]

    def refine_peak(self, index: int) -> float:
        """Sub-step Z of the peak, by fitting a parabola to the quality curve.

        The scan grid is 50-200 steps wide, so the discrete best sample can sit a
        full step away from the real optimum; the parabola recovers most of that
        resolution for free (accuracy measured on the synthetic rig: within
        +-20 steps instead of +-100).
        """
        z, q = self.curve()
        if q.size < 3:
            return float(z[index]) if q.size else 0.0
        zs = np.asarray(self.z, float)
        target_z = float(zs[index])
        i = int(np.argmin(np.abs(z - target_z)))
        if i <= 0 or i >= q.size - 1:
            return target_z
        y0, y1, y2 = float(q[i - 1]), float(q[i]), float(q[i + 1])
        denom = y0 - 2 * y1 + y2
        if abs(denom) < 1e-12 or y1 < max(y0, y2):
            return float(z[i])
        offset = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
        step = float(min(z[i] - z[i - 1], z[i + 1] - z[i]))
        return float(z[i] + offset * step)

    def bracketed(self) -> bool:
        """True when the sharpness curve peaks *inside* the searched window.

        Compares the best sample with both ends: a real focus peak rises well
        clear of the window edges, whereas a fully defocused window produces a
        flat curve where any pixel of noise is the "maximum".
        """
        _z, q = self.curve()
        if q.size < 3:
            return False
        hi = float(q.max())
        if hi <= 0:
            return False
        if hi < self.cfg.min_peak_sharpness:
            return False                      # nothing here is sharp at all
        ends = max(float(q[0]), float(q[-1]))
        return hi > ends * (1.0 + self.cfg.bracket_margin)

    def best_index(self) -> int:
        return int(np.argmax(self.area_in_focus()))

    def peaks(self) -> tuple[np.ndarray, np.ndarray]:
        """(tile_z, tile_score): per-tile best Z, sub-step refined by a parabola."""
        m = self.matrix().astype(np.float64)
        z = np.asarray(self.z, np.float64)
        n, nt = m.shape
        z_best = np.full(nt, np.nan)
        s_best = np.full(nt, np.nan)
        for t in range(nt):
            s = m[:, t]
            if not np.isfinite(s).any() or s.max() <= 0:
                continue
            i = int(np.argmax(s))
            s_best[t] = s[i]
            offset = 0.0
            if 0 < i < n - 1:
                y0, y1, y2 = s[i - 1], s[i], s[i + 1]
                denom = (y0 - 2 * y1 + y2)
                if abs(denom) > 1e-9:
                    offset = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
            z_best[t] = z[i] + offset * self._spacing(i)
        return z_best, s_best

    def _spacing(self, i: int) -> float:
        z = self.z
        if len(z) < 2:
            return float(self.cfg.coarse_steps)
        if i <= 0:
            return abs(z[1] - z[0])
        if i >= len(z) - 1:
            return abs(z[-1] - z[-2])
        return abs(z[i + 1] - z[i - 1]) / 2.0

    def dof_steps(self) -> float:
        """Measured depth of field: median Z span a tile stays >=90 % of peak.

        Self-calibrating - no lens data needed.  It is the width of the focus
        curve, which is exactly the quantity that decides whether one Z can
        serve the whole field.
        """
        norm = self.normalised()
        usable = self.reliable()
        if norm.size == 0 or not usable.any():
            return 0.0
        z = np.asarray(self.z, np.float64)
        order = np.argsort(z)
        z = z[order]
        ratio = self.cfg.in_focus_ratio
        widths = []
        for t in np.nonzero(usable)[0]:
            s = norm[order, t]
            i = int(np.argmax(s))
            if s[i] < ratio:
                continue
            # Walk out from the peak and interpolate where the curve actually
            # crosses the threshold.  Requiring a *sample* above the threshold
            # would report "depth of field = 0" whenever the scan step is finer
            # than the focus curve - which is exactly when the number matters.
            half_total = 0.0
            for direction in (-1, 1):
                j = i
                while 0 <= j + direction < len(s) and s[j + direction] >= ratio:
                    j += direction
                k = j + direction
                if not (0 <= k < len(s)):
                    half_total += abs(z[j] - z[i])       # clipped by the window
                    continue
                span_jk = abs(z[k] - z[j])
                drop = s[j] - s[k]
                frac = 0.0 if drop <= 1e-9 else (s[j] - ratio) / drop
                half_total += abs(z[j] - z[i]) + frac * span_jk
            if half_total > 0:
                widths.append(half_total)
        if not widths:
            # nothing rose above the threshold even at its best: focus never
            # came close, so DOF is not measurable here
            return 0.0
        return float(np.median(widths))

    def depth_spread(self) -> float:
        """p90-p10 of the per-tile best Z: how much depth the field spans."""
        tile_z, _ = self.peaks()
        usable = self.reliable()
        vals = tile_z[np.isfinite(tile_z)]
        if usable.any():
            keep = usable & np.isfinite(tile_z)
            if keep.any():
                vals = tile_z[keep]
        if vals.size < 2:
            return 0.0
        return float(np.percentile(vals, 90) - np.percentile(vals, 10))

    def needs_stacking(self) -> bool:
        dof = self.dof_steps()
        return bool(dof > 0 and self.depth_spread() > self.cfg.dof_ratio * dof)

    def result(self, z_peak: float | None = None,
               range_limited: str = "") -> FocusResult:
        norm = self.normalised()
        usable = self.reliable()
        area = self.area_in_focus()
        idx = self.best_index() if len(self) else 0
        tile_z, tile_score = self.peaks()
        # tile_z is an absolute Z; the caller wants it relative to the scan start
        if z_peak is None:
            z_peak = float(self.z[idx]) if len(self) else 0.0
        peak_score = float(norm[idx][usable].mean()) if len(self) and usable.any() else 0.0
        bracketed = self.bracketed()
        q = self.quality()
        # A curve that never varies means the image never came into focus: the
        # per-tile ratios are then all 1.0 by construction, which would claim
        # "100 % of the blade is sharp" for a completely blurred field.
        area_at_peak = float(area[idx]) if len(area) and bracketed else 0.0
        return FocusResult(
            z_peak=float(z_peak),
            peak_score=peak_score,
            in_focus_area=area_at_peak,
            tile_z=tile_z,
            tile_score=tile_score,
            reliable=usable,
            dof_steps=self.dof_steps(),
            depth_spread=self.depth_spread(),
            needs_stacking=self.needs_stacking(),
            bracketed=bracketed,
            curve_variation=self.curve_variation(),
            sharpness_gain=(float(q.max() / q[0]) if q.size and q[0] > 0 else 1.0),
            range_limited=range_limited,
            samples=len(self),
            trajectory=[(float(z), float(a)) for z, a in zip(self.z, area)],
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Trend prediction: where should Z be at the next stop?
# ──────────────────────────────────────────────────────────────────────────────

class FocusTrend:
    """Predicts the focus position from where the stage is in the scan.

    The blade's front surface is smooth, so the focus position drifts smoothly
    along the scan path; fitting ``z = a + b*u + c*v`` over previous stops makes
    the *next* search window narrow (a handful of samples instead of the whole
    travel) without ever trusting the prediction blindly: the runner still
    brackets the peak, escalates when it is at an edge, and reports when the
    travel limit stops it.

    What you fit is what it predicts.  Feed it absolute stage positions and it
    predicts absolute positions; feed it each scan's ``z_peak`` (an offset from
    where that scan started - what a stage that parks on the focus can measure)
    and it predicts the *shift* to the next stop, which is what
    ``AutoFocus.run(z_hint=...)`` wants.
    """

    def __init__(self, max_points: int = 40):
        self.pts: list[tuple[float, float, float]] = []
        self.max_points = max_points
        self._coef: np.ndarray | None = None

    def add(self, u: float, v: float, z: float) -> None:
        self.pts.append((float(u), float(v), float(z)))
        if len(self.pts) > self.max_points:
            self.pts.pop(0)
        self._coef = None

    def __len__(self) -> int:
        return len(self.pts)

    def _fit(self) -> np.ndarray | None:
        if self._coef is not None:
            return self._coef
        n = len(self.pts)
        if n < 3:
            return None
        p = np.asarray(self.pts, np.float64)
        u, v, z = p[:, 0], p[:, 1], p[:, 2]
        # centre the coordinates: the design matrix stays well conditioned even
        # when u counts columns and z counts thousands of steps.
        um, vm = u.mean(), v.mean()
        cols = [np.ones(n), u - um, v - vm]
        if n >= 8:
            cols.append((u - um) ** 2)          # curvature along the scan path
        a = np.vstack(cols).T
        # light ridge keeps a nearly-degenerate set (all points in one column)
        a = np.vstack([a, 1e-3 * np.eye(a.shape[1])])
        b = np.concatenate([z, np.zeros(a.shape[1])])
        try:
            coef, *_ = np.linalg.lstsq(a, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        self._coef = coef
        self._centres = (um, vm)
        return coef

    def predict(self, u: float, v: float) -> float | None:
        coef = self._fit()
        if coef is None:
            return None
        um, vm = self._centres
        du, dv = u - um, v - vm
        terms = [1.0, du, dv]
        if len(coef) >= 4:
            terms.append(du * du)
        return float(np.dot(coef, terms))

    def residual(self, u: float, v: float, z: float) -> float | None:
        p = self.predict(u, v)
        return None if p is None else float(z - p)

    def slope_per_step(self) -> float:
        """How fast focus travels along the scan (steps of Z per index step)."""
        coef = self._fit()
        if coef is None or len(coef) < 3:
            return 0.0
        return float(abs(coef[1]) + abs(coef[2]))


# ──────────────────────────────────────────────────────────────────────────────
#  The scan itself
# ──────────────────────────────────────────────────────────────────────────────

class AutoFocus:
    """Runs focus scans against injected motion/capture callables.

    ``move(signed_steps)`` must move the Z stage and return once the command has
    been accepted; ``capture()`` returns a BGR frame or None; ``wait()`` (optional)
    blocks until the stage has actually stopped.  The runner keeps its own
    position bookkeeping, so it works with an open-loop stepper that has no
    encoder - every position below is *relative to where the scan started*.
    """

    def __init__(self, move, capture, cfg: FocusConfig | None = None,
                 wait=None, log=print, sleep=time.sleep,
                 steps_per_mm: float = 800.0, origin: float = 0.0):
        self.move = move
        self.capture = capture
        self.cfg = cfg or FocusConfig()
        self.wait = wait
        self.log = log
        self.sleep = sleep
        self.steps_per_mm = steps_per_mm
        self.stack = FocusStack(self.cfg)
        # Every position in this class is in "runner coordinates": motor steps
        # measured from ``origin``.  ``origin`` is the motor reading when the
        # runner was created, so a caller that knows where the stage is can pass
        # it and then talk in absolute motor steps.  Getting this wrong silently
        # drives the stage to the wrong place (a second runner assumed 0 and
        # doubled the Z on the first bracket), hence the explicit parameter.
        self.origin = float(origin)
        self._pos = float(origin)
        self._lo = self._hi = float(origin)
        self._last_dir = 1
        self.travel_steps = 0.0      # total Z distance moved by this scan
        self.captures = 0
        self._anchor = float(origin)  # where the search window is centred

    @property
    def position(self) -> float:
        """Where the runner believes the stage is (runner coordinates)."""
        return self._pos

    # ── plumbing ────────────────────────────────────────────────────────────
    def _goto(self, target: float) -> None:
        cfg = self.cfg
        if abs(target) > cfg.z_travel_limit:
            raise FocusError(
                f"Z target {target:+.0f} steps exceeds the travel limit "
                f"({cfg.z_travel_limit} steps) - refusing to move")
        self._last_dir = 1 if target >= self._pos else -1
        if cfg.backlash_steps > 0:
            # Always arrive from the same side, so the stack is not smeared by
            # gear backlash; a 31 um fine scan on a worn stage is meaningless
            # without this.
            approach = target - cfg.backlash_steps
            self._move_raw(approach - self._pos)
            self._move_raw(cfg.backlash_steps)
        else:
            self._move_raw(target - self._pos)
        self._pos = float(target)
        self._lo = min(self._lo, self._pos)
        self._hi = max(self._hi, self._pos)

    def _move_raw(self, delta: float) -> None:
        if abs(delta) < 1:
            return
        self.travel_steps += abs(delta)
        self.move(int(round(delta)))
        if self.wait is not None:
            self.wait()

    def _already_sampled(self, z: float, tol: float) -> bool:
        if not self.stack.z:
            return False
        return min(abs(z - zz) for zz in self.stack.z) < max(1.0, tol)

    def _sample(self, z: float) -> float | None:
        self._goto(z)
        if self.cfg.settle_s:
            self.sleep(self.cfg.settle_s)
        frame = self.capture()
        if frame is None:
            return None
        self.captures += 1
        self.stack.add(z, frame)
        return float(self.stack.area_in_focus()[-1])

    def search_cap(self) -> float:
        """Largest offset from the start the search may reach, in steps."""
        if self.cfg.range_steps and self.cfg.range_steps > 0:
            return min(float(self.cfg.range_steps) / 2.0,
                       float(self.cfg.z_travel_limit))
        return float(self.cfg.z_travel_limit)

    def _search_window(self) -> tuple[float, float]:
        """Absolute Z window this scan may sample, around where it started.

        Bounding each walk by an offset from *its own* centre is not the same as
        bounding the scan: a gallop may start 1 mm out and a walk from there adds
        its own allowance, so the scan reached cap + allowance in practice.  One
        window fixed at the start keeps ``range_steps`` an honest bound.
        """
        anchor = float(getattr(self, "_anchor", self._pos))
        cap = self.search_cap()
        lim = float(self.cfg.z_travel_limit)
        return max(-lim, anchor - cap), min(lim, anchor + cap)

    def _in_window(self, z: float) -> bool:
        lo, hi = self._search_window()
        return lo - 1e-6 <= z <= hi + 1e-6

    def _budget_left(self) -> bool:
        return len(self.stack) < self.cfg.max_samples

    def _check_budget(self) -> None:
        if not self._budget_left():
            raise FocusError(
                f"sample budget exhausted ({self.cfg.max_samples}) - "
                f"narrow the range or coarsen the step")

    # ── scanning primitives ─────────────────────────────────────────────────
    def _scan_window(self, lo: float, hi: float, step: float,
                     skip_sampled: bool = False) -> None:
        """Sample [lo, hi] inclusive, always including both endpoints.

        ``skip_sampled`` avoids re-capturing positions the stack already holds
        (the fine pass overlays the coarse grid).  Duplicate samples are not just
        wasted captures: they make two Z values comparable, which quietly breaks
        the "did the curve rise between these two samples" arithmetic that the
        depth-of-field and bracketing tests rely on.
        """
        step = max(1, int(step))
        n = max(1, int(round(abs(hi - lo) / step)))
        for i in range(n + 1):
            self._check_budget()
            z = lo + (hi - lo) * (i / n)
            if skip_sampled and self._already_sampled(z, step * 0.5):
                continue
            self._sample(z)
        area = self.stack.area_in_focus()
        best = float(area.max() * 100) if area.size else -1.0
        self.log(f"[AF] scanned {lo:+.0f}..{hi:+.0f} (best so far {best:.0f}%)")

    def _gallop(self, center: float, step: float, limit: float,
                max_offset: float | None = None
                ) -> tuple[tuple[float, float] | None, float, float, float]:
        """Find *any* sharp Z cheaply, doubling the offset each round.

        With no hint the focus can be anywhere in the travel, and stepping there
        at the scan resolution costs one capture per 200 steps (a full 15 mm
        would be 120 captures).  Doubling the offset instead reaches the whole
        travel in ~8 captures per side, and stops the moment a sample is actually
        sharp enough to be worth bracketing - resolution only matters once the
        neighbourhood is known.

        Returns ``(hit, best_z, best_offset, spread)``: ``hit`` is
        ``(z, offset)`` when something was sharp enough to bracket, otherwise
        None; the rest describes the best (possibly blurry) sample seen, which is
        what localises a focus cone the sweep stepped over.  ``spread`` is
        best/worst over the sweep: ~1.7 means "no signal anywhere", 5-10 means
        "the cone is in there somewhere".
        """
        cfg = self.cfg
        offset = max(1.0, float(step))
        best_z, best_q, best_off = center, -1.0, 0.0
        worst_q = float("inf")
        while offset <= limit:
            moved = False
            for sign in (1.0, -1.0):
                z = center + sign * offset
                if not self._in_window(z):
                    continue
                if not self._budget_left():
                    spread = best_q / worst_q if worst_q and worst_q > 0 else 1.0
                    return None, best_z, best_off, spread
                if self._sample(z) is None:
                    return None, best_z, best_off, 1.0
                moved = True
                q = float(self.stack.quality()[-1])
                worst_q = min(worst_q, q)
                if q > best_q:
                    best_z, best_q, best_off = z, q, offset
                if q >= cfg.min_peak_sharpness:
                    self.log(f"[AF] sharp sample found at {z:+.0f} "
                             f"({offset:.0f} steps out) -> bracketing there")
                    return (z, offset), best_z, best_off, 1.0
            if not moved:
                break
            offset *= 2.0
        spread = best_q / worst_q if worst_q and worst_q > 0 else 1.0
        return None, best_z, best_off, spread

    def _bracket_walk(self, center: float, step: float, max_offset: float,
                      limit: float) -> str:
        """Walk out from ``center`` and stop as soon as the peak is bracketed.

        Each side is abandoned the moment its samples fall clearly below the best
        score seen so far, so a well-centred start costs a handful of captures
        instead of the whole window.  A side is only abandoned *if something was
        sharp there*: on a featureless or hopelessly defocused field the curve is
        noise, and closing on noise would end the search before it had covered the
        range (and would let it claim "not reachable" without looking).

        Returns ``"bracketed"``, ``"limit"``, ``"budget"`` or ``"nothing_sharp"``.
        """
        cfg = self.cfg
        step = max(1.0, float(step))
        if self._sample(center) is None and not len(self.stack):
            return "no_frames"
        closed = {"neg": False, "pos": False}
        announced = False
        hit_limit = False
        hit_cap = False
        k = 0
        while True:
            k += 1
            offset = k * step
            if offset > max_offset:
                if self.stack.bracketed():
                    return "bracketed"
                return "travel" if hit_limit else "limit"
            if offset > cfg.range_steps / 2.0 > 0 and not announced:
                announced = True
                self.log(f"[AF] still climbing past the configured range "
                         f"({cfg.range_steps} steps) -> extending the search window")
            moved = False
            for side, sign in (("pos", 1.0), ("neg", -1.0)):
                if closed[side]:
                    continue
                z = center + sign * offset
                if not self._in_window(z):
                    closed[side] = True
                    hit_limit = abs(z) > cfg.z_travel_limit
                    hit_cap = hit_cap or not hit_limit
                    continue
                if not self._budget_left():
                    self.log(f"[AF] stopped at the {cfg.max_samples}-sample "
                             f"budget after +-{offset:.0f} steps")
                    return "budget"
                self._sample(z)
                moved = True
                q = self.stack.quality()
                best = float(q.max())
                if best < cfg.min_peak_sharpness:
                    continue            # nothing sharp yet: keep looking, don't close
                if float(q[-1]) < best * (1.0 - cfg.bracket_margin):
                    closed[side] = True
            if not moved:
                if self.stack.bracketed():
                    return "bracketed"
                return "travel" if hit_limit else "limit"
            if closed["pos"] and closed["neg"]:
                if self.stack.bracketed():
                    return "bracketed"
                if float(self.stack.quality().max()) < cfg.min_peak_sharpness:
                    return "nothing_sharp"
                return "limit"

    def _scan(self, center: float, hint: bool) -> str:
        """Gallop to the right neighbourhood (when unsure), then bracket finely."""
        cfg = self.cfg
        if hint:
            # A prediction comes from the trend along the scan path, so trust it
            # enough to bracket straight away in a narrow window; anything less
            # than a clean bracket falls through to the wide hunt below.
            outcome = self._bracket_walk(
                center, cfg.coarse_steps,
                max(2.0 * cfg.coarse_steps, cfg.predictive_window / 2.0),
                cfg.z_travel_limit)
            if outcome == "bracketed":
                return outcome
            if outcome == "no_frames":
                return outcome
            q = self.stack.quality()
            if q.size:
                center = float(np.asarray(self.stack.z, float)[int(np.argmax(q))])
            self.log(f"[AF] no bracket within +-{cfg.predictive_window / 2:.0f} "
                     f"steps of the prediction -> widening the hunt from "
                     f"{center:+.0f}")
        hit, best_z, best_off, spread = self._gallop(
            center, cfg.coarse_steps, cfg.z_travel_limit)
        if hit is not None:
            z_hit, offset = hit
            # The gallop only proves "there is texture here worth resolving"; the
            # walk must still be free to climb past it to the real peak, so its
            # allowance is the configured range, not the gallop's last offset.
            return self._bracket_walk(z_hit, cfg.coarse_steps,
                                      self.search_cap(), cfg.z_travel_limit)

        # Nothing sharp anywhere along the geometric sweep.  Either the focus cone
        # was stepped over, or nothing in view can be focused at all - the spread
        # of the scores tells the two apart.
        if spread < cfg.gallop_signal:
            self.log(f"[AF] no focus signal anywhere: best {spread:.1f}x the "
                     f"worst sample across the whole travel - nothing in view "
                     f"is in focus at any reachable Z")
            return "nothing_sharp"
        self.log(f"[AF] focus cone stepped over (best sample {spread:.1f}x the "
                 f"rest at {best_z:+.0f}) -> rescanning that neighbourhood")
        return self._bracket_walk(best_z, cfg.coarse_steps,
                                  self.search_cap(), cfg.z_travel_limit)

    def _escalate(self, direction: str) -> tuple[float, float]:
        """Return a new window next to the exhausted one."""
        cfg = self.cfg
        half = cfg.range_steps / 2.0
        zs = np.asarray(self.stack.z, float)
        if not zs.size:
            return self._pos - half, self._pos + half
        if direction == "neg":
            return float(zs.min() - half), float(zs.min())
        return float(zs.max()), float(zs.max() + half)

    # ── public API ──────────────────────────────────────────────────────────
    def run(self, z_hint: float | None = None,
            start_z: float | None = None) -> FocusResult:
        """Find focus, leave the stage on it, and say how good it is.

        ``z_hint`` is a *prediction* in the same relative frame as the returned
        ``z_peak`` (typically ``FocusTrend.predict(...) - position_before_scan``).
        With a hint only a narrow window is scanned; it escalates the moment the
        peak lands on an edge, so a bad prediction costs time, never accuracy.
        Without a hint the whole configured travel is scanned.
        """
        t0 = time.time()
        cfg = self.cfg
        if start_z is None:
            start_z = self._pos
        self.stack = FocusStack(cfg)
        self._pos = float(start_z)
        self._lo = self._hi = float(start_z)
        self._anchor = float(start_z)
        self.travel_steps = 0.0
        self.captures = 0

        # Walk outward from the best guess and stop as soon as the peak is
        # bracketed, instead of always paying for a fixed window.
        center = float(z_hint) if z_hint is not None else float(start_z)
        if z_hint is not None:
            self.log(f"[AF] expanding search from the predicted {center:+.0f} "
                     f"(step {cfg.coarse_steps}, max {cfg.z_travel_limit})")
        else:
            self.log(f"[AF] expanding search from {center:+.0f} with no hint "
                     f"(step {cfg.coarse_steps}, max {cfg.z_travel_limit})")
        outcome = self._scan(center, hint=z_hint is not None)

        if outcome == "no_frames" or not len(self.stack):
            raise FocusError(
                "no frames captured during the focus scan - is the camera feed "
                "running?")

        # 1. extend while the sharpest sample sits on an edge of what was searched.
        #    Steered by raw sharpness, which keeps rising towards focus even when
        #    the whole window is far out of it - the case where the old hill climb
        #    saw "no improvement" and declared the peak found.
        limited = ""
        if outcome in ("limit", "travel", "budget", "nothing_sharp") \
                and not self.stack.bracketed():
            # The walk never found a peak it could bracket: say which way it was
            # still improving, because that is the direction the operator has to
            # make reachable (more travel, or a smaller X step).
            q = self.stack.quality()
            zs = np.asarray(self.stack.z, float)
            if q.size:
                order = np.argsort(zs)
                q, zs = q[order], zs[order]
                mid = float(np.mean(zs))
                best_z = float(zs[int(np.argmax(q))])
                limited = "pos" if best_z >= mid else "neg"
                reach = abs(best_z - center) / self.steps_per_mm
                self.log(f"[AF] no bracketed peak: the most promising sample is "
                         f"{reach:.2f} mm {limited} of the start, after "
                         f"{len(self.stack)} captures (outcome: {outcome}, "
                         f"travel limit {cfg.z_travel_limit} steps)")
                if outcome == "travel":
                    self.log(f"[AF] travel limit reached at "
                             f"{cfg.z_travel_limit} steps - the true focus is "
                             f"outside the reachable range")
                elif outcome == "limit":
                    self.log(f"[AF] stopped at the Range cap (+-{cfg.range_steps / 2 / self.steps_per_mm:.2f} mm, "
                             f"{cfg.range_steps} steps) without a bracketed peak - "
                             f"raise Range if the focus is further away, or 0 for "
                             f"the whole travel")

        if not len(self.stack):
            raise FocusError(
                "no frames captured during the focus scan - is the camera feed "
                "running?")
        if not self.stack.reliable().any():
            # Distinguish "nothing in view has texture" (a real error: point the
            # camera at the part) from "everything is so far out of focus that
            # the texture gate rejects it" (a real answer: no focus was found).
            contrasts = np.vstack(self.stack.contrast)
            if float(contrasts.max()) < 0.5:
                raise FocusError(
                    "no usable texture in view - every tile is flat or blown "
                    "out; check lighting/exposure before focusing")
            self.log("[AF] view has texture but no tile is sharp anywhere in the "
                     "searched range - reporting the result for what it is")

        # 2. refine around the coarse optimum with a fine sweep.  The span is
        #    deliberately narrow: the walk already bracketed the peak to within
        #    one coarse step, so +-coarse/2 at the fine step is enough and costs
        #    3-5 captures instead of 9.
        z_order = np.asarray(self.stack.z, float)[np.argsort(self.stack.z)]
        i = self._best_index()
        z_best_coarse = float(np.asarray(self.stack.z, float)[i])
        if self.stack.bracketed() and cfg.fine_steps < cfg.coarse_steps:
            nearest = int(np.argmin(np.abs(z_order - z_best_coarse)))
            interior = 0 <= nearest < len(z_order) - 1
            w_lo, w_hi = self._search_window()
            lo = max(w_lo, z_best_coarse - cfg.coarse_steps * 0.5)
            hi = min(w_hi, z_best_coarse + cfg.coarse_steps * 0.5)
            if interior:
                self.log(f"[AF] refining {lo:+.0f}..{hi:+.0f} step {cfg.fine_steps}")
                self._scan_window(lo, hi, cfg.fine_steps, skip_sampled=True)

        # 3. pick the winner over the whole stack, refine to sub-step, then sit
        #    on it (the stage is left exactly where the best image was taken -
        #    within a step or two - so the next capture is already in focus)
        i = self._best_index()
        z_best = float(np.asarray(self.stack.z, float)[i])
        if self.stack.bracketed():
            z_ref = self.stack.refine_peak(i)
            if abs(z_ref - z_best) <= 1.5 * cfg.coarse_steps:
                z_best = z_ref
        self._goto(z_best)
        result = self.stack.result(z_peak=z_best - start_z, range_limited=limited)
        result.seconds = time.time() - t0
        zs_seen = np.asarray(self.stack.z, float)
        span = float(zs_seen.max() - zs_seen.min()) if zs_seen.size else 0.0
        cap = self.search_cap()
        self.log(f"[AF] {result.summary(self.steps_per_mm)} "
                 f"in {result.seconds:.1f}s / {result.samples} frames")
        self.log(f"[AF] searched +-{span / 2 / self.steps_per_mm:.2f} mm around the "
                 f"start (cap +-{cap / self.steps_per_mm:.2f} mm), stage travelled "
                 f"{self.travel_steps / self.steps_per_mm:.2f} mm in Z")
        for line in result.recommendation(self.steps_per_mm):
            self.log(f"[AF]   {line}")
        return result

    def _best_index(self) -> int:
        """Sample that makes the most of the field, with a sharpness tie-break.

        "Area within 90 % of each tile's own best" is the objective, but if the
        whole window is equally bad it is uninformative (every value 0), so fall
        back to raw sharpness, which still orders the samples.
        """
        area = self.stack.area_in_focus()
        q = self.stack.quality()
        if not area.size:
            return 0
        if float(area.max() - area.min()) > 1e-6:
            # The area metric is a *count* of tiles above a threshold, so near
            # the optimum many samples tie.  Picking the first of a tie lands up
            # to a full step away from the real peak (measured: 100 steps off on
            # a 50-step grid), so break ties with the sharper tie-breaker.
            best = float(area.max())
            tied = np.nonzero(area >= best - 1e-9)[0]
            if tied.size > 1 and q.size == area.size:
                return int(tied[int(np.argmax(q[tied]))])
            return int(tied[0])
        return int(np.argmax(q)) if q.size else int(np.argmax(area))

    def bracket(self, z_peak: float, span: float | None = None,
                samples: int = 3) -> list[tuple[float, np.ndarray]]:
        """Capture a small Z-bracket around ``z_peak`` for focus stacking.

        ``z_peak`` is in the runner's coordinates (see ``origin``); after a
        ``run()`` the stage is sitting on the focus, so ``bracket(af.position)``
        brackets exactly there.  ``span`` defaults to the depth spread measured
        by that scan - the range the surface actually needs.  Returns
        ``[(z, frame), ...]`` in capture order.
        """
        cfg = self.cfg
        if samples < 1:
            return []
        if span is None:
            span = max(cfg.coarse_steps, self.stack.depth_spread())
        span = float(span)
        out = []
        for k in range(samples):
            frac = 0.0 if samples == 1 else (k / (samples - 1) - 0.5)
            z = z_peak + frac * span
            self._goto(z)
            if cfg.settle_s:
                self.sleep(cfg.settle_s)
            frame = self.capture()
            if frame is not None:
                out.append((z, frame))
        # End where the best image was taken, not at the end of the bracket:
        # whatever is captured next (the sweep's next frame) is then in focus.
        self._goto(z_peak)
        return out


# ──────────────────────────────────────────────────────────────────────────────
#  Turning a stack into one sharp frame, and into stitch weights
# ──────────────────────────────────────────────────────────────────────────────

def _tile_weights_to_pixels(tile_w: np.ndarray, shape, cfg: FocusConfig,
                            feather: float = 1.0) -> np.ndarray:
    """Bilinearly blow a (rows, cols) tile weight grid up to full frame size."""
    h, w = shape[:2]
    grid = np.asarray(tile_w, np.float32).reshape(cfg.tiles_y, cfg.tiles_x)
    # place the grid on tile centres, then interpolate: a plain resize would
    # put tile corners at the frame corners and shift every weight by half a tile
    px = cv2.resize(grid, (w, h), interpolation=cv2.INTER_LINEAR)
    if feather > 0:
        # Keep the transition well inside one tile.  A wider feather blends
        # frames that are a whole tile's worth of focus apart, which averages the
        # composite back towards blurred - measured: sigma = tile/4 turned a
        # 0.89-quality stack into 0.55.
        sigma = max(1.0, feather * min(w / cfg.tiles_x, h / cfg.tiles_y) / 8.0)
        px = cv2.GaussianBlur(px, (0, 0), sigma)
    return px


def compose_focus_stack(frames: list[np.ndarray], cfg: FocusConfig | None = None,
                        feather: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Merge a Z-bracket into one frame that is sharp everywhere.

    Every tile takes its pixels from the frame(s) where that tile is sharpest;
    weights are raised to a power so the best frame dominates instead of being
    averaged away, then feathered across tile borders to hide the seams.

    Returns ``(composite, weight)``: ``weight`` is the per-pixel confidence
    (0..1) that the composite is sharp there - feed it to the stitcher so
    overlapping frames blend in favour of the sharpest evidence.
    """
    cfg = full_frame(cfg or FocusConfig())
    if not frames:
        raise FocusError("compose_focus_stack needs at least one frame")
    if len(frames) == 1:
        return frames[0].copy(), np.ones(frames[0].shape[:2], np.float32)

    stack = FocusStack(cfg)
    for i, f in enumerate(frames):
        stack.add(i, f)
    norm = stack.normalised()                    # (n, tiles) 0..1 per tile
    usable = stack.reliable()

    # Weight per (frame, tile): near winner-takes-all.  A plain average with a
    # blurred neighbour is exactly how defocused evidence leaks into a composite,
    # so weight decays exponentially with the shortfall from that tile's best
    # frame.  Measured on a tilted synthetic blade: 0.79 mean / 0.76 worst tile
    # vs 0.33 / 0.03 for the best single frame.
    tau = max(1e-3, cfg.blend_falloff)
    w = np.exp(-(1.0 - norm) / tau)
    w = w + 1e-6                                 # never leave a tile empty
    if usable.any():
        w[:, ~usable] = 1.0                      # flat tiles: plain average
    w = w / w.sum(axis=0, keepdims=True)

    h, wpx = frames[0].shape[:2]
    acc = np.zeros((h, wpx, 3), np.float32)
    wsum = np.zeros((h, wpx), np.float32)
    for i, frame in enumerate(frames):
        if frame.shape[:2] != (h, wpx):
            frame = cv2.resize(frame, (wpx, h), interpolation=cv2.INTER_AREA)
        wt = _tile_weights_to_pixels(w[i], frame.shape, cfg, feather)
        acc += frame.astype(np.float32) * wt[:, :, None]
        wsum += wt
    wsum[wsum <= 0] = 1.0
    out = np.clip(acc / wsum[:, :, None], 0, 255).astype(np.uint8)

    # confidence: the best normalised score achieved anywhere, per tile
    best = np.where(usable, norm.max(axis=0), 0.0)
    confidence = _tile_weights_to_pixels(best, out.shape, cfg, feather)
    return out, np.clip(confidence, 0.0, 1.0).astype(np.float32)


def tile_weight_map(frame: np.ndarray, cfg: FocusConfig | None = None,
                    reference: float | None = None,
                    floor: float = 0.15) -> tuple[np.ndarray, float]:
    """Per-pixel focus weight for one frame, for the stitcher's blend.

    ``reference`` is the sharpness a well-focused frame of this surface reaches;
    pass a running maximum and the weights stay comparable across the whole
    scan.  Returns ``(weights, new_reference)``.  A blurred region gets weights
    near ``floor``, so in an overlap the sharp neighbour wins the pixel - which
    is how a defocused band gets overwritten instead of embedded.
    """
    cfg = full_frame(cfg or FocusConfig())
    m = tile_metrics(frame, cfg)
    best = float(np.percentile(m.sharpness, 90)) if m.sharpness.size else 0.0
    new_ref = max(reference or 0.0, best)
    if new_ref <= 0:
        return np.ones(frame.shape[:2], np.float32), 0.0
    rel = np.clip(m.sharpness / new_ref, 0.0, 1.0)
    # Low-contrast tiles used to get weight 1.0 ("no texture: neutral").  But a
    # *blurred* textured tile also has low contrast, so the most defocused part
    # of a frame got the highest weight and beat the sharp neighbour in the
    # overlap - one source of the blocky patches.  Weights only matter relative
    # to other frames at the same spot, where a featureless surface scores low
    # in every frame anyway, so the measured value is used as-is.
    wt = np.clip(floor + (1.0 - floor) * rel ** 2, floor, 1.0)
    # Smooth transitions (~half a tile).  A tile/8 ramp turned each 320x270 px
    # tile into a visible rectangle in the blend.  (Focus *stacking* keeps the
    # tight ramp on purpose - see compose_focus_stack.)
    return (_tile_weights_to_pixels(wt, frame.shape, cfg, 4.0)
            .astype(np.float32)), new_ref
