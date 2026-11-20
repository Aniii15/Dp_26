"""Lens model: which pixels of a frame are real image, and how bright they should be.

Why this exists (measured on ``main_code/blade_images/scan_20261105_200702``):

* The **top-right and bottom-right corners are clipped** by the optics - the
  95th-percentile brightness over the whole scan is ~0 there, i.e. those pixels
  are black in *every* frame.  The old stitchers treated the full 1920x1080
  rectangle as valid, so each frame painted two black wedges into the mosaic,
  and the feathering then smeared them into dark rectangular "patches" along
  the right side of the blade - the blocky look.
* **Vignetting**: brightness falls to ~85 % at the right edge and ~90 % at the
  top/bottom of the frame even inside the image circle.  With only a per-frame
  scalar gain every frame edge shows up as a step in brightness.

The model is estimated from the scan itself (no calibration target needed):
a high percentile over many frames removes the scene and keeps the optics.
Only the right half of the frame is used for the fit, because in this rig the
left part of the view is the blade edge / black background, which says nothing
about the lens.  The model is radially symmetric about the optical centre, so
fitting one half constrains the whole frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LensModel:
    """Per-pixel validity weight and flat-field gain for a fixed camera."""

    valid: np.ndarray        # HxW float32 in [0, 1]; 0 = clipped by the optics
    flat: np.ndarray         # HxW float32 gain (>= 1) that undoes vignetting
    radius: float            # image-circle radius in px (inf = not clipped)
    center: tuple[float, float]

    @property
    def shape(self) -> tuple[int, int]:
        return self.valid.shape[:2]

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Flat-field a frame (uint8 in, uint8 out)."""
        out = frame.astype(np.float32)
        out *= self.flat[:, :, None] if out.ndim == 3 else self.flat
        return np.clip(out, 0, 255).astype(np.uint8)

    def resized(self, w: int, h: int) -> "LensModel":
        if (h, w) == self.shape:
            return self
        k = w / self.shape[1]
        return LensModel(
            cv2.resize(self.valid, (w, h), interpolation=cv2.INTER_LINEAR),
            cv2.resize(self.flat, (w, h), interpolation=cv2.INTER_LINEAR),
            self.radius * k, (self.center[0] * k, self.center[1] * k))

    @staticmethod
    def identity(h: int, w: int) -> "LensModel":
        return LensModel(np.ones((h, w), np.float32), np.ones((h, w), np.float32),
                         float("inf"), (w / 2.0, h / 2.0))


def _radius_grid(h: int, w: int, cx: float, cy: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.hypot(xx - cx, yy - cy)


def estimate_lens(frames, percentile: float = 92.0, work_width: int = 480,
                  fit_x_from: float = 0.55, clip_level: float = 0.45,
                  margin_px: float = 24.0, feather_px: float = 60.0,
                  max_gain: float = 1.6) -> LensModel:
    """Estimate the lens model from an iterable of BGR/gray frames.

    ``fit_x_from``  - only columns right of this fraction are used for the fit
                      (the left of the view is blade edge / background here).
    ``clip_level``  - a radius counts as clipped where the percentile image
                      drops below this fraction of its central level.
    ``margin_px``   - pull the valid boundary inwards by this many full-res px
                      (JPEG ringing and the soft optical edge live there).
    ``feather_px``  - width of the 1 -> 0 ramp at the boundary.
    """
    small = []
    full_shape = None
    for f in frames:
        if f is None:
            continue
        g = f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if full_shape is None:
            full_shape = g.shape[:2]
        elif g.shape[:2] != full_shape:
            continue
        k = work_width / g.shape[1]
        small.append(cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA))
    if full_shape is None:
        raise ValueError("estimate_lens: no frames")
    H, W = full_shape
    if len(small) < 5:
        return LensModel.identity(H, W)

    stack = np.stack(small).astype(np.float32)
    hi = np.percentile(stack, percentile, axis=0)
    hi = cv2.GaussianBlur(hi, (0, 0), 2.0)
    h, w = hi.shape
    k = w / W
    cx, cy = w / 2.0, h / 2.0
    r = _radius_grid(h, w, cx, cy)
    use = np.zeros_like(hi, bool)
    use[:, int(w * fit_x_from):] = True

    # Radial profile from the right half: median brightness per radius bin.
    r_max = float(np.hypot(cx, cy))
    bins = np.linspace(0, r_max, 64)
    idx = np.clip(np.digitize(r[use], bins) - 1, 0, len(bins) - 2)
    vals = hi[use]
    prof = np.full(len(bins) - 1, np.nan, np.float32)
    for b in range(len(bins) - 1):
        sel = vals[idx == b]
        if sel.size >= 20:
            prof[b] = float(np.median(sel))
    centres = 0.5 * (bins[:-1] + bins[1:])
    ok = np.isfinite(prof)
    if ok.sum() < 8:
        return LensModel.identity(H, W)
    prof, centres = prof[ok], centres[ok]
    # central level: the brightest robust part of the profile (inner 60 %)
    inner = centres < 0.6 * r_max
    level = float(np.percentile(prof[inner] if inner.any() else prof, 75))
    if level < 10:
        return LensModel.identity(H, W)

    # Image circle: first radius (going outwards, past the centre region) where
    # the profile collapses below clip_level of the central level.
    radius = float("inf")
    for c, p in zip(centres, prof):
        if c > 0.4 * r_max and p < clip_level * level:
            radius = float(c)
            break

    # Vignetting: smooth, monotone fall-off inside the circle, fitted as
    # log(I) = a + b r^2 + c r^4 on unclipped bins.
    inside = centres < (radius - margin_px * k if np.isfinite(radius) else np.inf)
    inside &= prof > clip_level * level
    flat_small = np.ones((h, w), np.float32)
    if inside.sum() >= 6:
        rr = centres[inside] / r_max
        A = np.stack([np.ones_like(rr), rr ** 2, rr ** 4], 1)
        coef, *_ = np.linalg.lstsq(A, np.log(prof[inside]), rcond=None)
        rn = r / r_max
        model = np.exp(coef[1] * rn ** 2 + coef[2] * rn ** 4)   # 1 at the centre
        model = np.minimum(model, 1.0)          # never darken anything
        flat_small = np.clip(1.0 / np.maximum(model, 1e-3), 1.0, max_gain)

    # Full-resolution maps
    R = _radius_grid(H, W, W / 2.0, H / 2.0)
    if np.isfinite(radius):
        rad_full = radius / k
        edge = rad_full - margin_px
        valid = np.clip((edge - R) / max(feather_px, 1.0), 0.0, 1.0)
    else:
        rad_full = float("inf")
        valid = np.ones((H, W), np.float32)
    flat = cv2.resize(flat_small, (W, H), interpolation=cv2.INTER_LINEAR)
    return LensModel(valid.astype(np.float32), flat.astype(np.float32),
                     rad_full, (W / 2.0, H / 2.0))


def default_lens(h: int, w: int, radius_frac: float = 0.56,
                 margin_px: float = 24.0, feather_px: float = 60.0) -> LensModel:
    """A conservative lens model when nothing is known yet (live preview).

    ``radius_frac`` is the image-circle radius as a fraction of the frame
    *width*; 0.56 x 1920 = 1075 px matches the clipping measured on the
    recorded scans (corners at r = 1101 px are black, r < 1050 px is fine).
    """
    R = _radius_grid(h, w, w / 2.0, h / 2.0)
    rad = radius_frac * w
    valid = np.clip((rad - margin_px - R) / feather_px, 0.0, 1.0).astype(np.float32)
    return LensModel(valid, np.ones((h, w), np.float32), rad, (w / 2.0, h / 2.0))
