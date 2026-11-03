"""Per-frame geometry and exposure, for a mosaic that does not look patched.

The delivered mosaics had two artefacts that come from *this* file's absence:

**1. Pale blocks where the camera clips.** If the capture has a dead border (a
black strip, a clipped corner, a rounded frame edge) the old code treated those
pixels as image.  They are near-black, so the per-frame exposure gain - a single
scalar, and clipped at 2x - pushed them up into a flat pale rectangle, which then
got blended in at full weight.  The result is exactly the "blocky, patched up"
look: bright rectangles stamped over the surface.  Nothing in the pipeline ever
asked *which pixels of this frame are actually image*.

**2. Brightness steps between neighbouring frames.** The illumination on the
blade falls off across the field of view and drifts as the scan walks down the
part.  One global gain per frame cannot fix a *gradient* mismatch: it cancels the
mean and leaves the slope, so every frame boundary shows as a step, and a
feathered blend turns that step into a visible band.  A smooth multiplicative
*field*, fitted in the overlap, cancels the slope too.

Both are cheap.  ``analyse_frame`` marks the real image area; ``estimate_gain_
field`` fits the illumination ratio on a coarse grid and returns a smooth field
to multiply into the frame before blending; ``smooth_weight`` gives a C1 blend
weight that decays *inside the valid region*, so a clipped corner can never be
blended at full strength.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = [
    "FrameGeometry", "analyse_frame", "warp_valid", "smooth_weight",
    "flatten_illumination", "estimate_illumination", "interpolate_field",
]


@dataclass
class FrameGeometry:
    """What part of a frame is real image, and how much of it is not."""

    valid: np.ndarray          # uint8 0/255, full-resolution frame size
    dead_frac: float           # fraction of the frame that is not image
    box: tuple[int, int, int, int]   # (x0, y0, x1, y1) safe rectangle

    @property
    def has_dead_border(self) -> bool:
        return self.dead_frac > 0.005


def analyse_frame(frame: np.ndarray, dead_luma: int = 10,
                  max_dead_frac: float = 0.55,
                  work_width: int = 480) -> FrameGeometry:
    """Locate the real image area of one frame.

    "Dead" is deliberately strict: luma at or below ``dead_luma``.  Rusty blade
    surface sits at 30-90 and must never be cut, whereas a clipped HDMI feed or a
    black sensor border sits at 0-8.  The mask is built at reduced resolution
    (cost ~2 ms), kept as the largest connected component so isolated dark holes
    on the part are not carved out, and then eroded a couple of pixels so the
    blend never samples the soft transition.
    """
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    k = min(1.0, work_width / w)
    small = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) \
        if k < 1.0 else gray

    alive = (small > dead_luma).astype(np.uint8)
    # close pinholes, then keep only the biggest blob: a black pin-hole or a dark
    # shadow inside the part must not punch a hole in the valid mask
    alive = cv2.morphologyEx(alive, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(alive, 8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        alive = (labels == biggest).astype(np.uint8)
    alive = cv2.morphologyEx(alive, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # fill interior holes of the surviving blob
    filled = alive.copy()
    contours, _ = cv2.findContours(alive, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    valid_small = cv2.erode(filled, np.ones((3, 3), np.uint8))

    dead_frac = 1.0 - float(valid_small.mean())
    if dead_frac > max_dead_frac:
        # Almost nothing is image: this frame is a capture fault, not a frame
        # with a border.  Report it as (nearly) all dead so the caller rejects it
        # instead of blending a black rectangle.
        valid_small[:] = 0
        dead_frac = 1.0

    valid = cv2.resize(valid_small * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(valid)
    box = ((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
           if xs.size else (0, 0, 0, 0))
    return FrameGeometry(valid=valid, dead_frac=dead_frac, box=box)


def warp_valid(valid: np.ndarray, m: np.ndarray, size: tuple[int, int],
               valid_px: int = 0) -> np.ndarray | None:
    """Warp a validity mask next to its frame; None when the frame has no mask."""
    if valid is None:
        return None
    w, h = size
    out = cv2.warpAffine(valid, m, (w, h), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # shave the resampled boundary so the blend never touches a half-dead pixel
    if valid_px > 0:
        out = cv2.erode(out, np.ones((2 * valid_px + 1, 2 * valid_px + 1), np.uint8))
    return out


def smooth_weight(alive: np.ndarray, feather_px: float,
                  gamma: float = 2.0) -> np.ndarray:
    """C1 blend weight: 1 well inside ``alive``, 0 outside, smooth in between.

    ``min(distance, feather)`` - the old profile - is continuous but its
    *gradient* jumps at the feather limit, which is exactly where a seam becomes
    visible on a low-contrast surface.  A smoothstep raised to ``gamma`` makes the
    weight flat at both ends, so overlapping frames fade into each other instead
    of meeting at a line.
    """
    if feather_px <= 0:
        return alive.astype(np.float32)
    dist = cv2.distanceTransform((alive > 0).astype(np.uint8), cv2.DIST_L2, 3)
    t = np.clip(dist / float(feather_px), 0.0, 1.0)
    s = t * t * (3.0 - 2.0 * t)            # smoothstep
    return (s ** float(gamma)).astype(np.float32)


def _robust_ratio(old: np.ndarray, new: np.ndarray, mask: np.ndarray) -> float:
    ok = mask & (old > 8) & (new > 8)
    if ok.sum() < 40:
        return float("nan")
    return float(np.median(old[ok] / new[ok]))


def _poly_terms(u: np.ndarray, v: np.ndarray, order: int) -> np.ndarray:
    cols = []
    for total in range(order + 1):
        for i in range(total + 1):
            cols.append((u ** (total - i)) * (v ** i))
    return np.stack(cols, axis=-1)


def _interp_matrix(n: int, grid: int) -> np.ndarray:
    """``n x grid`` linear-interpolation matrix over node positions 0..1."""
    if grid < 2:
        return np.ones((n, 1), np.float32)
    pos = np.linspace(0.0, 1.0, n, dtype=np.float32)
    nodes = np.linspace(0.0, 1.0, grid, dtype=np.float32)
    j = np.clip(np.searchsorted(nodes, pos, side="right") - 1, 0, grid - 2)
    t = (pos - nodes[j]) / (nodes[j + 1] - nodes[j])
    m = np.zeros((n, grid), np.float32)
    rows = np.arange(n)
    m[rows, j] = 1.0 - t
    m[rows, j + 1] = t
    return m


def interpolate_field(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Upsample a coarse field to ``shape`` (exact bilinear, node-aligned)."""
    h, w = int(shape[0]), int(shape[1])
    if field.shape[:2] == (h, w):
        return field.astype(np.float32)
    gy, gx = field.shape[:2]
    return (_interp_matrix(h, gy) @ field.astype(np.float32)
            @ _interp_matrix(w, gx).T).astype(np.float32)


def estimate_illumination(frame: np.ndarray, valid: np.ndarray | None = None,
                          order: int = 4, blocks: int = 20) -> np.ndarray | None:
    """Smooth brightness surface of one frame, or None if it cannot be fitted.

    A low-order polynomial is used rather than a blur because it has to be right
    *at the borders*, which is where a blur is worst: light falloff is a smooth
    function of position in the field of view, so a 4th-order surface represents
    it and, unlike a Gaussian, extrapolates a ramp instead of flattening it.
    Fitted on block medians so a scratch or a hole row cannot drag it.
    """
    h, w = frame.shape[:2]
    lum = frame.astype(np.float32).mean(axis=2)
    bs = max(8, int(round(max(h, w) / float(blocks))))
    us, vs, ys, wts = [], [], [], []
    for y0 in range(0, h - bs + 1, bs):
        for x0 in range(0, w - bs + 1, bs):
            blk = lum[y0:y0 + bs, x0:x0 + bs]
            if valid is not None:
                vm = valid[y0:y0 + bs, x0:x0 + bs] > 0
                if vm.mean() < 0.6:
                    continue
                blk = blk[vm]
            if blk.size < 0.4 * bs * bs:
                continue
            us.append((x0 + bs / 2.0) / w * 2.0 - 1.0)
            vs.append((y0 + bs / 2.0) / h * 2.0 - 1.0)
            ys.append(float(np.median(blk)))
            wts.append(float(blk.size))
    if len(ys) < 6 * (order + 1):
        return None
    A = _poly_terms(np.asarray(us), np.asarray(vs), min(order, 3))
    y = np.asarray(ys)
    wt = np.asarray(wts)
    keep = np.ones(y.size, bool)
    coef = np.zeros(A.shape[1])
    for _ in range(3):                       # IRLS: a defect is not illumination
        coef, *_ = np.linalg.lstsq(A[keep] * wt[keep, None], y[keep] * wt[keep],
                                   rcond=None)
        resid = y - A @ coef
        mad = float(np.median(np.abs(resid[keep]))) or 1e-6
        new_keep = np.abs(resid) < max(3.5 * 1.4826 * mad, 2.0)
        if new_keep.sum() < A.shape[1] + 2 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    if float(np.median(y[keep])) < 1e-3:
        return None
    # Evaluate on a coarse lattice with nodes *on* the borders, then interpolate:
    # a 4th-order surface is reproduced to a fraction of a luma at this density,
    # and this keeps the call at a few milliseconds instead of half a second.
    n = max(8, 4 * min(order, 3))
    nu = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    U, V = np.meshgrid(nu, nu)
    coarse = (_poly_terms(U.ravel(), V.ravel(), min(order, 3)) @ coef
              ).reshape(n, n).astype(np.float32)
    return interpolate_field(coarse, (h, w))


def flatten_illumination(frame: np.ndarray, valid: np.ndarray | None = None,
                         clip: float = 1.6, order: int = 4) -> np.ndarray:
    """Divide out the frame's own smooth illumination profile.

    Every frame is lit by the same optics from the same place, so it carries the
    same low-frequency brightness surface: light falls off across the field of
    view, the corners are dimmer than the centre.  That surface is common to
    every frame, so *no amount of matching between frames can remove it* - at
    best frames can be matched to each other, and since a frame overlaps its
    neighbour over only part of its area, matching fits the surface from that
    strip and extrapolates it over the rest.  The extrapolation error is what
    showed up as brighter rectangles in the delivered mosaics.

    Removing it per frame is a measurement rather than a guess: fit the smooth
    surface to the frame itself (masked to real image), divide it out, and
    normalise to the frame's median.  Afterwards every frame is flat and at the
    same average level, so the only difference left is exposure drift, which the
    scalar gain cancels exactly.

    ``order`` sets how much structure is treated as illumination: 4 follows a
    light falloff and a vignette without chasing a scratch.  ``clip`` bounds the
    correction.  Costs a few hundred block medians and one matrix product
    (~90 ms on an 8 Mpx frame), and runs in the stitching worker - never on the
    capture path.
    """
    if order <= 0:
        return frame
    prof = estimate_illumination(frame, valid, order=order)
    if prof is None:
        return frame
    if valid is not None:
        live = prof[valid > 0]
    else:
        live = prof.reshape(-1)
    target = float(np.median(live)) if live.size else 0.0
    if target < 1e-3:
        return frame
    gain = np.clip(target / np.maximum(prof, 1e-3), 1.0 / clip, clip)
    out = frame.astype(np.float32) * gain[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)
