"""Streaming mosaic accumulator.

Differences from ``blade_stitch.compose()``:

* No batch pass: each accepted frame is blended into a growing canvas the
  moment it is registered, so a partial mosaic exists after every few frames
  (``flush_every``).  A cancel, crash or power loss can no longer destroy the
  whole scan - the old script ran 16m42s over the full folder and then died
  with ``Mosaic canvas exceeds safety limit`` without writing a single image.
* All per-frame work is limited to the frame's destination bounding box
  (typically 2 MP) instead of the whole canvas (which grows to tens of MP), and
  the exposure gain is estimated on a subsampled grid.  Measured: ~40 ms/frame
  instead of 200-440 ms.
* Feathered weights come from a distance transform of the frame mask (same
  seam quality as before) but are accumulated, not recomputed over all frames.
* Disconnected frames (motor jump, column change with no overlap) are parked in
  a separate band with a visible gap instead of corrupting the mosaic through a
  bogus transform.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

GROW_MARGIN = 256


@dataclass
class AddResult:
    new_area: float          # fraction of the frame that was previously empty
    gain: float
    canvas_shape: tuple[int, int]


class IncrementalMosaic:
    """Accumulates frames with global 3x3 poses (full-res frame -> mosaic px)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._acc: np.ndarray | None = None      # HxWx3 float32
        self._wsum: np.ndarray | None = None     # HxW float32
        self._origin = np.array([0.0, 0.0])      # mosaic coord of canvas (0,0)
        self._lowq: np.ndarray | None = None     # pixels blended from soft frames
        self._placed = 0
        self._disconnected = 0

    # ── canvas bookkeeping ──────────────────────────────────────────────────
    @property
    def count(self) -> int:
        return self._placed

    @property
    def disconnected(self) -> int:
        return self._disconnected

    def bbox(self) -> tuple[int, int, int, int] | None:
        """Content bounding box in mosaic coordinates."""
        if self._wsum is None:
            return None
        covered = self._wsum > 0
        if not covered.any():
            return None
        cols = np.where(covered.any(axis=0))[0]
        rows = np.where(covered.any(axis=1))[0]
        return (int(cols[0] + self._origin[0]), int(rows[0] + self._origin[1]),
                int(cols[-1] + 1 + self._origin[0]), int(rows[-1] + 1 + self._origin[1]))

    def _ensure(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Grow the canvas so that mosaic-space box (x0,y0,x1,y1) fits."""
        if self._wsum is None:
            w = int(x1 - x0) + 2 * GROW_MARGIN
            h = int(y1 - y0) + 2 * GROW_MARGIN
            self._check(w, h)
            self._acc = np.zeros((h, w, 3), np.float32)
            self._wsum = np.zeros((h, w), np.float32)
            self._lowq = np.zeros((h, w), np.uint8)
            self._origin = np.array([x0 - GROW_MARGIN, y0 - GROW_MARGIN], float)
            return

        h, w = self._wsum.shape
        cx0, cy0 = self._origin
        cx1, cy1 = cx0 + w, cy0 + h
        def need(pad):
            return int(np.ceil(pad)) + GROW_MARGIN if pad > 0 else 0
        pad_l = need(cx0 - x0)
        pad_t = need(cy0 - y0)
        pad_r = need(x1 - cx1)
        pad_b = need(y1 - cy1)
        if not any((pad_l, pad_t, pad_r, pad_b)):
            return
        new_w, new_h = w + pad_l + pad_r, h + pad_t + pad_b
        self._check(new_w, new_h)
        acc = np.zeros((new_h, new_w, 3), np.float32)
        wsum = np.zeros((new_h, new_w), np.float32)
        lowq = np.zeros((new_h, new_w), np.uint8)
        acc[pad_t:pad_t + h, pad_l:pad_l + w] = self._acc
        wsum[pad_t:pad_t + h, pad_l:pad_l + w] = self._wsum
        lowq[pad_t:pad_t + h, pad_l:pad_l + w] = self._lowq
        self._acc, self._wsum, self._lowq = acc, wsum, lowq
        self._origin = np.array([cx0 - pad_l, cy0 - pad_t], float)

    def _check(self, w: int, h: int) -> None:
        if w * h > self.cfg.max_canvas_px:
            raise MemoryError(
                f"mosaic canvas would reach {w}x{h} px; registration has "
                "drifted. Inspect the frame log before raising max_canvas_px.")

    # ── placement ───────────────────────────────────────────────────────────
    def _roi(self, frame_shape: tuple[int, int], to_mosaic: np.ndarray):
        """Destination ROI in canvas pixels plus the crop-adjusted transform."""
        h, w = frame_shape[:2]
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]],
                           float) @ to_mosaic.T
        xs, ys = corners[:, 0], corners[:, 1]
        px0, py0 = int(np.floor(xs.min())), int(np.floor(ys.min()))
        px1, py1 = int(np.ceil(xs.max())), int(np.ceil(ys.max()))
        self._ensure(px0, py0, px1, py1)
        # ROI in canvas coordinates
        cx0 = px0 - int(round(self._origin[0]))
        cy0 = py0 - int(round(self._origin[1]))
        cx1 = px1 - int(round(self._origin[0]))
        cy1 = py1 - int(round(self._origin[1]))
        H, W = self._wsum.shape
        cx0, cy0 = max(0, cx0), max(0, cy0)
        cx1, cy1 = min(W, cx1), min(H, cy1)
        # transform: mosaic -> frame-local ROI pixel
        crop = np.array([[1, 0, -(self._origin[0] + cx0)],
                         [0, 1, -(self._origin[1] + cy0)],
                         [0, 0, 1]])
        m = (crop @ to_mosaic)[:2].astype(np.float32)
        return (slice(cy0, cy1), slice(cx0, cx1)), (cx1 - cx0, cy1 - cy0), m

    def _warp(self, frame: np.ndarray, m: np.ndarray, size: tuple[int, int]):
        w, h = size
        img = cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
        mask = cv2.warpAffine(np.ones(frame.shape[:2], np.uint8), m, (w, h),
                              flags=cv2.INTER_NEAREST) > 0
        return img, mask

    # ── blending ────────────────────────────────────────────────────────────
    def _weight(self, mask: np.ndarray) -> np.ndarray:
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        return (np.minimum(dist, float(self.cfg.feather_px)) * mask).astype(np.float32)

    def _exposure_gain(self, warped: np.ndarray, mask: np.ndarray, roi) -> float:
        """Robust luma ratio against the already-blended canvas (subsampled).

        The median is computed on a 1/16 pixel subset: full-resolution medians
        cost ~110 ms/frame and give an identical gain.
        """
        sub = 4
        ws = self._wsum[roi][::sub, ::sub]
        known = ws > 0
        overlap = mask[::sub, ::sub] & known
        if overlap.sum() < 200:
            return 1.0
        old_y = self._acc[roi][::sub, ::sub].mean(axis=2) / np.maximum(ws, 1e-3)
        new_y = warped[::sub, ::sub].astype(np.float32).mean(axis=2)
        o, n = old_y[overlap], new_y[overlap]
        ok = (o > 8) & (n > 8)
        if ok.sum() < 100:
            return 1.0
        lo, hi = self.cfg.exposure_gain_clip
        return float(np.clip(np.median(o[ok] / n[ok]), lo, hi))

    def _scaled(self, frame: np.ndarray, to_mosaic: np.ndarray):
        """Apply ``accum_scale`` to a frame and its pose."""
        s = self.cfg.accum_scale
        if s == 1.0:
            return frame, to_mosaic
        small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        inv = np.array([[1 / s, 0, 0], [0, 1 / s, 0], [0, 0, 1]], float)
        return small, to_mosaic @ inv

    def add(self, frame: np.ndarray, to_mosaic: np.ndarray,
            empty_only: bool = False, low_quality: bool = False) -> AddResult:
        """Blend a frame.  ``empty_only`` fills canvas holes without touching
        already-covered pixels (used for soft frames), and ``low_quality``
        records those pixels in the QC mask."""
        frame, to_mosaic = self._scaled(frame, to_mosaic)
        roi, size, m = self._roi(frame.shape, to_mosaic)
        warped, mask = self._warp(frame, m, size)
        known = self._wsum[roi] > 0
        total = float(mask.sum())
        new_area = float((mask & ~known).sum()) / total if total else 0.0
        gain = self._exposure_gain(warped, mask, roi)
        if empty_only:
            mask = mask & ~known
            if not mask.any():
                return AddResult(0.0, gain, self._wsum.shape)
        w = self._weight(mask)
        if gain != 1.0:
            warped = np.clip(warped.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        # Blend only inside the frame's ROI, keeping temporaries frame-sized.
        acc_roi = self._acc[roi]
        acc_roi += warped.astype(np.float32) * w[:, :, None]
        self._wsum[roi] += w
        if low_quality:
            self._lowq[roi][mask] = 255
        self._placed += 1
        return AddResult(new_area, gain, self._wsum.shape)

    def probe_new_area(self, frame: np.ndarray, to_mosaic: np.ndarray) -> float:
        """Fraction of ``frame`` landing on canvas area not yet covered.

        Rejects duplicate frames (stalled motor, repeated capture) without
        touching the accumulator.
        """
        if self._wsum is None:
            return 1.0
        roi, size, m = self._roi(frame.shape, to_mosaic)
        _, mask = self._warp(frame, m, size)
        total = float(mask.sum())
        if total <= 0:
            return 0.0
        return float((mask & (self._wsum[roi] <= 0)).sum()) / total

    def add_disconnected(self, frame: np.ndarray) -> tuple[AddResult, np.ndarray]:
        """Place a frame that could not be registered, to the right with a gap."""
        box = self.bbox()
        x = float(box[2] + 64) if box else 0.0
        y = float(box[1]) if box else 0.0
        to_mosaic = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], float)
        self._disconnected += 1
        return self.add(frame, to_mosaic), to_mosaic

    # ── output ──────────────────────────────────────────────────────────────
    def render(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (mosaic, coverage mask, low-quality mask), cropped to content."""
        if self._wsum is None:
            z = np.zeros((1, 1), np.uint8)
            return np.zeros((1, 1, 3), np.uint8), z, z.copy()
        valid = self._wsum > 0
        out = np.zeros(self._acc.shape, np.uint8)
        out[valid] = np.clip(self._acc[valid] / self._wsum[valid, None],
                             0, 255).astype(np.uint8)
        x0, y0, x1, y1 = self.bbox()
        cx, cy = int(round(self._origin[0])), int(round(self._origin[1]))
        sl = (slice(y0 - cy, y1 - cy), slice(x0 - cx, x1 - cx))
        mosaic, coverage = out[sl], (valid[sl] * 255).astype(np.uint8)
        lowq = self._lowq[sl]
        s = self.cfg.output_scale
        if s != 1.0:
            mosaic = cv2.resize(mosaic, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            coverage = cv2.resize(coverage, None, fx=s, fy=s,
                                  interpolation=cv2.INTER_NEAREST)
            lowq = cv2.resize(lowq, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        return mosaic, coverage, lowq

    def save(self, prefix: str, quality: int = 95) -> dict:
        mosaic, coverage, lowq = self.render()
        paths = {
            "jpg": f"{prefix}_mosaic.jpg",
            "png": f"{prefix}_mosaic.png",
            "coverage": f"{prefix}_coverage.png",
            "low_quality": f"{prefix}_low_quality.png",
        }
        cv2.imwrite(paths["jpg"], mosaic, [cv2.IMWRITE_JPEG_QUALITY, quality])
        cv2.imwrite(paths["png"], mosaic, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(paths["coverage"], coverage)
        cv2.imwrite(paths["low_quality"], lowq)
        paths["shape"] = mosaic.shape
        paths["low_quality_pct"] = round(float((lowq > 0).mean() * 100), 2)
        return paths
