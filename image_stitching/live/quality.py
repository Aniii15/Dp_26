"""Frame-quality assessment for live mosaic stitching.

Why not ``cv2.Laplacian(...).var()`` (what most examples use)?

Measured on the real scans in ``sample_scans/test_images``: two consecutive
frames where one is visibly out of focus and the next is sharp both score a
Laplacian variance of 13-14, because the dark vignette and the slowly varying
blade surface dominate the statistic.  The absolute number is also useless for
thresholding because the scene brightness drifts from mean 145 to 30 while the
scan walks down the part.

What works far better here:

* **Tenengrad / mean-luma²** - exposure invariant, and it cleanly separated the
  pre-autofocus frames (100-150 raw, ratio<1) from focused ones (230+).
* **Same-pixels comparison** - compare the sharpness of the incoming frame
  against the already accepted frame *inside the overlap region only*.  Both
  scores then describe identical physical surface, so the ratio is a direct
  "is this frame worse than what I already have?" test.  It flagged the
  genuinely defocused captures (ratio 0.22, 0.54, 0.60) without touching the
  good ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def sharpness(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Exposure-normalised Tenengrad focus measure.

    ``sum(|grad|²) / mean(luma)²`` - dividing by the squared mean removes the
    global brightness/contrast drift, so scores stay comparable while the scan
    moves across differently lit parts of the blade.
    """
    g = gray.astype(np.float32, copy=False)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    energy = gx * gx + gy * gy
    if mask is not None:
        if mask.sum() < 200:
            return 0.0
        m = mask.astype(bool)
        return float(energy[m].mean() / (float(g[m].mean()) ** 2 + 1e-6))
    return float(energy.mean() / (float(g.mean()) ** 2 + 1e-6))


def directional_blur(gray: np.ndarray) -> float:
    """Ratio of gradient energy across axes; ~1 sharp, >>1 smeared along one axis.

    A motor that is still moving while the frame is grabbed smears the image in
    the travel direction (vertical here), which looks sharp to an isotropic
    metric but is useless for feature matching.
    """
    g = gray.astype(np.float32, copy=False)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return float((gx * gx).mean() + 1e-6) / float((gy * gy).mean() + 1e-6)


@dataclass
class QualityReport:
    ok: bool = True
    reasons: list[str] = field(default_factory=list)
    sharpness: float = 0.0
    mean_luma: float = 0.0
    contrast: float = 0.0
    clipped_low: float = 0.0
    clipped_high: float = 0.0
    directional: float = 1.0

    def reject(self, reason: str) -> "QualityReport":
        self.ok = False
        self.reasons.append(reason)
        return self


def assess_intrinsic(frame: np.ndarray, cfg, width: int = 640) -> QualityReport:
    """Cheap checks that need no registration: exposure, contrast, sharpness.

    Runs before feature extraction so a black or badly defocused frame costs
    ~8 ms instead of a registration attempt.  Metrics are computed on a
    downscaled copy - these are all relative measures and 640 px is plenty.
    """
    gray = to_gray(frame)
    if width and gray.shape[1] > width:
        gray = cv2.resize(gray, (width, int(round(gray.shape[0] * width / gray.shape[1]))),
                          interpolation=cv2.INTER_AREA)
    g = gray.astype(np.float32)
    rep = QualityReport(
        mean_luma=float(g.mean()),
        contrast=float(g.std()),
        clipped_low=float((gray < 8).mean()),
        clipped_high=float((gray > 247).mean()),
        directional=directional_blur(gray),
    )
    rep.sharpness = sharpness(gray) * (1920 / max(gray.shape[1], 1)) ** 2
    if rep.clipped_low > cfg.max_clipped_low:
        rep.reject("underexposed")
    if rep.clipped_high > cfg.max_clipped_high:
        rep.reject("overexposed")
    if rep.contrast < cfg.min_contrast:
        rep.reject("no_contrast")
    if rep.sharpness < cfg.min_sharpness:
        rep.reject("out_of_focus")
    return rep


def focus_ratio(cur_gray: np.ndarray, prev_gray: np.ndarray,
                mask: np.ndarray) -> float:
    """Sharpness of ``cur`` relative to ``prev`` on identical pixels.

    ``mask`` marks the valid overlap in ``prev``'s coordinate frame; ``cur``
    must already be warped into that frame.
    """
    m = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8),
                  iterations=1).astype(bool)
    if m.sum() < 5000:
        return 1.0  # not enough shared area to judge; do not reject on this
    prev_s = sharpness(prev_gray, m)
    cur_s = sharpness(cur_gray, m)
    if prev_s <= 0:
        return 1.0
    return cur_s / prev_s
