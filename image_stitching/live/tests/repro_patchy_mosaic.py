#!/usr/bin/env python3
"""Reproduce - and measure - the "patched up" mosaic, on synthetic frames.

The delivered mosaics show pale rectangular blocks and hard brightness steps.
Two of the three causes can be reproduced exactly here, without a camera:

* a **clipped border**: the capture has a dead (black) strip / clipped corner on
  one side, which the stitcher used to treat as image;
* a **gradient in the illumination** that differs between frames, which a single
  scalar exposure gain cannot cancel.

Run ``python live/tests/repro_patchy_mosaic.py``.  It stitches the same scan twice
- once with the old behaviour (`scalar_gain`, no validity mask, linear feather)
and once with the new one - and prints two numbers per mosaic:

``step``   the mean absolute discontinuity across the seam between frames, i.e.
           how visible the frame boundaries are (lower is better);
``tp``     fraction of the mosaic that is "too pale" - luma far above the local
           surface - which is what the clipped-border blocks look like.

It also writes ``before.png`` / ``after.png`` next to the script.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # image_stitching/

import cv2
import numpy as np

from live.config import LiveConfig
from live.mosaic import IncrementalMosaic

FRAME_H, FRAME_W = 300, 400
STEP_Y = 210                  # 30 % overlap
COLUMNS = 2
STEP_X = 300
TILT_DEG = 0.25               # the rig is not perfectly square


def surface(seed: int = 5) -> np.ndarray:
    """Rusty, holey surface - fine texture plus the periodic hole rows."""
    rng = np.random.default_rng(seed)
    h, w = 1400, 900
    base = rng.normal(120, 26, (h, w)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 1.0)
    for y in range(40, h - 30, 46):
        for x in (w - 260, w - 170):
            cv2.ellipse(base, (x, y), (13, 7), 10, 0, 360, 55, -1)
            cv2.ellipse(base, (x, y), (13, 7), 10, 0, 360, 205, 1)
    for _ in range(200):
        x, y = rng.integers(0, w), rng.integers(0, h)
        cv2.line(base, (int(x), int(y)),
                 (int(x + rng.integers(-30, 30)), int(y + rng.integers(10, 60))),
                 float(rng.integers(60, 95)), 1)
    img = np.dstack([base * 0.92, base * 0.99, base * 1.05])
    return np.clip(img, 0, 255).astype(np.uint8)


SURF = surface()


def capture(col: int, row: int, dead_side: str = "right",
            gradient: bool = True, drift: bool = True, clip: bool = True,
            tilt: bool = True) -> tuple[np.ndarray, dict]:
    """One camera frame.

    The three things that go wrong on the real rig are separable, because they
    are *not* equally fixable:

    ``gradient``  light falls off across the field of view.  Every frame of the
                  same part is affected the same way, so no amount of matching
                  between frames can remove it - a stitcher that "fixes" it is
                  inventing data.  It is in the truth mosaic too.
    ``drift``     auto-exposure moves the overall level between frames.  This is
                  what leaves brightness steps, and it *is* removable.
    ``clip``      a dead sensor border / clipped corner.  Not image at all; it
                  must be masked out, never brightened.

    Ground truth is therefore ``gradient=True, drift=False, clip=False``: the
    part as the lens sees it, without the faults that have to be compensated.
    """
    x0, y0 = col * STEP_X, row * STEP_Y
    patch = SURF[y0:y0 + FRAME_H, x0:x0 + FRAME_W].copy()
    if patch.shape[:2] != (FRAME_H, FRAME_W):
        patch = cv2.copyMakeBorder(patch, 0, FRAME_H - patch.shape[0], 0,
                                   FRAME_W - patch.shape[1],
                                   cv2.BORDER_REPLICATE)
    # light falls off across the field of view, and the *mean* drifts as the
    # scan walks down the part (auto-exposure) - both per frame
    yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W]
    grad = 1.0 + 0.28 * (xx / FRAME_W - 0.5) + 0.10 * (yy / FRAME_H - 0.5)
    drift = 1.0 + 0.05 * np.sin(row * 0.7 + col) + 0.03 * row
    gain = np.ones((FRAME_H, FRAME_W), np.float32)
    if gradient:
        gain = gain * grad
    if drift:
        gain = gain * drift
    out = np.clip(patch.astype(np.float32) * gain[:, :, None], 0, 255)
    # the rig is not square: rotate slightly about the frame centre
    tilt_deg = TILT_DEG if tilt else 0.0
    m = cv2.getRotationMatrix2D((FRAME_W / 2, FRAME_H / 2), tilt_deg, 1.0)
    out = cv2.warpAffine(out.astype(np.uint8), m, (FRAME_W, FRAME_H),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    # clipped border: the right 18 % of the sensor returns black
    frame = out.copy()
    if clip and dead_side == "right":
        frame[:, int(FRAME_W * 0.82):] = 0
    elif clip and dead_side == "corner":
        frame[:int(FRAME_H * 0.25), int(FRAME_W * 0.75):] = 0
    pose = np.array([[np.cos(np.deg2rad(tilt_deg)), -np.sin(np.deg2rad(tilt_deg)),
                      x0 + 0.0],
                     [np.sin(np.deg2rad(tilt_deg)), np.cos(np.deg2rad(tilt_deg)),
                      y0 + 0.0], [0, 0, 1]], float)
    return frame, {"pose": pose, "col": col, "row": row}


def scan_stops():
    for c in range(COLUMNS):
        rows = 0
        while (rows + 1) * STEP_Y + FRAME_H <= SURF.shape[0]:
            yield c, rows
            rows += 1


def stitch(new_style: bool, **faults):
    """Stitch the scan; returns (mosaic, coverage, seam rows).

    ``faults`` is passed to ``capture`` (``gradient`` / ``drift`` / ``clip`` /
    ``tilt``); the default is the full set of real-rig faults.
    """
    from live.frames import analyse_frame
    mosaic = IncrementalMosaic(LiveConfig(flush_every=0, accum_scale=1.0))
    mosaic.cfg.scalar_gain = not new_style
    mosaic.cfg.flatten_illumination = new_style
    seams = []
    for col, row in scan_stops():
        frame, meta = capture(col, row, **faults)
        kw = {}
        if new_style:
            kw["valid"] = analyse_frame(frame).valid
        mosaic.add(frame, meta["pose"], **kw)
        if row:
            seams.append(row * STEP_Y)          # top edge of every later frame
    out, coverage, _lowq = mosaic.render()
    return out, coverage, seams


# ── metrics ─────────────────────────────────────────────────────────────────

def _align(mosaic: np.ndarray, cov: np.ndarray, ref: np.ndarray,
           ref_cov: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Crop ``mosaic`` so its top-left covered pixel matches ``ref``'s."""
    y0, x0 = np.argwhere(cov > 0).min(axis=0)
    ry, rx = np.argwhere(ref_cov > 0).min(axis=0)
    dy, dx = int(ry) - int(y0), int(rx) - int(x0)
    h = min(mosaic.shape[0], ref.shape[0]) - max(0, -dy)
    w = min(mosaic.shape[1], ref.shape[1]) - max(0, -dx)
    a = mosaic[max(0, dy):max(0, dy) + h, max(0, dx):max(0, dx) + w]
    b = ref[max(0, -dy):max(0, -dy) + h, max(0, -dx):max(0, -dx) + w]
    ca = cov[max(0, dy):max(0, dy) + h, max(0, dx):max(0, dx) + w]
    cb = ref_cov[max(0, -dy):max(0, -dy) + h, max(0, -dx):max(0, -dx) + w]
    return a.astype(np.float32), b.astype(np.float32), (ca > 0) & (cb > 0)


def fidelity(mosaic: np.ndarray, cov: np.ndarray, truth: np.ndarray,
             truth_cov: np.ndarray) -> tuple[float, float]:
    """Mean and 99th-percentile |error| against a clean stitch of the same part.

    This is the number that decides whether the mosaic is usable: it asks "does
    it look like the part", not "is there a gradient somewhere".
    """
    a, b, shared = _align(mosaic, cov, truth, truth_cov)
    if shared.mean() < 0.5:
        return float("nan"), float("nan")
    err = np.abs(a - b).mean(axis=2)[shared]
    return float(err.mean()), float(np.percentile(err, 99))


def seam_step(mosaic: np.ndarray, seams: list[int]) -> float:
    """Brightness step across each known frame boundary, in luma.

    The scan advances in Y, so the boundaries are horizontal lines.  For each
    one, compare the low-passed luma a little above and a little below: texture
    is averaged away, what survives is the visible edge - the "band".
    """
    g = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.blur(g, (31, 31))
    steps = []
    for y in seams:
        if y - 18 < 0 or y + 18 >= g.shape[0]:
            continue
        above = g[y - 16:y - 4].mean(axis=0)
        below = g[y + 4:y + 16].mean(axis=0)
        live = np.isfinite(above) & np.isfinite(below) & (above > 1) & (below > 1)
        if live.sum() > 50:
            steps.append(abs(float(above[live].mean() - below[live].mean())))
    return float(np.median(steps)) if steps else float("nan")


def pale_fraction(mosaic: np.ndarray, win: int = 61) -> float:
    """Fraction of pixels far brighter than their surroundings (the pale blocks)."""
    g = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local = cv2.blur(g, (win, win))
    return float(((g - local) > 45).mean() * 100)


def main() -> int:
    truth, truth_cov, seams = stitch(new_style=True, drift=False, clip=False)
    rows = []
    for name, new in (("before", False), ("after", True)):
        img, cov, _ = stitch(new_style=new)
        cv2.imwrite(str(HERE / f"{name}.png"), img)
        err_mean, err_p99 = fidelity(img, cov, truth, truth_cov)
        rows.append((name, img.shape, err_mean, err_p99,
                     seam_step(img, seams), pale_fraction(img)))
    print(f"{'':>6s}  {'size':>11s}  {'err mean':>8s}  {'err p99':>7s}  "
          f"{'seam':>5s}  {'pale':>6s}")
    for name, shape, em, e99, seam, pale in rows:
        print(f"{name:>6s}  {shape[1]:5d}x{shape[0]:<5d}  {em:8.1f}  {e99:7.1f}  "
              f"{seam:5.1f}  {pale:5.1f}%")
    (_, _, bm, bp99, bs, bpale) = rows[0]
    (_, _, am, ap99, asx, apale) = rows[1]
    print()
    print(f"error vs clean    {bm:.1f} -> {am:.1f} luma "
          f"({100 * (bm - am) / max(bm, 1e-6):+.0f}%)")
    print(f"worst 1%          {bp99:.1f} -> {ap99:.1f} luma")
    print(f"seam step         {bs:.1f} -> {asx:.1f} luma")
    print(f"pale blocks       {bpale:.1f}% -> {apale:.1f}%")
    ok = am < bm * 0.5 and asx < bs and apale < bpale
    print("\nPASS" if ok else "\nFAIL (no improvement)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
