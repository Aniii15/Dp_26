"""Scan geometry: turning the two step commands into the overlap the stitcher needs.

Why this module exists
----------------------
A serpentine sweep moves in two directions and the operator picks both blindly:

* **Y** - a linear actuator driven by PWM for a fixed *time* (``1200 ms``).
  Time is not distance.  Measured on the recorded scan, the same command moved
  the part between **211 and 669 px** (frame height 1080): a 3.2x spread,
  caused by supply sag, friction and load.  The stitch quality is decided by how
  much the frames *overlap*, so this is the one number that must be controlled
  and the one number that was not measurable at the time.
* **X** - a stepper given a *distance* (``0.4 mm`` = 320 steps), but nobody
  converts that to overlap.  Measured, one column advance moved the image
  **~363 px of a 1920 px frame**, i.e. **81 % overlap**: every column re-imaged
  4/5 of the previous one.  The sweep therefore takes ~4x more columns than the
  stitching can justify.

The camera is the missing encoder
---------------------------------
No axis has a position readout, so the loop looks unclosable.  It is not: the
stitcher measures the real image displacement of every accepted frame
(``Decision.dx/dy``, in full-resolution pixels).  That is a direct measurement
of what the motor actually did, it is free, and it is already being computed.

Two things fall out of it:

1. **A scale calibration, for free.**  One X move of known step count moves the
   image by a measured number of pixels, so ``px_per_mm = dx_px / dx_mm``.  That
   converts any overlap target into a command, and any command into an expected
   overlap - previously unknown, which is why 0.4 mm was arbitrary.
2. **A closed loop.**  Y travel is proportional to pulse time, so
   ``new_time = old_time * target_dy / measured_dy`` converges; the same
   measurement tells us when the actuator has drifted and by how much.

Everything here is pure arithmetic on measured numbers: no Tk, no camera, no
state beyond an EWMA, so it can be unit-tested against recorded data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

X_STEPS_PER_MM = 800.0
"""Stepper steps per millimetre of X travel (main3.X_STEPS_PER_MM)."""


def _ewma(prev: float | None, value: float, alpha: float) -> float:
    if prev is None or not np.isfinite(prev):
        return float(value)
    return float(prev + alpha * (float(value) - prev))


@dataclass
class ScanGeometry:
    """Commands for both sweep axes, and what they actually achieve.

    ``y_time_s`` and ``x_step_mm`` are what get sent to the hardware.  The
    ``measured_*`` fields are filled in from the stitcher and are what the
    operator should actually look at; ``advice()`` turns them into a sentence.
    """

    # what we command
    y_speed: int = 70
    y_time_s: float = 1.20
    x_step_mm: float = 0.40

    # what we want
    target_y_overlap: float = 0.65
    target_x_overlap: float = 0.50
    """Kept higher than the Y target on purpose: the horizontal overlap is the
    *only* link between columns, so it carries all of the accumulated drift.  A
    column costs one X move either way - the saving is in having fewer of them,
    which 50 % already delivers (~2.3x fewer on a 4 mm strip)."""
    auto_tune: bool = True
    max_adjust: float = 0.25
    """Largest correction applied per observation (fraction).  Keeps one bad
    registration - a locked-on hole row - from doubling the step size."""

    # what we measured (px, in captured-frame pixels)
    px_per_mm: float | None = None
    measured_y_px: float | None = None
    measured_x_px: float | None = None
    y_samples: int = 0
    x_samples: int = 0
    _log: list = field(default_factory=list, repr=False)

    # ── measurement ─────────────────────────────────────────────────────────
    def observe(self, axis: str, dx: float, dy: float,
                frame_shape: tuple[int, int], moved_mm: float | None = None,
                alpha: float = 0.35) -> None:
        """Feed one measured frame-to-frame displacement.

        ``axis`` is the *commanded* axis ("Y"/"X"); ``dx, dy`` are the measured
        image displacement in captured pixels; ``moved_mm`` is the commanded X
        distance, needed the first time so the pixel scale can be calibrated.
        """
        if not np.all(np.isfinite([dx, dy])):
            return
        if axis == "Y":
            # the step is the displacement *along* Y; sideways drift is not it
            step = abs(float(dy))
            if step < 1.0 or step > frame_shape[0]:
                return                      # duplicate, or not one step at all
            self.measured_y_px = _ewma(self.measured_y_px, step, alpha)
            self.y_samples += 1
        elif axis == "X":
            step = abs(float(dx))
            if step < 1.0 or step > frame_shape[1]:
                return
            self.measured_x_px = _ewma(self.measured_x_px, step, alpha)
            self.x_samples += 1
            if moved_mm and moved_mm > 0:
                ppm = step / float(moved_mm)
                # 1920 px spanning 0.5-6 mm is the plausible range for this rig;
                # anything outside is a mis-registration, not a calibration
                if 150.0 < ppm < 4000.0:
                    self.px_per_mm = _ewma(self.px_per_mm, ppm, 0.5)

    # ── what the numbers mean ───────────────────────────────────────────────
    def y_overlap(self, frame_h: int) -> float | None:
        if self.measured_y_px is None:
            return None
        return float(np.clip(1.0 - self.measured_y_px / max(frame_h, 1), 0.0, 1.0))

    def x_overlap(self, frame_w: int) -> float | None:
        if self.measured_x_px is None:
            return None
        return float(np.clip(1.0 - self.measured_x_px / max(frame_w, 1), 0.0, 1.0))

    def columns_needed(self, blade_width_px: float, frame_w: int,
                       overlap: float | None = None) -> int:
        """How many columns a strip of this width needs at the given overlap."""
        ov = self.target_x_overlap if overlap is None else overlap
        advance = max(1.0, frame_w * (1.0 - ov))
        return int(np.ceil(max(blade_width_px - frame_w, 0) / advance)) + 1

    # ── what to change ──────────────────────────────────────────────────────
    def suggested_y_time(self, frame_h: int) -> float | None:
        """Actuator pulse time that would achieve ``target_y_overlap``."""
        if self.measured_y_px is None or self.measured_y_px < 1.0:
            return None
        want = max(1.0, frame_h * (1.0 - self.target_y_overlap))
        ratio = want / self.measured_y_px
        ratio = float(np.clip(ratio, 1.0 - self.max_adjust, 1.0 + self.max_adjust))
        return float(np.clip(self.y_time_s * ratio, 0.15, 6.0))

    def suggested_x_step_mm(self, frame_w: int) -> float | None:
        """X travel that would achieve ``target_x_overlap``."""
        if self.px_per_mm is None:
            return None
        want_px = max(1.0, frame_w * (1.0 - self.target_x_overlap))
        return float(np.clip(want_px / self.px_per_mm, 0.02, 20.0))

    def apply(self, frame_shape: tuple[int, int]) -> dict:
        """Adopt the suggestions.  Returns what changed, for the log."""
        changes = {}
        if not self.auto_tune:
            return changes
        h, w = frame_shape[:2]
        t = self.suggested_y_time(h)
        if t is not None and abs(t - self.y_time_s) > 0.005:
            changes["y_time_s"] = (round(self.y_time_s, 2), round(t, 2))
            self.y_time_s = t
        x = self.suggested_x_step_mm(w)
        if x is not None and abs(x - self.x_step_mm) > 0.005:
            changes["x_step_mm"] = (round(self.x_step_mm, 2), round(x, 2))
            self.x_step_mm = x
        return changes

    # ── operator-facing ─────────────────────────────────────────────────────
    def advice(self, frame_shape: tuple[int, int]) -> str:
        h, w = frame_shape[:2]
        parts = []
        ov_y = self.y_overlap(h)
        if ov_y is None:
            parts.append("Y overlap: not measured yet")
        else:
            parts.append(f"Y step {self.measured_y_px:.0f} px → {ov_y:.0%} overlap "
                         f"(want {self.target_y_overlap:.0%})")
        ov_x = self.x_overlap(w)
        if ov_x is not None:
            parts.append(f"X step {self.measured_x_px:.0f} px → {ov_x:.0%} "
                         f"(want {self.target_x_overlap:.0%})")
        if self.px_per_mm:
            parts.append(f"{self.px_per_mm:.0f} px/mm "
                         f"({w / self.px_per_mm:.1f} mm field)")
        return "  ·  ".join(parts)

    def describe(self) -> str:
        """One line for the log at the start of a column."""
        bits = [f"Y {self.y_time_s:.2f}s @ PWM {self.y_speed}",
                f"X {self.x_step_mm:.2f} mm"]
        if self.px_per_mm:
            bits.append(f"scale {self.px_per_mm:.0f} px/mm")
        if self.measured_y_px:
            bits.append(f"measured Y {self.measured_y_px:.0f} px")
        if self.measured_x_px:
            bits.append(f"measured X {self.measured_x_px:.0f} px")
        return " | ".join(bits)
