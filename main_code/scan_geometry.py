"""Scan geometry: what the two step commands actually do, and what they should be.

The sweep moves in two directions and both were fixed constants inside
``_sweep_worker``:

* **Y** - a linear actuator driven by PWM for a fixed time (``1200 ms``).
  Time is not distance.
* **X** - a stepper given a distance (``0.4 mm`` = 320 steps), but nobody
  converted that to *overlap*, which is the only thing the stitcher cares about.

Measured on the two recorded scans (``sample_scans/``), with the
same registrar the live stitcher uses:

===============================================  ============  ============
                                                 scan A        scan B
===============================================  ============  ============
Y step, same 1200 ms command, frame = 1080 px   208-669 px    233-449 px
  -> overlap between consecutive frames         38-81 %       58-78 %
X step, 320 steps = "0.4 mm", frame = 1920 px   273-338 px    400-453 px
  -> overlap between columns                    82-86 %       76-79 %
===============================================  ============  ============

Three things follow, and they are the reason this module exists.

1. **The overlap was never known while scanning** - only reconstructable
   afterwards, from the saved frames.  The one number that decides whether a
   scan is stitcheable was invisible at capture time.
2. **X was ~4x more conservative than it needed to be.**  At 78 % overlap each
   new column re-images four fifths of the previous one; the stitcher needs a
   fraction of that.
3. **The px/mm scale is not portable between scans** (310 px vs 426 px for the
   same 320-step command).  Either the rig changed or the command was not what
   the code says - and with no record of the command in the scan folder, that
   cannot be decided after the fact.  So the geometry that produced a scan is
   now written *into* the scan folder (``geometry.json``).

Why a closed loop is possible at all: neither axis reports its position, but the
stitcher measures the true image displacement of every frame it registers.  That
is a direct measurement of what the motor did, it is free, and it was already
being logged.  Two loops use it:

* **during the scan** - the live stitcher's per-frame ``dx/dy`` adjusts the next
  column (coarse: greedy, half-resolution preview);
* **after the scan** - the global stitcher learns a displacement model per move
  type over *every* frame of a globally-solved graph
  (``GlobalStitcher.motion``); ``from_motion()`` turns those into achieved
  overlaps and a recommendation for the next scan.  This is the accurate one.

Everything here is arithmetic on measured numbers: no Tk, no camera, no I/O
except the two small JSON helpers, so it is unit-tested against recorded data.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field

import numpy as np

X_STEPS_PER_MM = 800.0
"""Stepper steps per millimetre of X travel (``main3.X_STEPS_PER_MM``)."""

GEOMETRY_FILE = "geometry.json"
"""Written into each scan folder: the commands that produced that scan."""

# Phases whose displacement is one pure move of the axis.  ``Xr_AF``/``Ycor_AF``
# are excluded: an AF frame sits part-way along the move that preceded it, so its
# displacement is not a step.
_MOVE_RE = re.compile(r"^(?P<kind>ydown|yup)_?(?P<idx>\d+)$", re.I)


def single_step(prev_tag: str | None, tag: str | None) -> str | None:
    """Axis of the single move between two *consecutive* captures, or None.

    The live stitcher reports each frame's displacement against whichever anchor
    matched best, which is usually - but not always - the previous frame.  When
    frames are dropped the displacement spans several moves and must not be fed
    to the calibration as if it were one.  Requiring the two tags to be
    consecutive captures of the same move family is what keeps the sample clean;
    the recorded scans show why it matters: an AF frame lives ~265 px into the
    Y move before it, so mixing those in inflates the spread enormously.
    """
    if not prev_tag or not tag:
        return None
    a = phase_of(prev_tag)
    b = phase_of(tag)
    if b is None:
        return None
    if b.lower().startswith("x") and a and a.lower().startswith("ycor"):
        return "X"                      # Xr_move follows Ycor_AF directly
    ma, mb = _MOVE_RE.match(a or ""), _MOVE_RE.match(b)
    if not ma or not mb:
        return None
    if ma.group("kind").lower() != mb.group("kind").lower():
        return None
    if int(mb.group("idx")) != int(ma.group("idx")) + 1:
        return None
    return "Y"


def phase_of(tag: str) -> str | None:
    """``col02_Ydown_14_155152`` -> ``Ydown_14``."""
    if not tag:
        return None
    parts = tag.split("_")
    if len(parts) >= 3 and parts[0].lower().startswith("col"):
        body = "_".join(parts[1:-1])
        return body or None
    body = "_".join(parts[:-1]) if parts[-1].isdigit() else "_".join(parts)
    return body or None


def move_of(tag: str) -> str | None:
    """Move type that happened before this frame: ``Ydown``/``Yup``/``Xr``."""
    ph = phase_of(tag)
    if ph is None:
        return None
    p = ph.lower()
    if re.match(r"y(down|up)_?\d+$", p):
        return "Ydown" if p.startswith("ydown") else "Yup"
    if p.startswith("x") and p.endswith("move"):
        return "Xr"
    return None


def _ewma(prev: float | None, value: float, alpha: float) -> float:
    if prev is None or not np.isfinite(prev):
        return float(value)
    return float(prev + alpha * (float(value) - prev))


@dataclass
class ScanGeometry:
    """Commands for both sweep axes, the calibration, and what to change.

    ``y_time_s`` / ``y_speed`` / ``x_step_mm`` are what gets sent to the hardware.
    The ``measured_*`` fields are filled from the stitcher and are what the
    operator should actually look at; :meth:`advice` turns them into one line.
    """

    # ── what we command ─────────────────────────────────────────────────────
    y_speed: int = 70
    y_time_s: float = 1.20
    x_step_mm: float = 0.40
    x_return_mm: float = 0.40
    """X travel of the return move at the end of a column.  Kept equal to
    ``x_step_mm`` unless the operator overrides it (a backlash-prone belt can
    need a longer return than advance)."""

    # ── what we want ────────────────────────────────────────────────────────
    target_y_overlap: float = 0.65
    target_x_overlap: float = 0.50
    """Horizontal overlap is kept higher than the vertical one on purpose: it is
    the *only* link between columns, so it carries all accumulated drift."""

    auto_tune: bool = True
    max_adjust: float = 0.25
    """Largest correction per observation (fraction).  One mis-registered frame
    - a locked-on cooling-hole row - must not be able to double the step."""

    # ── what we measured ────────────────────────────────────────────────────
    px_per_mm: float | None = None
    """Image pixels per millimetre of X travel, from ``x_px / x_cmd_mm``."""

    measured_y_px: float | None = None
    y_spread_px: float | None = None
    measured_x_px: float | None = None
    x_spread_px: float | None = None
    y_samples: int = 0
    x_samples: int = 0
    frame_hw: tuple[int, int] | None = None
    """(h, w) the measurements refer to - the pixel scale is per resolution."""

    # ── measurement ─────────────────────────────────────────────────────────
    def _plausible(self, step: float, axis: str, frame_shape) -> bool:
        """Is this displacement a single step of the rig, at all?"""
        if not np.isfinite(step) or step < 1.0:
            return False
        h, w = frame_shape[:2]
        limit = h if axis == "Y" else w
        # More than three quarters of a frame in one step leaves under a quarter
        # overlap, which no stitcher survives - so it is a mis-registration (a
        # cooling-hole row locking onto its neighbour), not a step.  Measured
        # steps on this rig are 208-669 px of 1080, i.e. at most 62 %.
        if step >= 0.75 * limit:
            return False
        expect = self.measured_y_px if axis == "Y" else self.measured_x_px
        if expect and expect > 1.0:
            # reject anything more than 3x away from what this axis is doing;
            # that is a skipped frame or a wrong match, not a step
            if not (expect / 3.0 < step < expect * 3.0):
                return False
        return True

    def observe(self, axis: str, dx: float, dy: float,
                frame_shape: tuple[int, int], moved_mm: float | None = None,
                alpha: float = 0.35) -> bool:
        """Feed one measured single-step displacement.  True if it was used.

        ``axis`` is the commanded axis of the move that produced the frame,
        ``dx, dy`` the measured image displacement in captured pixels, and
        ``moved_mm`` the commanded X distance of that move (needed to calibrate
        the pixel scale).  Implausible samples are refused rather than smoothed
        in - an EWMA cannot recover from a 500 px outlier, it just takes a while.
        """
        d = np.array([dx, dy], float)
        if not np.all(np.isfinite(d)):
            return False
        if axis == "Y":
            step = abs(float(d[1]))
            if not self._plausible(step, "Y", frame_shape):
                return False
            self.measured_y_px = _ewma(self.measured_y_px, step, alpha)
            dev = abs(step - self.measured_y_px)
            self.y_spread_px = _ewma(self.y_spread_px, dev, alpha)
            self.y_samples += 1
        elif axis == "X":
            step = abs(float(d[0]))
            if not self._plausible(step, "X", frame_shape):
                return False
            self.measured_x_px = _ewma(self.measured_x_px, step, alpha)
            self.x_spread_px = _ewma(
                self.x_spread_px, abs(step - self.measured_x_px), alpha)
            self.x_samples += 1
            mm = moved_mm or self.x_step_mm
            if mm and mm > 0:
                ppm = step / float(mm)
                # 1920 px across 0.5-6 mm is the plausible range for this rig;
                # anything outside it is a mis-registration, not a calibration
                if 150.0 < ppm < 4000.0:
                    self.px_per_mm = _ewma(self.px_per_mm, ppm, alpha)
        else:
            return False
        self.frame_hw = (int(frame_shape[0]), int(frame_shape[1]))
        return True

    def absorb_motion(self, motion: dict, frame_shape: tuple[int, int],
                      x_cmd_mm: float | None = None,
                      alpha: float = 0.6) -> dict:
        """Calibrate from the global stitcher's per-move-type models.

        ``motion`` is ``GlobalStitcher.motion``: ``{move: (median_xy, sigma,
        count)}``.  This is the authoritative measurement of a finished scan -
        every frame, a globally consistent solve - so it is trusted more
        (``alpha`` 0.6) than a single live observation.  Returns what was
        learned, for the log.
        """
        got: dict[str, object] = {}
        if not motion or not frame_shape:
            return got
        y_steps = [abs(float(np.asarray(m)[1])) for mv, (m, _s, _c) in motion.items()
                   if mv in ("Ydown", "Yup")]
        if y_steps:
            step = float(np.median(y_steps))
            if self._plausible(step, "Y", frame_shape):
                self.measured_y_px = _ewma(self.measured_y_px, step, alpha)
                # the spread is the solver's own per-move sigma: the honest
                # step-to-step band, not the difference between move *types*
                sig = [float(s_) for mv, (_m, s_, _c) in motion.items()
                       if mv in ("Ydown", "Yup")]
                if sig:
                    prev = self.y_spread_px
                    self.y_spread_px = _ewma(prev, float(np.median(sig)), alpha)
                self.y_samples += sum(int(c) for mv, (_m, _s, c) in motion.items()
                                      if mv in ("Ydown", "Yup"))
                got["y_px"] = round(self.measured_y_px, 1)
        xs = [abs(float(np.asarray(m)[0])) for mv, (m, _s, _c) in motion.items()
              if mv == "Xr"]
        if xs:
            step = float(np.median(xs))
            if self._plausible(step, "X", frame_shape):
                self.measured_x_px = _ewma(self.measured_x_px, step, alpha)
                sig = [float(s_) for mv, (_m, s_, _c) in motion.items()
                       if mv == "Xr"]
                if sig:
                    self.x_spread_px = _ewma(self.x_spread_px,
                                             float(np.median(sig)), alpha)
                mm = x_cmd_mm or self.x_step_mm
                if mm and mm > 0 and 150.0 < step / mm < 4000.0:
                    self.px_per_mm = _ewma(self.px_per_mm, step / mm, alpha)
                self.x_samples += sum(int(c) for mv, (_m, _s, c) in motion.items()
                                      if mv == "Xr")
                got["x_px"] = round(self.measured_x_px, 1)
                got["px_per_mm"] = round(self.px_per_mm or 0.0, 1)
        if got:
            self.frame_hw = (int(frame_shape[0]), int(frame_shape[1]))
        return got

    # ── what the numbers mean ───────────────────────────────────────────────
    def y_overlap(self, frame_h: int | None = None) -> float | None:
        h = frame_h or (self.frame_hw[0] if self.frame_hw else None)
        if self.measured_y_px is None or not h:
            return None
        return float(np.clip(1.0 - self.measured_y_px / h, 0.0, 1.0))

    def x_overlap(self, frame_w: int | None = None) -> float | None:
        w = frame_w or (self.frame_hw[1] if self.frame_hw else None)
        if self.measured_x_px is None or not w:
            return None
        return float(np.clip(1.0 - self.measured_x_px / w, 0.0, 1.0))

    def y_overlap_range(self, frame_h: int) -> tuple[float, float] | None:
        """Worst and best overlap implied by the measured spread.

        The actuator is open loop, so a *target* overlap is a target for the
        mean; this is the honest band, and the reason a target that is perfectly
        safe on average can still produce a gap.
        """
        if self.measured_y_px is None:
            return None
        s = self.y_spread_px or 0.0
        hi = 1.0 - max(1.0, self.measured_y_px - s) / frame_h
        lo = 1.0 - (self.measured_y_px + s) / frame_h
        return float(np.clip(lo, 0, 1)), float(np.clip(hi, 0, 1))

    def columns_needed(self, blade_w_px: float, frame_w: int,
                       overlap: float | None = None) -> int:
        ov = self.target_x_overlap if overlap is None else overlap
        advance = max(1.0, frame_w * (1.0 - ov))
        return int(np.ceil(max(blade_w_px - frame_w, 0) / advance)) + 1

    # ── what to change ──────────────────────────────────────────────────────
    def suggested_y_time(self, frame_h: int | None = None) -> float | None:
        h = frame_h or (self.frame_hw[0] if self.frame_hw else None)
        if self.measured_y_px is None or self.measured_y_px < 1.0 or not h:
            return None
        want = max(1.0, h * (1.0 - self.target_y_overlap))
        ratio = float(np.clip(want / self.measured_y_px,
                              1.0 - self.max_adjust, 1.0 + self.max_adjust))
        return float(np.clip(self.y_time_s * ratio, 0.15, 6.0))

    def suggested_x_step_mm(self, frame_w: int | None = None) -> float | None:
        w = frame_w or (self.frame_hw[1] if self.frame_hw else None)
        if self.px_per_mm is None or not w:
            return None
        want_px = max(1.0, w * (1.0 - self.target_x_overlap))
        return float(np.clip(want_px / self.px_per_mm, 0.02, 20.0))

    def apply(self, frame_shape: tuple[int, int] | None = None) -> dict:
        """Adopt the suggestions.  Returns ``{name: (old, new)}`` for the log."""
        changes: dict[str, tuple[float, float]] = {}
        if not self.auto_tune or not frame_shape:
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
            self.x_return_mm = x
        return changes

    # ── operator-facing ─────────────────────────────────────────────────────
    def advice(self, frame_shape: tuple[int, int] | None = None) -> str:
        shape = frame_shape or self.frame_hw
        if not shape:
            return "waiting for the first camera frame"
        h, w = shape[:2]
        parts = []
        ov_y = self.y_overlap(h)
        if ov_y is None:
            parts.append("Y step: not measured yet")
        else:
            band = self.y_overlap_range(h)
            band_s = (f" ({band[0]:.0%}-{band[1]:.0%} step to step)"
                      if band else "")
            parts.append(f"Y {self.measured_y_px:.0f} px -> {ov_y:.0%} overlap"
                         f", want {self.target_y_overlap:.0%}{band_s}")
        ov_x = self.x_overlap(w)
        if ov_x is None:
            parts.append("X step: not measured yet")
        else:
            parts.append(f"X {self.measured_x_px:.0f} px -> {ov_x:.0%} "
                         f"overlap, want {self.target_x_overlap:.0%}")
        if self.px_per_mm:
            parts.append(f"{self.px_per_mm:.0f} px/mm "
                         f"({w / self.px_per_mm:.1f} mm field)")
        return "  ·  ".join(parts)

    def describe(self) -> str:
        bits = [f"Y {self.y_time_s:.2f}s @ PWM {self.y_speed}",
                f"X {self.x_step_mm:.2f} mm"]
        if self.px_per_mm:
            bits.append(f"scale {self.px_per_mm:.0f} px/mm")
        if self.measured_y_px:
            bits.append(f"measured Y {self.measured_y_px:.0f} px")
        if self.measured_x_px:
            bits.append(f"measured X {self.measured_x_px:.0f} px")
        return " | ".join(bits)

    def recommendation(self, frame_shape: tuple[int, int] | None = None) -> str:
        """What to use for the *next* scan, in the units the UI takes."""
        shape = frame_shape or self.frame_hw
        if not shape:
            return "no measurement yet"
        h, w = shape[:2]
        t = self.suggested_y_time(h)
        x = self.suggested_x_step_mm(w)
        if t is None and x is None:
            return "no measurement yet"
        bits = []
        if t is not None:
            bits.append(f"Y pulse {t:.2f} s")
        if x is not None:
            bits.append(f"X step {x:.2f} mm")
        return ", ".join(bits)

    # ── persistence ─────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        d = asdict(self)
        if self.frame_hw:
            d["frame_hw"] = list(self.frame_hw)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ScanGeometry":
        known = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in (d or {}).items() if k in known}
        fh = kw.get("frame_hw")
        if isinstance(fh, (list, tuple)) and len(fh) == 2:
            kw["frame_hw"] = (int(fh[0]), int(fh[1]))
        return cls(**kw)

    def save(self, path: str) -> bool:
        """Persist the calibration so it survives a restart.  Best effort."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception:                      # noqa: BLE001 - never break a scan
            return False

    @classmethod
    def load(cls, path: str) -> "ScanGeometry | None":
        try:
            with open(path) as f:
                return cls.from_dict(json.load(f))
        except Exception:                      # noqa: BLE001
            return None


def write_scan_geometry(folder: str, geo: ScanGeometry,
                        extra: dict | None = None) -> str | None:
    """Record the commands that produced this scan *in the scan folder*.

    Without this the pixel scale cannot be attributed afterwards: the two
    recorded scans disagree by 37 % on px/0.4 mm with identical code, and with
    no record of the command there is no way to tell a changed rig from a changed
    constant.  ``capture_log.csv`` already records the order; this records the
    geometry.
    """
    try:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, GEOMETRY_FILE)
        d = geo.to_dict()
        if extra:
            d.update(extra)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        return path
    except Exception:                          # noqa: BLE001
        return None


def read_scan_geometry(folder: str) -> ScanGeometry | None:
    return ScanGeometry.load(os.path.join(folder, GEOMETRY_FILE))


def geometry_lines(geo: ScanGeometry, motion: dict | None = None,
                   frame_shape: tuple[int, int] | None = None,
                   x_cmd_mm: float | None = None) -> list[str]:
    """Report section: achieved steps, overlaps, and what to change."""
    shape = frame_shape or geo.frame_hw
    lines = ["", "Sweep geometry (measured, not commanded):",
             f"  Y pulse {geo.y_time_s:.2f} s @ PWM {geo.y_speed}, "
             f"X step {(x_cmd_mm or geo.x_step_mm):.2f} mm"]
    if not shape:
        lines.append("  no frame size recorded - cannot express overlap")
        return lines
    h, w = shape[:2]
    if motion:
        # Only the moves that are one clean step of an axis: "AF:..." entries
        # inherit the move before them (an AF frame sits part-way along it) and
        # the Ycor/return moves are corrections, not steps.  They are counted,
        # not interpreted, or the table reads as ten different step sizes.
        steps = {mv: v for mv, v in motion.items()
                 if mv in ("Ydown", "Yup", "Xr")}
        other = len(motion) - len(steps)
        lines.append("  per move type (from the global solve, frame-centre px):")
        for mv, (med, sig, cnt) in sorted(steps.items()):
            m = np.asarray(med, float)
            ov = 1.0 - abs(m[1]) / h if mv.startswith("Y") else 1.0 - abs(m[0]) / w
            lines.append(f"    {mv:10s} dx={m[0]:7.1f} dy={m[1]:7.1f} px  "
                         f"spread {sig:5.1f}  n={cnt:<4d} overlap {ov:5.1%}")
        if other:
            lines.append(f"    ({other} further move types: AF frames and "
                         "corrections, not steps)")
    ov_y = geo.y_overlap(h)
    ov_x = geo.x_overlap(w)
    if ov_y is not None:
        band = geo.y_overlap_range(h)
        extra = (f"  (per-step band {band[0]:.0%}-{band[1]:.0%})" if band else "")
        lines.append(f"  Y overlap {ov_y:.1%} of the frame height "
                     f"(target {geo.target_y_overlap:.0%}){extra}")
    if ov_x is not None:
        lines.append(f"  X overlap {ov_x:.1%} of the frame width "
                     f"(target {geo.target_x_overlap:.0%})")
    if geo.px_per_mm:
        field_mm = w / geo.px_per_mm
        lines.append(f"  scale {geo.px_per_mm:.0f} px/mm -> "
                     f"{field_mm:.2f} mm across the frame")
        if ov_x is not None:
            # a representative part width, so the numbers mean something: how
            # many columns the same part needs now versus at the target overlap
            part_mm = max(2.0 * field_mm, 1.5)
            part_px = part_mm * geo.px_per_mm
            cols_now = geo.columns_needed(part_px, w, ov_x)
            cols_tgt = geo.columns_needed(part_px, w, geo.target_x_overlap)
            saving = (1.0 - cols_tgt / cols_now) if cols_now else 0.0
            lines.append(f"  a {part_mm:.1f} mm wide part needs {cols_now} "
                         f"columns now, {cols_tgt} at the target "
                         f"({saving:.0%} fewer moves)")
    rec = geo.recommendation(shape)
    lines.append(f"  recommended for the next scan: {rec}")
    return lines
