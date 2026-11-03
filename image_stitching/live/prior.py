"""What the stage can physically do, learned while the scan runs.

A serpentine sweep is not a sequence of purely vertical then purely horizontal
jumps: the rig is not square to the camera, the actuator pushes the part at a
slight angle, and the X axis creeps a few rows while Y sweeps.  Registering frame
N+1 against frame N *unconstrained* therefore lets a mismatched feature pair drag
the transform off by a few degrees, or onto the next hole of the periodic row,
and each such error is blended in permanently.

The machine, however, knows a great deal that the image does not:

* which axis it just commanded (so the expected motion is one of two families,
  not "anywhere"),
* that consecutive frames are *neighbours* - one Y step apart (or one X step at
  a column turn), never far apart,
* that the Y step is roughly constant while the sweep runs.

``StagePrior`` learns the *observed* signature of that commanded motion - the
mean displacement vector in mosaic pixels, its spread, and the rotation the rig
actually shows - and then

1. supplies the expected displacement as a registration prior (narrow search band,
   so periodic holes cannot win), and
2. vetoes a registration whose displacement disagrees with the commanded axis by
   more than a tolerance, before the frame is blended.

It is permissive while it is learning (``min_samples`` observations) and never
vetoes anything it has no evidence for.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["AxisModel", "StagePrior"]


@dataclass
class AxisModel:
    """Observed motion for one commanded axis, in mosaic pixels."""

    axis: str = ""
    samples: int = 0
    mean: np.ndarray = field(default_factory=lambda: np.zeros(2))
    var: np.ndarray = field(default_factory=lambda: np.ones(2))
    rotation_deg: float = 0.0        # direction of travel vs the image axis

    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.var, 1.0))

    def tolerance(self, rel: float = 0.35, floor: float = 25.0) -> np.ndarray:
        """How far from the mean a step may land and still be that axis's step.

        Wide enough for the real variation in an open-loop actuator (measured on
        the production scans: step swings between 13 % and 62 % overlap), narrow
        enough to exclude a hole pitch of drift (~50 px) and any X step.
        """
        return np.maximum(self.std() * 3.0, self.mean * rel).clip(min=floor)


class StagePrior:
    """Learns per-axis displacement and uses it as a registration constraint."""

    def __init__(self, min_samples: int = 3, max_samples: int = 60,
                 strict: bool = True, rel_tol: float = 0.35):
        self.models: dict[str, AxisModel] = {}
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.strict = strict
        self.rel_tol = rel_tol
        self._history: dict[str, list[np.ndarray]] = {}
        self.rejections = 0

    # ── learning ────────────────────────────────────────────────────────────
    def observe(self, axis: str | None, displacement) -> None:
        """Record the displacement actually measured for a commanded axis."""
        if not axis:
            return
        d = np.asarray(displacement, float)[:2]
        if not np.all(np.isfinite(d)):
            return
        hist = self._history.setdefault(axis, [])
        hist.append(d)
        if len(hist) > self.max_samples:
            hist.pop(0)
        m = self.models.setdefault(axis, AxisModel(axis=axis))
        arr = np.vstack(hist)
        m.samples = len(hist)
        m.mean = arr.mean(axis=0)
        m.var = arr.var(axis=0) if len(hist) > 1 else np.ones(2)
        if np.linalg.norm(m.mean) > 1e-6:
            m.rotation_deg = float(np.degrees(np.arctan2(m.mean[0], m.mean[1])))

    def known(self, axis: str | None) -> bool:
        m = self.models.get(axis or "")
        return bool(m and m.samples >= self.min_samples)

    # ── prediction ──────────────────────────────────────────────────────────
    def expected(self, axis: str | None) -> np.ndarray | None:
        """Expected displacement for one commanded step of ``axis``, or None."""
        if not self.known(axis):
            return None
        return self.models[axis].mean.copy()

    def predicted_pose_step(self, axis: str | None) -> np.ndarray | None:
        """3x3 pose increment (translation only) for one commanded step."""
        d = self.expected(axis)
        if d is None:
            return None
        t = np.eye(3)
        t[0, 2], t[1, 2] = float(d[0]), float(d[1])
        return t

    # ── veto ────────────────────────────────────────────────────────────────
    def check(self, axis: str | None, displacement) -> tuple[bool, str]:
        """Is this displacement consistent with the commanded axis?

        Returns ``(ok, reason)``.  ``ok`` is True while the prior is still
        learning, when no axis was given, or when the displacement agrees.  A
        disagreement means the registration locked onto the wrong feature - a
        periodic hole, a reflection - and must not be blended.
        """
        if not self.strict or not self.known(axis):
            return True, ""
        m = self.models[axis]
        d = np.asarray(displacement, float)[:2]
        err = np.abs(d - m.mean)
        tol = m.tolerance(self.rel_tol)
        if np.all(err <= tol):
            return True, ""
        self.rejections += 1
        return False, (f"off_axis(expected {m.mean[0]:+.0f},{m.mean[1]:+.0f} "
                       f"got {d[0]:+.0f},{d[1]:+.0f}, tol "
                       f"{tol[0]:.0f},{tol[1]:.0f})")

    # ── reporting ───────────────────────────────────────────────────────────
    def summary(self) -> dict:
        return {a: {"samples": m.samples,
                    "step": [round(float(v), 1) for v in m.mean],
                    "std": [round(float(v), 1) for v in m.std()],
                    "rotation_deg": round(m.rotation_deg, 2)}
                for a, m in self.models.items()}

    def describe(self) -> str:
        if not self.models:
            return "no axis learned yet"
        parts = []
        for axis, m in sorted(self.models.items()):
            if m.samples == 0:
                continue
            parts.append(f"{axis}: {m.mean[1]:+.0f}px/step "
                         f"(+{m.std()[1]:.0f}), rot {m.rotation_deg:+.1f}deg, "
                         f"n={m.samples}")
        return "; ".join(parts) or "no axis learned yet"
