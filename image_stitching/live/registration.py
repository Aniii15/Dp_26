"""Bounded feature registration between consecutive scan frames.

The original ``blade_stitch.estimate()`` seeded ``findTransformECC`` with
``cv2.phaseCorrelate`` and accepted whatever came back if ``cc > 0.12``.  On
these frames that is unsafe in two ways:

1. The blade carries a *periodic* row of cooling holes.  Phase correlation and
   ECC both lock onto whichever hole period is closest to the initial guess, so
   the estimator can be off by one full hole pitch (~50 px) while still showing
   a correlation of 0.98 - a silent, invisible misregistration that shows up as
   doubled holes in the mosaic.
2. ``cc`` measures photometric agreement after warping, not geometric
   correctness.  It stays high even when the transform is wrong by a pitch.

Measured per-step travel on the real scans is 210-490 px (frame height 1080),
i.e. the actuator is open loop and the overlap swings between 13% and 62%.

This module therefore:

* extracts ORB features (verified: 40-130 ms/frame, hundreds of inliers),
* constrains matches to a *prior* displacement band when the machine knows
  which way it commanded the axis to move,
* removes period ambiguity by voting on the translation, then refining,
* verifies the result photometrically (NCC over the real overlap) and demands
  that the implied motion is physically plausible.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Low-texture pairs (dark columns, heavy vignetting) can only be matched with a
# looser ratio test: measured 6-10 inliers, yet the resulting model verifies at
# NCC 0.89-0.97 over 67% overlap.  The NCC gate is what keeps these honest.
LOOSE_MIN_MATCHES = 6
LOOSE_MIN_INLIERS = 5


@dataclass
class Registration:
    """``affine`` maps *current* frame coordinates into *previous* frame ones."""

    affine: np.ndarray          # 2x3, in work-scale pixels
    scale: float                # work-scale -> full-resolution factor
    ncc: float
    overlap: float
    inliers: int
    matches: int
    method: str
    prior_ok: bool = True
    ncc_hp: float = 0.0
    """NCC of the *band-passed* images over the overlap.  Plain NCC is
    dominated by vignetting and the slow brightness gradient of the blade:
    measured on scan_20261105_200702, a registration deliberately shifted by
    77 px still scores plain NCC 0.90 (median), while its band-passed NCC is
    <= 0.07 (p90).  Correct matches score 0.36-0.85 (median).  This is the
    number that actually says whether a registration is right."""

    @property
    def shift(self) -> np.ndarray:
        """(dx, dy) translation in full-resolution pixels."""
        return self.affine[:, 2] / self.scale

    def full_res(self) -> np.ndarray:
        """2x3 affine in full-resolution pixel coordinates."""
        a = self.affine.astype(np.float64).copy()
        a[:, 2] /= self.scale
        return a

    @property
    def score(self) -> float:
        q = self.ncc_hp if self.ncc_hp > 0 else self.ncc
        return float(q) * float(min(1.0, self.inliers / 200.0))


def highpass(img: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    f = img.astype(np.float32)
    return f - cv2.GaussianBlur(f, (0, 0), sigma)


class Registrar:
    """Caches features so a frame can be matched against several anchors."""

    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.feature == "sift" and hasattr(cv2, "SIFT_create"):
            self._det = cv2.SIFT_create(nfeatures=cfg.n_features,
                                        contrastThreshold=0.01)
            self._norm = cv2.NORM_L2
            self._ratio = 0.75
        else:
            self._det = cv2.ORB_create(cfg.n_features, scaleFactor=1.2,
                                       nlevels=8, fastThreshold=7)
            self._norm = cv2.NORM_HAMMING
            self._ratio = cfg.ratio_test
        self._matcher = cv2.BFMatcher(self._norm)
        self._feat: dict[str, tuple] = {}
        self._gray: dict[str, np.ndarray] = {}
        self._scale: dict[str, float] = {}
        self._valid: dict[str, np.ndarray] = {}

    # ── preparation ─────────────────────────────────────────────────────────
    def prepare(self, token: str, frame: np.ndarray,
                valid: np.ndarray | None = None) -> np.ndarray:
        """Scale a full-resolution frame to work size and cache features.

        ``valid`` is an optional full-resolution 0..1 map of pixels that are
        real image (see ``lens.py``).  Without it the black, optically clipped
        frame corners take part in matching: their curved edge is a strong,
        *static* feature that matches itself at zero displacement in every
        pair, and the black area inflates the NCC check.
        """
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = self.cfg.work_width / gray.shape[1]
        if scale < 1.0:
            work = cv2.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            work = gray
        self._gray[token] = work
        self._scale[token] = scale
        vmask = None
        if valid is not None:
            v = cv2.resize(np.asarray(valid, np.float32),
                           (work.shape[1], work.shape[0]),
                           interpolation=cv2.INTER_AREA)
            vmask = (v > 0.98).astype(np.uint8)
            # keep descriptors well away from the optical edge
            vmask = cv2.erode(vmask, np.ones((9, 9), np.uint8))
            self._valid[token] = vmask
        if token not in self._feat:
            self._feat[token] = self._det.detectAndCompute(work, vmask)
        return work

    def forget(self, tokens) -> None:
        for t in list(tokens):
            self._feat.pop(t, None)
            self._gray.pop(t, None)
            self._scale.pop(t, None)
            self._valid.pop(t, None)

    def work_gray(self, token: str) -> np.ndarray | None:
        return self._gray.get(token)

    def scale(self, token: str) -> float:
        return self._scale.get(token, 1.0)

    def keep_only(self, tokens: set) -> None:
        stale = [t for t in list(self._gray) if t not in tokens]
        if stale:
            self.forget(stale)

    def cached_count(self) -> int:
        return len(self._feat)

    # ── registration ────────────────────────────────────────────────────────
    def register(self, prev_token: str, cur_token: str,
                 prior: np.ndarray | None = None) -> Registration | None:
        """Estimate the transform that maps ``cur`` into ``prev`` coordinates.

        Two descriptor-match sets are computed (strict ratio test, and a looser
        one for low-texture pairs), each fitted both unbounded and restricted to
        the commanded motion.  Every resulting candidate is verified
        photometrically and the best one wins, so a candidate only needs enough
        inliers to be *fit*, not to be *believed* - the NCC decides.
        """
        f1 = self._feat.get(prev_token) or (None, None)
        f2 = self._feat.get(cur_token) or (None, None)
        k1, d1 = f1
        k2, d2 = f2
        a = self._gray.get(prev_token)
        b = self._gray.get(cur_token)
        if d1 is None or d2 is None or a is None or b is None:
            return None
        if len(k1) < 8 or len(k2) < 8:
            return None

        pairs = self._matcher.knnMatch(d2, d1, k=2)
        strict = [m for m, n in pairs if m.distance < self._ratio * n.distance]
        loose_ratio = min(0.9, self._ratio + self.cfg.loose_ratio_retry)
        loose = [m for m, n in pairs
                 if m.distance < loose_ratio * n.distance] if self.cfg.loose_ratio_retry > 0 else []

        attempts: list[tuple[list, int]] = []
        if len(strict) >= self.cfg.min_inliers:
            attempts.append((strict, self.cfg.min_inliers))
        if len(loose) >= LOOSE_MIN_MATCHES:
            attempts.append((loose, LOOSE_MIN_INLIERS))
        if not attempts:
            return None

        n_matches = len(attempts[0][0])
        candidates: list[tuple[np.ndarray, int, str, np.ndarray | None]] = []
        for good, floor in attempts:
            src = np.float32([k2[m.queryIdx].pt for m in good])
            dst = np.float32([k1[m.trainIdx].pt for m in good])
            model, inl = self._fit(src, dst, min_inliers=floor)
            if model is not None:
                candidates.append((model, int(inl.sum()), "orb", None))
            if prior is None:
                continue
            d_pred = np.asarray(prior, np.float64)
            tol = self.cfg.prior_tol_px + self.cfg.prior_tol_rel * float(np.linalg.norm(d_pred))
            keep = np.linalg.norm((dst - src) - d_pred, axis=1) < tol
            if keep.sum() >= floor:
                model, inl = self._fit(src[keep], dst[keep], min_inliers=floor)
                if model is not None:
                    candidates.append((model, int(inl.sum()), "orb+prior", d_pred))
        if not candidates:
            return None

        h, w = a.shape[:2]
        max_step = self.cfg.max_step_frac * max(h, w)
        va, vb = self._valid.get(prev_token), self._valid.get(cur_token)
        ha, hb = highpass(a), highpass(b)
        best = None                       # (hp, method, model, overlap, inliers, ncc)
        for model, inl_n, method, d_pred in candidates:
            if np.linalg.norm(model[:, 2]) > max_step:
                continue                  # physically impossible for one step
            ncc, overlap = self._verify(a, b, model, va, vb)
            hp, _ = self._verify(ha, hb, model, va, vb, centered=True)
            if best is None or self._better(hp, model, d_pred, best):
                best = (hp, method, model, overlap, inl_n, ncc)
        if best is None:
            return None
        hp, method, model, overlap, inl_n, ncc = best

        # Optional sub-pixel polish, kept only when it improves the match.
        if ncc < self.cfg.polish_below_ncc:
            refined = self._polish(a, b, model)
            if refined is not None:
                hp2, overlap2 = self._verify(ha, hb, refined, va, vb, centered=True)
                if hp2 > hp + 0.01:
                    ncc2, _ = self._verify(a, b, refined, va, vb)
                    model, ncc, hp, overlap, method = (refined, ncc2, hp2, overlap2,
                                                       method + "+ecc")

        return Registration(model, self._scale.get(prev_token, 1.0), ncc, overlap,
                            inl_n, n_matches, method, prior is not None, hp)

    def _better(self, ncc: float, model: np.ndarray, pred: np.ndarray | None,
                best) -> bool:
        """Rank candidates by band-passed NCC; on a near-tie prefer the
        commanded motion."""
        best_ncc, _, best_model = best[0], best[1], best[2]
        if ncc > best_ncc + 0.02:
            return True
        if pred is not None and abs(ncc - best_ncc) <= 0.02:
            return float(np.linalg.norm(model[:, 2] - pred)) < \
                float(np.linalg.norm(best_model[:, 2] - pred))
        return ncc > best_ncc

    # ── internals ───────────────────────────────────────────────────────────
    def _fit(self, src, dst, min_inliers: int | None = None):
        model, inl = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC,
            ransacReprojThreshold=self.cfg.ransac_px,
            maxIters=4000, confidence=0.999)
        need = self.cfg.min_inliers if min_inliers is None else min_inliers
        if model is None or inl is None or int(inl.sum()) < need:
            return None, None
        return model.astype(np.float32), inl

    def _polish(self, a, b, model):
        """Sub-pixel ECC refinement at reduced resolution (~60 ms, not 400)."""
        pw = self.cfg.polish_width
        if pw <= 0:
            return None
        k = min(1.0, pw / a.shape[1])
        if k < 1.0:
            sa = cv2.resize(a, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            sb = cv2.resize(b, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        else:
            k, sa, sb = 1.0, a, b
        seed = model.copy()
        seed[:, 2] *= k
        try:
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-5)
            cc, w = cv2.findTransformECC(sa, sb, seed, cv2.MOTION_EUCLIDEAN,
                                         crit, None, 3)
        except cv2.error:
            return None
        if cc <= 0.5:
            return None
        out = w.astype(np.float32)
        out[:, 2] /= k
        return out

    def _verify(self, a, b, model, va=None, vb=None, centered=False):
        h, w = a.shape[:2]
        warped = cv2.warpAffine(b, model, (w, h), flags=cv2.INTER_LINEAR)
        src = np.ones(b.shape, np.uint8) if vb is None else vb
        mask = cv2.warpAffine(src, model, (w, h), flags=cv2.INTER_NEAREST) > 0
        if va is not None:
            mask &= va > 0
        overlap = float(mask.mean())
        m = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        if m.sum() < 2000:
            return 0.0, overlap
        x = warped[m].astype(np.float64)
        y = a[m].astype(np.float64)
        if x.std() < 1e-3 or y.std() < 1e-3:
            return 0.0, overlap
        if centered:          # band-passed input: zero-mean by construction
            ncc = float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))
        else:
            ncc = float(np.corrcoef(x, y)[0, 1])
        return (ncc if np.isfinite(ncc) else 0.0), overlap
