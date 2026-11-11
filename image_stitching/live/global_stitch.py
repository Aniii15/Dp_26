"""Global (after-the-scan) blade stitcher.

The live stitcher has to decide about each frame the moment it arrives, so it
throws frames away and blends greedily.  That is fine for a preview but it is
what made the delivered mosaics look like a patchwork.  This module re-stitches
a finished scan folder with everything known at once:

1. **Every frame is used.**  Nothing is rejected.  A blurred or duplicate frame
   simply loses the per-pixel competition wherever a better observation of the
   same spot exists, and fills the spot where it is the only observation.  A
   frame that cannot be registered at all is still placed (by the motion
   model), flagged, and only used where nothing else covers the spot.

2. **The scan geometry is a constraint, not an assumption.**  Consecutive
   frames are neighbours (vertical inside a pass, horizontal across an X move).
   Each move type (``Ydown``, ``Yup``, ``Xr``, ``AF`` ...) gets a displacement
   model *learned from this scan's own registrations* - a mean vector plus a
   spread - so a Y move that also drifts sideways, or an autofocus that shifts
   and rotates the view, is modelled as it really is.  These models:
     * bound the feature search (kills the cooling-hole period ambiguity),
     * seed a blur-tolerant coarse matcher for frames ORB cannot match,
     * keep the graph connected when nothing else can (dead reckoning),
   while every overlapping pair - not just consecutive ones - adds a measured
   constraint.  Down-pass and up-pass strips overlap ~80 % horizontally, so
   there are hundreds of loop closures.

3. **Poses are solved globally** (robust least squares over the whole graph,
   rotation + scale first, then translation) instead of being chained frame to
   frame, so drift and one bad match cannot shear the rest of the mosaic.
   Autofocus frames come back rotated by up to ~1 deg and scaled by up to
   ~1 % (focus breathing); a translation-only model leaves 15-20 px seams.

4. **Blending has no blocks.**
     * The lens model (``lens.py``) removes the optically clipped corners and
       flattens the vignetting before anything is blended.
     * Each mosaic pixel is assigned to ONE frame: the sharpest observation of
       that spot, ties broken by how central the pixel is in the frame.  The
       sharpness map is continuous per pixel, not a 6x4 tile grid, and the
       assignment is smoothed (Potts MRF) so seams are few and long.
     * Exposure is equalised by a global gain solve over all overlaps.
     * Seams are hidden by multi-band (Laplacian pyramid) blending: low
       frequencies (brightness) blend over a wide band, fine detail switches
       over a few pixels, so there is neither a visible edge nor ghosting.

Entry point: ``stitch_folder(path, out_prefix)``; CLI: ``run_global.py``.
"""
from __future__ import annotations

import csv
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import LiveConfig
from .lens import LensModel, estimate_lens
from .registration import Registrar

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ──────────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GlobalConfig:
    # registration
    work_width: int = 1000
    n_features: int = 4000
    min_ncc: float = 0.55
    min_overlap: float = 0.10
    skip_edges: int = 2
    """Also register frame i with i-2 .. i-skip_edges (redundancy against one
    bad frame in the chain)."""
    cross_pairs: int = 5
    """Per frame, extra non-consecutive neighbours registered after the first
    solve (other pass / other column).  These are the loop closures."""
    cross_min_overlap: float = 0.30
    min_ncc_hp: float = 0.15
    """Band-passed NCC a feature registration must reach.  Plain NCC cannot
    tell a right match from one that is off by 77 px on this surface (0.90
    either way); band-passed NCC is <= 0.07 for wrong and 0.36-0.85 for right
    matches (see ``Registration.ncc_hp``)."""
    coarse_min_ncc: float = 0.20
    """Band-passed NCC the blur-tolerant template matcher must reach (it is
    used only where ORB found nothing)."""
    # pose graph
    prior_sigma_floor: float = 30.0
    """Smallest std-dev (px) a learned move model may claim.  The actuator is
    open loop, so a Y step is really 210-490 px: the prior is soft."""
    unknown_move_sigma: float = 600.0
    robust_px: float = 8.0
    outlier_px: float = 45.0
    irls_iters: int = 10
    solve_rotation: bool = True
    # compositing
    label_scale: int = 8
    """Seam selection runs on a grid this many times coarser than the mosaic."""
    focus_power: float = 3.0
    """How strongly a sharper observation beats a more central one."""
    seam_smoothness: float = 1.0
    """Potts weight of the seam MRF, in log-score units: a frame may keep a
    cell it would otherwise lose by up to e^1 = 2.7x in score if its
    neighbourhood is already its own.  Fewer, longer seams."""
    consensus_sigma: float = 12.0
    """Grey levels (on the 8x-reduced, lightly blurred label grid) at which a
    frame starts to count as disagreeing with the other views of a spot."""
    consensus_weight: float = 2.0
    """Log-score penalty per ``consensus_sigma`` of disagreement."""
    unlinked_penalty: float = 0.1
    """Score multiplier for frames only placed by the motion model: they fill
    holes but never displace a registered frame."""
    sharp_sigma: float = 40.0
    num_bands: int = 6
    output_scale: float = 1.0
    max_canvas_px: float = 160e6
    jpeg_quality: int = 95
    low_detail_ratio: float = 0.12
    threads: int = 0
    lens_frames: int = 80
    """Frames sampled for the lens / vignetting estimate."""


# ──────────────────────────────────────────────────────────────────────────────
#  Scan bookkeeping
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ScanFrame:
    index: int
    path: Path
    tag: str
    column: int
    phase: str
    move: str
    t: float


@dataclass
class Edge:
    """Relative pose of frame j with respect to frame i.

    ``m``      where j's centre lands in i's pixel coordinates, minus i's centre
               (so for a pure translation it is the displacement), full px.
    ``theta``  rotation of j relative to i (rad), ``logs`` log scale.
    """
    i: int
    j: int
    m: np.ndarray
    sigma: float
    kind: str               # seq | skip | cross | coarse | prior
    theta: float = 0.0
    logs: float = 0.0
    rot_sigma: float = 0.0  # 0 = this edge says nothing about rotation/scale
    ncc: float = 0.0
    inliers: int = 0
    overlap: float = 0.0
    d: np.ndarray | None = None      # m rotated into the mosaic frame
    weight: float = 0.0
    residual: float = 0.0
    active: bool = True

    @property
    def measured(self) -> bool:
        return self.kind != "prior"


_NAME = re.compile(r"^col(\d+)_(.+?)_(\d{6})(?:_(\d+))?$", re.I)
_PASS_ORDER = {"ycor_up": 1, "ycor_af": 2, "xr_move": 3, "xr_af": 4,
               "ycor_down": 6, "yup_af": 7, "xr2_move": 8, "xr2_af": 9}


def move_of(phase: str) -> str:
    """Kind of motion that happened *before* this frame was captured."""
    p = phase.lower()
    if p.endswith("_af") or p == "af":
        return "AF"
    if p.startswith("ycor_up"):
        return "Ycor_up"
    if p.startswith("ycor_down"):
        return "Ycor_down"
    if p.startswith("ydown"):
        return "Ydown"
    if p.startswith("yup"):
        return "Yup"
    if p.startswith("x") and p.endswith("move"):
        return "Xr"
    return "other"


def _phase_rank(phase: str) -> tuple[int, float]:
    p = phase.lower()
    m = re.match(r"(ydown|yup)_?(\d+)(_af)?$", p)
    if m:
        return (0 if m.group(1) == "ydown" else 5,
                int(m.group(2)) + (0.5 if m.group(3) else 0.0))
    return (_PASS_ORDER.get(p, 10), 0.0)


def _contextualise(frames: list[ScanFrame]) -> list[ScanFrame]:
    """Autofocus frames inherit the move that preceded them.

    Measured: an AF frame sits ~265 px further along the previous Y move than
    the frame saved just before it (the stage was still moving when that
    frame was grabbed - see the capture notes in README).  So "AF after a
    Ydown" and "AF after a Yup" are two different, each very consistent,
    displacements; lumping them into one "AF" model gave a 560 px spread."""
    last = "start"
    for f in frames:
        if f.move == "AF":
            f.move = f"AF:{last}"
        else:
            last = f.move
    return frames


def list_scan(folder: Path) -> list[ScanFrame]:
    """All frames of a scan folder in capture order.

    Uses ``capture_log.csv`` when the sweep wrote one (exact order and move
    type); otherwise reconstructs both from the file names.
    """
    folder = Path(folder)
    log = folder / "capture_log.csv"
    out: list[ScanFrame] = []
    if log.exists():
        with open(log, newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            p = folder / (r.get("file") or "")
            if not r.get("file") or not p.exists():
                continue
            m = _NAME.match(p.stem)
            col = int(r.get("column") or (m.group(1) if m else 0) or 0)
            phase = r.get("phase") or (m.group(2) if m else p.stem)
            out.append(ScanFrame(len(out), p, p.stem, col, phase,
                                 r.get("move") or move_of(phase),
                                 float(r.get("t") or len(out))))
        if out:
            return _contextualise(out)

    keyed = []
    for p in folder.iterdir():
        if p.suffix.lower() not in IMG_EXT:
            continue
        m = _NAME.match(p.stem)
        if m:
            col, phase, hhmmss = int(m.group(1)), m.group(2), m.group(3)
            t = int(hhmmss[:2]) * 3600 + int(hhmmss[2:4]) * 60 + int(hhmmss[4:])
            seq = int(m.group(4)) if m.group(4) else -1
        else:
            col, phase, t, seq = 0, p.stem, 0, -1
        keyed.append(((col, seq if seq >= 0 else t, _phase_rank(phase), p.name),
                      p, col, phase, t))
    keyed.sort(key=lambda k: k[0])
    for _, p, col, phase, t in keyed:
        out.append(ScanFrame(len(out), p, p.stem, col, phase, move_of(phase),
                             float(t)))
    return _contextualise(out)


# ──────────────────────────────────────────────────────────────────────────────
#  Registration (thread-parallel, shared feature cache)
# ──────────────────────────────────────────────────────────────────────────────

def _highpass(img: np.ndarray, sigma: float) -> np.ndarray:
    f = img.astype(np.float32)
    return f - cv2.GaussianBlur(f, (0, 0), sigma)


class _Registry:
    def __init__(self, cfg: GlobalConfig, lens: LensModel):
        self.cfg = cfg
        self.lens = lens
        self.lcfg = LiveConfig(work_width=cfg.work_width,
                               n_features=cfg.n_features,
                               min_ncc=cfg.min_ncc)
        self._feat: dict = {}
        self._gray: dict = {}
        self._scale: dict = {}
        self._valid: dict = {}
        self._tl = threading.local()

    def _reg(self) -> Registrar:
        r = getattr(self._tl, "reg", None)
        if r is None:
            r = Registrar(self.lcfg)
            r._feat, r._gray, r._scale, r._valid = (
                self._feat, self._gray, self._scale, self._valid)
            self._tl.reg = r
        return r

    def prepare(self, token: str, frame: np.ndarray) -> None:
        self._reg().prepare(token, frame, valid=self.lens.valid)

    def register(self, prev: str, cur: str, prior_full, shape):
        """Feature registration.  Returns (m, theta, logs, Registration)."""
        reg = self._reg()
        k = reg.scale(prev)
        r = reg.register(prev, cur, None if prior_full is None
                         else np.asarray(prior_full, float) * k)
        if r is None:
            return None
        A = r.full_res()
        h, w = shape
        c = np.array([w / 2.0, h / 2.0])
        m = A[:, :2] @ c + A[:, 2] - c
        L = A[:, :2]
        theta = math.atan2(L[1, 0], L[0, 0])
        logs = 0.5 * math.log(max(abs(np.linalg.det(L)), 1e-9))
        return m, theta, logs, r

    def coarse(self, prev: str, cur: str, prior_full, radius_full: float):
        """Blur-tolerant translation search (template matching on band-passed
        images), for frames too soft for ORB.  Returns (m_full, ncc, overlap)."""
        reg = self._reg()
        a, b = reg.work_gray(prev), reg.work_gray(cur)
        if a is None or b is None:
            return None
        k = reg.scale(prev)
        va = self._valid.get(prev)
        vb = self._valid.get(cur)
        p = np.asarray(prior_full, float) * k               # work px
        est = None
        for q, sig, rad in ((4, 6.0, radius_full * k / 4 + 6), (1, 5.0, 10.0)):
            if q > 1:
                aq = cv2.resize(a, None, fx=1 / q, fy=1 / q, interpolation=cv2.INTER_AREA)
                bq = cv2.resize(b, None, fx=1 / q, fy=1 / q, interpolation=cv2.INTER_AREA)
                vaq = None if va is None else cv2.resize(va, (aq.shape[1], aq.shape[0]),
                                                          interpolation=cv2.INTER_NEAREST)
                vbq = None if vb is None else cv2.resize(vb, (bq.shape[1], bq.shape[0]),
                                                          interpolation=cv2.INTER_NEAREST)
                center = p / q
            else:
                aq, bq, vaq, vbq = a, b, va, vb
                center = est
            ha, wa = aq.shape
            fa, fb = _highpass(aq, sig), _highpass(bq, sig)
            if vaq is not None:
                fa[vaq == 0] = 0
            if vbq is not None:
                fb[vbq == 0] = 0
            r = int(math.ceil(rad))
            cx, cy = int(round(center[0])), int(round(center[1]))
            # template: part of b that lands inside a for every shift in window
            tx0 = max(0, -cx + r)
            ty0 = max(0, -cy + r)
            tx1 = min(wa, wa - cx - r)
            ty1 = min(ha, ha - cy - r)
            if tx1 - tx0 < 24 or ty1 - ty0 < 16:
                return None
            tmpl = fb[ty0:ty1, tx0:tx1]
            sx0, sy0 = tx0 + cx - r, ty0 + cy - r
            search = fa[sy0:sy0 + (ty1 - ty0) + 2 * r, sx0:sx0 + (tx1 - tx0) + 2 * r]
            if search.shape[0] < tmpl.shape[0] or search.shape[1] < tmpl.shape[1]:
                return None
            res = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
            _, _, _, loc = cv2.minMaxLoc(res)
            sub = np.array([loc[0], loc[1]], float)
            # parabolic sub-pixel refinement
            for ax in (0, 1):
                i0 = loc[ax]
                lim = res.shape[1 - ax]
                if 0 < i0 < lim - 1:
                    if ax == 0:
                        l, c0, rr = res[loc[1], i0 - 1], res[loc[1], i0], res[loc[1], i0 + 1]
                    else:
                        l, c0, rr = res[i0 - 1, loc[0]], res[i0, loc[0]], res[i0 + 1, loc[0]]
                    den = l - 2 * c0 + rr
                    if abs(den) > 1e-9:
                        sub[ax] += 0.5 * (l - rr) / den
            shift = np.array([cx - r + sub[0], cy - r + sub[1]])
            est = shift * q
        model = np.float32([[1, 0, est[0]], [0, 1, est[1]]])
        ncc, overlap = reg._verify(_highpass(a, 6.0), _highpass(b, 6.0), model,
                                   va, vb, centered=True)
        return est / k, ncc, overlap


# ──────────────────────────────────────────────────────────────────────────────
#  Pose graph
# ──────────────────────────────────────────────────────────────────────────────

def _irls(n: int, I, J, D, W0, robust: np.ndarray, c: float, iters: int,
          outlier: float | None = None):
    """Robust LS for x_j - x_i = D (rows of D are k-vectors).  ``robust`` marks
    measured rows (re-weighted, may be switched off); others keep W0."""
    D = np.asarray(D, float).reshape(len(I), -1)
    W = W0.astype(float).copy()
    active = np.ones(len(I), bool)
    X = np.zeros((n, D.shape[1]))
    res = np.zeros(len(I))
    for it in range(iters + 1):
        w = W * active
        Hm = np.zeros((n, n))
        np.add.at(Hm, (I, I), w)
        np.add.at(Hm, (J, J), w)
        np.add.at(Hm, (I, J), -w)
        np.add.at(Hm, (J, I), -w)
        b = np.zeros((n, D.shape[1]))
        np.add.at(b, J, w[:, None] * D)
        np.add.at(b, I, -w[:, None] * D)
        Hm[0, 0] += 1e6 * max(1.0, float(np.max(W0)))
        Hm += np.eye(n) * 1e-12 * max(1.0, float(np.max(W0)))
        X = np.linalg.solve(Hm, b)
        res = np.linalg.norm(X[J] - X[I] - D, axis=1)
        if it == iters:
            break
        W = np.where(robust, W0 / (1.0 + (res / c) ** 2), W0)
        if outlier is not None and it == iters - 3:
            active &= ~(robust & (res > outlier))
    return X, res, active, W


def solve_graph(n: int, edges: list[Edge], cfg: GlobalConfig) -> tuple[np.ndarray, np.ndarray]:
    """Returns (T (n,2) centre translations, R (n,2) [theta, log-scale])."""
    RS = np.zeros((n, 2))
    if cfg.solve_rotation:
        rot = [e for e in edges if e.rot_sigma > 0 and e.active]
        if rot:
            I = np.array([e.i for e in rot])
            J = np.array([e.j for e in rot])
            # theta in units of its sigma-ish scale; log-scale rescaled so one
            # IRLS handles both (0.003 rad ~ 0.003 log-scale)
            D = np.array([[e.theta, e.logs] for e in rot])
            W0 = np.array([1.0 / e.rot_sigma ** 2 for e in rot])
            robust = np.array([e.measured for e in rot])
            RS, _, _, _ = _irls(n, I, J, D, W0, robust, c=0.004, iters=6)
    for e in edges:
        th, ls = RS[e.i]
        cs, sn = math.cos(th), math.sin(th)
        sc = math.exp(ls)
        e.d = sc * np.array([cs * e.m[0] - sn * e.m[1], sn * e.m[0] + cs * e.m[1]])
    act = [e for e in edges if e.active]
    I = np.array([e.i for e in act])
    J = np.array([e.j for e in act])
    D = np.array([e.d for e in act])
    W0 = np.array([1.0 / e.sigma ** 2 for e in act])
    robust = np.array([e.measured for e in act])
    T, res, active, W = _irls(n, I, J, D, W0, robust, cfg.robust_px,
                              cfg.irls_iters, cfg.outlier_px)
    for e, r, a, w in zip(act, res, active, W):
        e.residual, e.weight = float(r), float(w)
        if not a:
            e.active = False
    return T, RS


def _overlap_frac(dv: np.ndarray, w: int, h: int) -> float:
    return max(0.0, w - abs(dv[0])) * max(0.0, h - abs(dv[1])) / float(w * h)


# ──────────────────────────────────────────────────────────────────────────────
#  The stitcher
# ──────────────────────────────────────────────────────────────────────────────

class GlobalStitcher:
    def __init__(self, cfg: GlobalConfig | None = None, log=print):
        self.cfg = cfg or GlobalConfig()
        self.log = log
        self.frames: list[ScanFrame] = []
        self.shape: tuple[int, int] | None = None
        self.lens: LensModel | None = None
        self.edges: list[Edge] = []
        self.T: np.ndarray | None = None
        self.RS: np.ndarray | None = None
        self.motion: dict[str, tuple[np.ndarray, float, int]] = {}
        self.timings: dict[str, float] = {}
        self.unreadable: list[str] = []
        self.placed_by_prior: list[int] = []
        self.stats: dict = {}

    # ── loading ─────────────────────────────────────────────────────────────
    def _read(self, f: ScanFrame) -> np.ndarray | None:
        img = cv2.imread(str(f.path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        if self.shape is not None and img.shape[:2] != self.shape:
            img = cv2.resize(img, (self.shape[1], self.shape[0]),
                             interpolation=cv2.INTER_AREA)
        return img

    def _threads(self) -> int:
        return self.cfg.threads or max(1, min(8, (os.cpu_count() or 2) - 1))

    def load(self, folder: Path) -> None:
        t0 = time.perf_counter()
        frames = list_scan(folder)
        if not frames:
            raise FileNotFoundError(f"no images in {folder}")
        first = None
        for f in frames:
            first = cv2.imread(str(f.path), cv2.IMREAD_GRAYSCALE)
            if first is not None:
                break
        if first is None:
            raise RuntimeError("no readable frame in the scan")
        self.shape = first.shape[:2]
        step = max(1, len(frames) // max(1, self.cfg.lens_frames))
        self.lens = estimate_lens(
            cv2.imread(str(f.path), cv2.IMREAD_GRAYSCALE) for f in frames[::step])
        self.log(f"[GLOBAL] {len(frames)} frames, lens circle r="
                 f"{self.lens.radius:.0f}px, vignetting gain up to "
                 f"x{float(self.lens.flat.max()):.2f}")

        self.reg = _Registry(self.cfg, self.lens)
        good: list[ScanFrame] = []

        def prep(f: ScanFrame):
            img = cv2.imread(str(f.path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return f, False
            if img.shape[:2] != self.shape:
                img = cv2.resize(img, (self.shape[1], self.shape[0]),
                                 interpolation=cv2.INTER_AREA)
            self.reg.prepare(f.tag, self.lens.apply(img))
            return f, True

        with ThreadPoolExecutor(self._threads()) as ex:
            for f, ok in ex.map(prep, frames):
                if ok:
                    good.append(f)
                else:
                    self.unreadable.append(f.path.name)
        for k, f in enumerate(good):
            f.index = k
        self.frames = good
        self.timings["load+features"] = time.perf_counter() - t0

    # ── registration ────────────────────────────────────────────────────────
    def _pair(self, i: int, j: int, prior, kind: str) -> Edge | None:
        fi, fj = self.frames[i], self.frames[j]
        res = self.reg.register(fi.tag, fj.tag, prior, self.shape)
        if res is None:
            return None
        m, theta, logs, r = res
        if (r.ncc < self.cfg.min_ncc or r.ncc_hp < self.cfg.min_ncc_hp
                or r.overlap < self.cfg.min_overlap):
            return None
        root = math.sqrt(max(r.inliers, 1))
        sigma = 1.5 + 6.0 * (1.0 - r.ncc_hp) + 30.0 / root
        rot_sigma = 0.0006 + 0.02 / root          # rad (~0.03 deg + ...)
        return Edge(i, j, np.asarray(m, float), sigma, kind, theta, logs,
                    rot_sigma, r.ncc_hp, r.inliers, r.overlap)

    def _coarse_pair(self, i: int, j: int, prior, radius: float) -> Edge | None:
        res = self.reg.coarse(self.frames[i].tag, self.frames[j].tag, prior, radius)
        if res is None:
            return None
        m, ncc, ov = res
        if ncc < self.cfg.coarse_min_ncc or ov < max(0.15, self.cfg.min_overlap):
            return None
        return Edge(i, j, np.asarray(m, float), 4.0 + 10.0 * (1.0 - ncc), "coarse",
                    ncc=ncc, overlap=ov)

    def _run(self, fn, jobs) -> list:
        with ThreadPoolExecutor(self._threads()) as ex:
            return list(ex.map(lambda a: fn(*a), jobs))

    def _learn_motion(self, seq: dict[int, Edge | None]) -> None:
        """Per move type: robust mean displacement and spread."""
        by: dict[str, list[np.ndarray]] = {}
        for j, e in seq.items():
            if e is not None and e.kind != "coarse" and e.ncc >= 0.35:
                by.setdefault(self.frames[j].move, []).append(e.m)
        self.motion = {}
        for mv, ds in by.items():
            a = np.array(ds)
            med = np.median(a, axis=0)
            spread = float(np.median(np.linalg.norm(a - med, axis=1))) * 1.4826
            self.motion[mv] = (med, max(self.cfg.prior_sigma_floor, 1.5 * spread),
                               len(ds))

    def _prior(self, j: int) -> tuple[np.ndarray | None, float]:
        mv = self.frames[j].move
        if mv in self.motion:
            med, sig, _ = self.motion[mv]
            return med, sig
        return None, self.cfg.unknown_move_sigma

    def _solve(self) -> None:
        self.T, self.RS = solve_graph(len(self.frames), self.edges, self.cfg)
        self._components()

    def register_all(self) -> None:
        t0 = time.perf_counter()
        n = len(self.frames)
        cfg = self.cfg
        h, w = self.shape

        # Pass 1: consecutive neighbours, unbounded (learn the motion model).
        jobs = [(j - 1, j, None, "seq") for j in range(1, n)]
        seq: dict[int, Edge | None] = {
            j: e for (_, j, _, _), e in zip(jobs, self._run(self._pair, jobs))}
        self._learn_motion(seq)

        # Pass 2: anything that failed or disagrees with its move model gets
        # a bounded retry around the learned motion.
        retry = []
        for j in range(1, n):
            pd, sig = self._prior(j)
            if pd is None:
                continue
            e = seq.get(j)
            if e is None or np.linalg.norm(e.m - pd) > 3.0 * sig + 40.0:
                retry.append((j - 1, j, pd, "seq"))
        for (i, j, pd, _), e in zip(retry, self._run(self._pair, retry)):
            old = seq.get(j)
            if e is not None and (old is None or e.ncc >= old.ncc - 0.01):
                seq[j] = e
        self._learn_motion(seq)

        # Pass 3: consecutive pairs ORB still cannot match (soft frames) get
        # the blur-tolerant matcher, seeded by the move model.
        jobs = []
        for j in range(1, n):
            if seq.get(j) is None:
                pd, sig = self._prior(j)
                if pd is not None:
                    jobs.append((j - 1, j, pd, 2.0 * sig + 40.0))
        for (i, j, _, _), e in zip(jobs, self._run(self._coarse_pair, jobs)):
            if e is not None:
                e.kind = "coarse"
                seq[j] = e
        n_coarse_seq = sum(1 for e in seq.values() if e is not None and e.kind == "coarse")

        edges: list[Edge] = [e for e in seq.values() if e is not None]

        # Skip edges i -> i+k with the summed prior.
        jobs = []
        for k in range(2, cfg.skip_edges + 1):
            for j in range(k, n):
                pd = np.zeros(2)
                known = True
                for mm in range(j - k + 1, j + 1):
                    e = seq.get(mm)
                    if e is not None:
                        pd = pd + e.m
                    else:
                        p, _ = self._prior(mm)
                        if p is None:
                            known = False
                            break
                        pd = pd + p
                if known and _overlap_frac(pd, w, h) < cfg.min_overlap:
                    continue
                jobs.append((j - k, j, pd if known else None, "skip"))
        edges += [e for e in self._run(self._pair, jobs) if e is not None]

        # Bridges over autofocus frames.  An AF frame is not where the stage
        # ends up (it shows the tail of the previous move), so the frame after
        # it is linked straight to the last non-AF frame with *its own* move
        # model - image link if possible, motion-model link always.
        last_still = [-1] * n
        cur = -1
        for k in range(n):
            last_still[k] = cur
            if not self.frames[k].move.startswith("AF"):
                cur = k
        bridges = []
        for j in range(1, n):
            i = last_still[j]
            if self.frames[j].move.startswith("AF") or i < 0 or i == j - 1:
                continue
            bridges.append((i, j))
        jobs = [(i, j, self._prior(j)[0], "bridge") for i, j in bridges]
        got = self._run(self._pair, jobs)
        retry = [(i, j, pd, 2.0 * self._prior(j)[1] + 40.0)
                 for (i, j, pd, _), e in zip(jobs, got) if e is None and pd is not None]
        got = [e for e in got if e is not None]
        for e in self._run(self._coarse_pair, retry):
            if e is not None:
                got.append(e)
        edges += got

        # Motion-model priors: every frame is linked to its predecessor, and
        # every frame after an AF frame also to the last still frame.
        for j in range(1, n):
            pd, sig = self._prior(j)
            af = self.frames[j].move.startswith("AF")
            edges.append(Edge(j - 1, j, np.zeros(2) if pd is None else pd,
                              sig, "prior", rot_sigma=0.02 if af else 0.008))
        for i, j in bridges:
            pd, sig = self._prior(j)
            if pd is not None:
                edges.append(Edge(i, j, pd, sig, "prior", rot_sigma=0.008))
        self.edges = edges
        self._solve()

        # Loop closures: non-consecutive frames predicted to overlap.  Two
        # rounds - the second uses the poses the first round corrected.
        have = {(min(e.i, e.j), max(e.i, e.j)) for e in self.edges if e.measured}
        tried = 0
        for _round in range(2):
            jobs = []
            for j in range(n):
                cands = []
                for i in range(n):
                    if abs(i - j) <= 1 or (min(i, j), max(i, j)) in have:
                        continue
                    ov = _overlap_frac(self.T[j] - self.T[i], w, h)
                    if ov >= cfg.cross_min_overlap:
                        cands.append((ov, i))
                cands.sort(reverse=True)
                for ov, i in cands[:cfg.cross_pairs]:
                    key = (min(i, j), max(i, j))
                    if key in have:
                        continue
                    have.add(key)
                    jobs.append((i, j, self._pred_m(i, j), "cross"))
            tried += len(jobs)
            added = 0
            for e in self._run(self._pair, jobs):
                if e is None:
                    continue
                pred = self._pred_m(e.i, e.j)
                near = np.linalg.norm(e.m - pred) <= 120.0 + 0.1 * np.linalg.norm(pred)
                strong = e.inliers >= 150 and e.ncc >= 0.5
                if near or strong:       # strong links may correct the graph
                    self.edges.append(e)
                    added += 1
            self._solve()
            if not added:
                break
        self.stats["cross_tried"] = tried

        # Frames still not tied to the main group by an image link: try the
        # coarse matcher against every placed frame they should overlap.
        n_rescued = 0
        for _round in range(2):
            if not self.placed_by_prior:
                break
            main = set(range(n)) - set(self.placed_by_prior)
            jobs = []
            for j in self.placed_by_prior:
                cands = []
                for i in main:
                    ov = _overlap_frac(self.T[j] - self.T[i], w, h)
                    if ov >= 0.25:
                        cands.append((ov, i))
                cands.sort(reverse=True)
                for ov, i in cands[:6]:
                    jobs.append((i, j, self._pred_m(i, j), 160.0))
            got = [e for e in self._run(self._coarse_pair, jobs) if e is not None]
            if not got:
                break
            self.edges += got
            n_rescued += len({e.j for e in got})
            self._solve()

        self.timings["registration"] = time.perf_counter() - t0
        kinds: dict[str, int] = {}
        for e in self.edges:
            if e.measured:
                kinds[e.kind] = kinds.get(e.kind, 0) + 1
        self.stats["edges"] = kinds
        self.stats["coarse_seq"] = n_coarse_seq
        self.stats["outliers"] = sum(1 for e in self.edges
                                     if not e.active and e.measured)

    def _pred_m(self, i: int, j: int) -> np.ndarray:
        """Current estimate of edge (i -> j) in frame i's pixel axes."""
        d = self.T[j] - self.T[i]
        if self.RS is None:
            return d
        th, ls = self.RS[i]
        cs, sn = math.cos(-th), math.sin(-th)
        return math.exp(-ls) * np.array([cs * d[0] - sn * d[1], sn * d[0] + cs * d[1]])

    def _components(self) -> None:
        """Frames linked to the main group only through motion priors."""
        n = len(self.frames)
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for e in self.edges:
            if e.measured and e.active:
                parent[find(e.i)] = find(e.j)
        roots: dict[int, list[int]] = {}
        for k in range(n):
            roots.setdefault(find(k), []).append(k)
        main = set(max(roots.values(), key=len))
        self.placed_by_prior = [k for k in range(n) if k not in main]
        self.stats["groups"] = len(roots)

    # ── compositing ─────────────────────────────────────────────────────────
    def _frame_maps(self, img: np.ndarray, ls: int):
        """Low-res colour, sharpness and luma maps for seam selection."""
        h, w = img.shape[:2]
        half = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        sig = self.cfg.sharp_sigma / 2.0
        e = cv2.GaussianBlur(gx * gx + gy * gy, (0, 0), sig)
        m = cv2.GaussianBlur(g, (0, 0), sig)
        S = e / (m * m + 20.0 ** 2)
        lw, lh = math.ceil(w / ls), math.ceil(h / ls)
        S = cv2.resize(S, (lw, lh), interpolation=cv2.INTER_AREA)
        luma = cv2.resize(m, (lw, lh), interpolation=cv2.INTER_AREA)
        small = cv2.resize(img, (lw, lh), interpolation=cv2.INTER_AREA)
        return small, S, luma

    def _affines(self, s: float):
        """Per-frame 2x3 (scaled frame px -> canvas px) and canvas size."""
        H0, W0 = self.shape
        c = np.array([W0 / 2.0, H0 / 2.0])
        corners = np.array([[0, 0], [W0, 0], [W0, H0], [0, H0]], float)
        L = []
        allc = []
        for k in range(len(self.frames)):
            th, lsc = self.RS[k] if self.RS is not None else (0.0, 0.0)
            sc = math.exp(lsc)
            R = sc * np.array([[math.cos(th), -math.sin(th)],
                               [math.sin(th), math.cos(th)]])
            t = c + self.T[k] - R @ c          # full-res frame -> mosaic
            L.append((R, t))
            allc.append(corners @ R.T + t)
        allc = np.vstack(allc)
        x0, y0 = allc.min(axis=0)
        x1, y1 = allc.max(axis=0)
        area = (x1 - x0) * (y1 - y0)
        if area * s * s > self.cfg.max_canvas_px:
            s = math.sqrt(self.cfg.max_canvas_px / area)
            self.log(f"[GLOBAL] canvas {x1 - x0:.0f}x{y1 - y0:.0f}px exceeds the "
                     f"limit - rendering at scale {s:.2f}")
        M = []
        for R, t in L:
            # scaled frame p' = s p ;  canvas = s (R p + t - origin)
            M.append(np.hstack([R, (s * (t - np.array([x0, y0])))[:, None]]))
        Wc = int(math.ceil((x1 - x0) * s)) + 2
        Hc = int(math.ceil((y1 - y0) * s)) + 2
        return M, Wc, Hc, s, (x0, y0)

    @staticmethod
    def _roi(M: np.ndarray, w: int, h: int, Wc: int, Hc: int, pad: int = 1):
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
        pts = corners @ M[:, :2].T + M[:, 2]
        rx0 = max(0, int(math.floor(pts[:, 0].min())) - pad)
        ry0 = max(0, int(math.floor(pts[:, 1].min())) - pad)
        rx1 = min(Wc, int(math.ceil(pts[:, 0].max())) + pad)
        ry1 = min(Hc, int(math.ceil(pts[:, 1].max())) + pad)
        Mr = M.copy()
        Mr[0, 2] -= rx0
        Mr[1, 2] -= ry0
        return rx0, ry0, rx1, ry1, Mr

    def compose(self, out_prefix: str) -> dict:
        cfg = self.cfg
        t0 = time.perf_counter()
        n = len(self.frames)
        H0, W0 = self.shape
        M, Wc, Hc, s, origin = self._affines(cfg.output_scale)
        Wf, Hf = int(round(W0 * s)), int(round(H0 * s))
        lens = self.lens.resized(Wf, Hf)
        ls = cfg.label_scale
        lw, lh = math.ceil(Wf / ls), math.ceil(Hf / ls)
        Lw, Lh = math.ceil(Wc / ls) + 1, math.ceil(Hc / ls) + 1

        valid_lr = cv2.resize(lens.valid, (lw, lh), interpolation=cv2.INTER_AREA)
        yy, xx = np.mgrid[0:lh, 0:lw].astype(np.float32)
        central = ((1.0 - np.abs((xx + 0.5) / lw * 2 - 1))
                   * (1.0 - np.abs((yy + 0.5) / lh * 2 - 1))).astype(np.float32)
        unlinked = set(self.placed_by_prior)

        def lr_affine(k):
            A = M[k].copy()
            A[:, 2] /= ls
            return A          # LR frame px -> LR canvas px (linear part unchanged)

        # Pass A: per-frame low-res maps, warped onto the label grid
        def pass_a(k):
            img = self._read(self.frames[k])
            if img is None:
                return k, None
            img = self.lens.apply(img)
            if s != 1.0:
                img = cv2.resize(img, (Wf, Hf), interpolation=cv2.INTER_AREA)
            small, S, luma = self._frame_maps(img, ls)
            dark = np.clip((luma - 8.0) / 25.0, 0.05, 1.0)
            base = valid_lr * (0.05 + central) * dark
            if k in unlinked:
                base = base * cfg.unlinked_penalty
            rx0, ry0, rx1, ry1, Mr = self._roi(lr_affine(k), lw, lh, Lw, Lh)
            size = (rx1 - rx0, ry1 - ry0)
            if size[0] <= 0 or size[1] <= 0:
                return k, None
            wp = lambda a, interp=cv2.INTER_LINEAR: cv2.warpAffine(
                a, Mr, size, flags=interp, borderMode=cv2.BORDER_CONSTANT)
            v = wp(valid_lr)
            return k, ((rx0, ry0, rx1, ry1), wp(S), wp(base), wp(small), v)

        maps: list = [None] * n
        with ThreadPoolExecutor(self._threads()) as ex:
            for k, m in ex.map(pass_a, range(n)):
                maps[k] = m

        Smax = np.zeros((Lh, Lw), np.float32)
        for m in maps:
            if m is None:
                continue
            (x0, y0, x1, y1), S, _, _, v = m
            sl = (slice(y0, y1), slice(x0, x1))
            np.maximum(Smax[sl], S * (v > 0.5), out=Smax[sl])
        allS = np.concatenate([m[1][m[4] > 0.5].ravel() for m in maps if m is not None])
        s_ref = float(np.percentile(allS, 90)) if allS.size else 1.0
        eps = 0.02 * float(np.median(allS)) + 1e-6

        gains = self._gains(maps)

        # Multi-view consensus.  Where several registered frames see the same
        # spot, a frame that shows something different there is either
        # misregistered or - the common case on this rig - geometrically
        # stretched: the blade curves away on the right, so a frame taken at
        # another Z (autofocus) or from another strip maps that bland, curved
        # part differently than any rigid model predicts.  Measured: 86 px of
        # disagreement at the right edge of an AF frame whose textured left
        # half registers to 1.6 px.  Without this term each such frame paints
        # its own copy of a cooling hole next to the true one ("doubled holes").
        lum: list = [None] * n
        for k, m in enumerate(maps):
            if m is None:
                continue
            y = (m[3].astype(np.float32) * gains[k][None, None, :]).mean(axis=2)
            lum[k] = cv2.GaussianBlur(y, (0, 0), 1.0)
        mu = None
        for it in range(3):
            num = np.zeros((Lh, Lw), np.float32)
            den = np.zeros((Lh, Lw), np.float32)
            for k, m in enumerate(maps):
                if m is None:
                    continue
                x0, y0, x1, y1 = m[0]
                w = (m[4] > 0.5).astype(np.float32)
                if k in unlinked:
                    w *= 0.2
                if mu is not None:
                    r = (lum[k] - mu[y0:y1, x0:x1]) / cfg.consensus_sigma
                    w = w / (1.0 + r * r)
                num[y0:y1, x0:x1] += w * lum[k]
                den[y0:y1, x0:x1] += w
            mu = num / np.maximum(den, 1e-6)
        support = den

        # log-scores per frame (in its ROI)
        logsc: list = [None] * n
        for k, m in enumerate(maps):
            if m is None:
                continue
            (x0, y0, x1, y1), S, base, _, v = m
            rel = (S + eps) / (Smax[y0:y1, x0:x1] + eps)
            sc = base * rel ** cfg.focus_power
            lg = np.log(np.maximum(sc, 1e-12)).astype(np.float32)
            dev = np.abs(lum[k] - mu[y0:y1, x0:x1]) / cfg.consensus_sigma
            # only meaningful where at least ~2 frames vote
            lg -= cfg.consensus_weight * dev * np.clip(support[y0:y1, x0:x1] - 1.0, 0, 1)
            lg[(v <= 0.5) | (sc <= 0)] = -np.inf
            logsc[k] = lg

        def assign(label_prev=None):
            best = np.full((Lh, Lw), -np.inf, np.float32)
            label = np.full((Lh, Lw), -1, np.int32)
            for k, m in enumerate(maps):
                if m is None:
                    continue
                x0, y0, x1, y1 = m[0]
                e = logsc[k]
                if label_prev is not None and cfg.seam_smoothness > 0:
                    own = (label_prev[y0:y1, x0:x1] == k).astype(np.float32)
                    frac = cv2.boxFilter(own, -1, (5, 5), normalize=True,
                                         borderType=cv2.BORDER_REPLICATE)
                    e = e + cfg.seam_smoothness * frac
                sl = (slice(y0, y1), slice(x0, x1))
                win = e > best[sl]
                best[sl][win] = e[win]
                label[sl][win] = k
            return label, best

        label, best = assign()
        for _ in range(5):                       # Potts MRF by ICM
            label, best = assign(label)
        won = np.bincount(label[label >= 0].ravel(), minlength=n)
        self.timings["seams+gains"] = time.perf_counter() - t0

        # Pass B: multi-band blend at output resolution
        t1 = time.perf_counter()
        blender = cv2.detail_MultiBandBlender(0, cfg.num_bands)
        blender.prepare((0, 0, Wc, Hc))
        k5 = np.ones((5, 5), np.uint8)

        def pass_b(k):
            img = self._read(self.frames[k])
            if img is None:
                return None
            img = self.lens.apply(img)
            if s != 1.0:
                img = cv2.resize(img, (Wf, Hf), interpolation=cv2.INTER_AREA)
            img = np.clip(img.astype(np.float32) * gains[k][None, None, :], 0, 255)
            rx0, ry0, rx1, ry1, Mr = self._roi(M[k], Wf, Hf, Wc, Hc, pad=2)
            size = (rx1 - rx0, ry1 - ry0)
            warped = cv2.warpAffine(img, Mr, size, flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT)
            vw = cv2.warpAffine(lens.valid, Mr, size, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
            # label indicator at full res: bilinear upsample, threshold, then a
            # small dilation so neighbouring masks overlap instead of leaving
            # pin-holes where three labels meet.
            lx0, ly0 = rx0 // ls, ry0 // ls
            lx1 = min(Lw, -(-rx1 // ls) + 1)
            ly1 = min(Lh, -(-ry1 // ls) + 1)
            ind = (label[ly0:ly1, lx0:lx1] == k).astype(np.float32)
            up = cv2.resize(ind, ((lx1 - lx0) * ls, (ly1 - ly0) * ls),
                            interpolation=cv2.INTER_LINEAR)
            ox, oy = rx0 - lx0 * ls, ry0 - ly0 * ls
            up = up[oy:oy + size[1], ox:ox + size[0]]
            if up.shape != (size[1], size[0]):
                pad = np.zeros((size[1], size[0]), np.float32)
                pad[:up.shape[0], :up.shape[1]] = up
                up = pad
            mask = cv2.dilate((up > 0.5).astype(np.uint8), k5)
            mask &= (vw > 0.15).astype(np.uint8)
            if not mask.any():
                return None
            # Fill the optically clipped corners with nearby content so the
            # pyramid's low bands do not drag black into the seam.
            if (vw < 0.95).any():
                q = 8
                sm = cv2.resize(warped * vw[:, :, None], None, fx=1 / q, fy=1 / q,
                                interpolation=cv2.INTER_AREA)
                sw = cv2.resize(vw, None, fx=1 / q, fy=1 / q,
                                interpolation=cv2.INTER_AREA)
                sm = cv2.GaussianBlur(sm, (0, 0), 6)
                sw = cv2.GaussianBlur(sw, (0, 0), 6)
                fill = sm / np.maximum(sw, 1e-3)[:, :, None]
                fill = cv2.resize(fill, size, interpolation=cv2.INTER_LINEAR)
                a = vw[:, :, None]
                warped = warped * a + fill * (1 - a)
            return warped.astype(np.int16), mask * 255, (rx0, ry0)

        fed = 0
        order = [k for k in range(n) if won[k] > 0]
        with ThreadPoolExecutor(min(4, self._threads())) as ex:
            for item in ex.map(pass_b, order):
                if item is None:
                    continue
                img16, mask, tl = item
                blender.feed(img16, mask, tl)
                fed += 1
        result, rmask = blender.blend(None, None)
        mosaic = np.clip(result, 0, 255).astype(np.uint8)
        mosaic[rmask == 0] = 0
        del result
        self.timings["blend"] = time.perf_counter() - t1

        # QC maps on the label grid
        S_win = np.zeros((Lh, Lw), np.float32)
        for k, m in enumerate(maps):
            if m is None or won[k] == 0:
                continue
            x0, y0, x1, y1 = m[0]
            sel = label[y0:y1, x0:x1] == k
            S_win[y0:y1, x0:x1][sel] = m[1][sel]
        covered = label >= 0
        lowd = covered & (S_win < cfg.low_detail_ratio * s_ref)
        low_pct = 100.0 * lowd.sum() / max(1, covered.sum())
        rng = np.random.default_rng(7)
        colours = rng.integers(40, 255, size=(n + 1, 3), dtype=np.uint8)
        colours[0] = 0
        src = colours[label + 1]
        for k in unlinked:
            src[label == k] = (0, 0, 255)
        seams = cv2.morphologyEx((label + 1).astype(np.float32), cv2.MORPH_GRADIENT,
                                 np.ones((3, 3), np.uint8)) > 0
        src[seams] = 255

        def up_full(a):
            a = cv2.resize(a, (Lw * ls, Lh * ls), interpolation=cv2.INTER_NEAREST)
            return a[:Hc, :Wc]

        out = {
            "jpg": f"{out_prefix}_mosaic.jpg",
            "png": f"{out_prefix}_mosaic.png",
            "coverage": f"{out_prefix}_coverage.png",
            "sources": f"{out_prefix}_sources.png",
            "low_detail": f"{out_prefix}_low_detail.png",
        }
        Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
        ys, xs = np.where(rmask > 0)
        if ys.size:
            cy0, cy1, cx0, cx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        else:
            cy0, cy1, cx0, cx1 = 0, Hc, 0, Wc
        crop = (slice(cy0, cy1), slice(cx0, cx1))
        cv2.imwrite(out["jpg"], mosaic[crop], [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
        cv2.imwrite(out["png"], mosaic[crop], [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(out["coverage"], rmask[crop])
        src_full = up_full(src)[crop]
        k_src = 1600.0 / max(1, src_full.shape[0])
        if k_src < 1:
            src_full = cv2.resize(src_full, None, fx=k_src, fy=k_src,
                                  interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(out["sources"], src_full)
        cv2.imwrite(out["low_detail"], (up_full(lowd.astype(np.uint8)) * 255)[crop])
        out["shape"] = mosaic[crop].shape
        out["scale"] = s
        out["low_detail_pct"] = round(float(low_pct), 2)
        out["frames_used"] = int(fed)
        self.won = won
        self.label, self.label_scale_px, self.canvas_crop = label, ls, (cx0, cy0)
        self.label_cells = lw * lh
        self.gains = gains
        self.timings["compose"] = time.perf_counter() - t0
        return out

    def _gains(self, maps) -> np.ndarray:
        """Per-frame, per-channel exposure gains (OpenCV GainCompensator form)."""
        n = len(self.frames)
        alpha, beta = 1.0 / 10.0 ** 2, 1.0 / 0.1 ** 2
        g_out = np.ones((n, 3), np.float32)
        Nn = np.zeros((n, n))
        Iij = np.zeros((n, n, 3))
        for i in range(n):
            if maps[i] is None:
                continue
            ax0, ay0, ax1, ay1 = maps[i][0]
            for j in range(i + 1, n):
                if maps[j] is None:
                    continue
                bx0, by0, bx1, by1 = maps[j][0]
                ix0, iy0 = max(ax0, bx0), max(ay0, by0)
                ix1, iy1 = min(ax1, bx1), min(ay1, by1)
                if ix1 - ix0 < 4 or iy1 - iy0 < 4:
                    continue
                si = (slice(iy0 - ay0, iy1 - ay0), slice(ix0 - ax0, ix1 - ax0))
                sj = (slice(iy0 - by0, iy1 - by0), slice(ix0 - bx0, ix1 - bx0))
                a = maps[i][3][si].astype(np.float32)
                b = maps[j][3][sj].astype(np.float32)
                la, lb = a.mean(axis=2), b.mean(axis=2)
                m = ((maps[i][4][si] > 0.9) & (maps[j][4][sj] > 0.9)
                     & (la > 15) & (lb > 15) & (la < 245) & (lb < 245))
                cnt = int(m.sum())
                if cnt < 30:
                    continue
                Nn[i, j] = Nn[j, i] = cnt
                Iij[i, j] = a[m].mean(axis=0)
                Iij[j, i] = b[m].mean(axis=0)
        for c in range(3):
            A = np.zeros((n, n))
            bvec = np.zeros(n)
            for i in range(n):
                row = Nn[i]
                if row.sum() == 0:
                    A[i, i] = 1.0
                    bvec[i] = 1.0
                    continue
                for j in np.nonzero(row)[0]:
                    A[i, i] += row[j] * (Iij[i, j, c] ** 2 * alpha + beta)
                    A[i, j] -= row[j] * alpha * Iij[i, j, c] * Iij[j, i, c]
                    bvec[i] += beta * row[j]
            try:
                g_out[:, c] = np.linalg.solve(A, bvec)
            except np.linalg.LinAlgError:
                pass
        return np.clip(g_out, 0.5, 2.0)

    # ── reporting ───────────────────────────────────────────────────────────
    def write_reports(self, out_prefix: str, paths: dict) -> None:
        n = len(self.frames)
        deg = np.zeros(n, int)
        res: list[list[float]] = [[] for _ in range(n)]
        for e in self.edges:
            if e.measured and e.active:
                deg[e.i] += 1
                deg[e.j] += 1
                res[e.i].append(e.residual)
                res[e.j].append(e.residual)
        unlinked = set(self.placed_by_prior)
        with open(f"{out_prefix}_poses.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["index", "file", "column", "phase", "move", "x_px", "y_px",
                         "rot_deg", "scale", "placement", "links",
                         "mean_residual_px", "area_won_pct",
                         "gain_b", "gain_g", "gain_r"])
            for k, fr in enumerate(self.frames):
                th, lsc = self.RS[k]
                wr.writerow([k, fr.path.name, fr.column, fr.phase, fr.move,
                             round(float(self.T[k, 0]), 1), round(float(self.T[k, 1]), 1),
                             round(math.degrees(th), 3), round(math.exp(lsc), 4),
                             "motion_model" if k in unlinked else "registered",
                             int(deg[k]),
                             round(float(np.mean(res[k])), 2) if res[k] else "",
                             round(100.0 * self.won[k] / self.label_cells, 1),
                             *[round(float(g), 3) for g in self.gains[k]]])
        lines = ["Global stitch report", "=" * 62,
                 f"frames in scan           : {n + len(self.unreadable)}",
                 f"frames used (none dropped): {n}"
                 + (f"  (+{len(self.unreadable)} unreadable files)" if self.unreadable else ""),
                 f"frames visible in mosaic : {paths.get('frames_used')}"
                 "  (others are fully covered by sharper / more central views)",
                 f"image links              : {self.stats.get('edges')}",
                 f"  of which blur-tolerant : {self.stats.get('edges', {}).get('coarse', 0)}",
                 f"links rejected as inconsistent: {self.stats.get('outliers')}",
                 f"placed by motion model only   : {len(self.placed_by_prior)}",
                 f"mosaic                   : {paths.get('shape')} at scale "
                 f"{paths.get('scale'):.2f}",
                 f"low-detail area          : {paths.get('low_detail_pct')} % "
                 "(no frame resolved fine detail there: blurred or featureless)",
                 "", "Learned motion model (frame-centre displacement, px):"]
        for mv, (med, sig, cnt) in sorted(self.motion.items()):
            lines.append(f"  {mv:10s} dx={med[0]:8.1f}  dy={med[1]:8.1f}  "
                         f"spread={sig:6.1f}  (n={cnt})")
        act = [e.residual for e in self.edges if e.measured and e.active]
        if act:
            lines += ["", f"link residuals after solve: median {np.median(act):.2f} px, "
                          f"p95 {np.percentile(act, 95):.2f} px, max {np.max(act):.2f} px"]
        if self.RS is not None:
            rot = np.degrees(self.RS[:, 0])
            sc = np.exp(self.RS[:, 1])
            lines.append(f"frame rotation range: {rot.min():.2f} .. {rot.max():.2f} deg,"
                         f" scale range {sc.min():.4f} .. {sc.max():.4f}")
        if self.placed_by_prior:
            lines += ["", "Frames placed by the motion model (no image link) - "
                          "shown only where nothing else covers the spot:"]
            for k in self.placed_by_prior:
                lines.append(f"  {self.frames[k].path.name}")
        if self.unreadable:
            lines += ["", "Unreadable files:"] + [f"  {u}" for u in self.unreadable]
        lines += ["", "Timings (s): " + ", ".join(f"{k} {v:.1f}" for k, v in self.timings.items()),
                  "", f"mosaic: {paths.get('png')}"]
        Path(f"{out_prefix}_report.txt").write_text("\n".join(lines) + "\n")
        self.log("[GLOBAL] " + "\n[GLOBAL] ".join(lines[:13]))


def stitch_folder(folder, out_prefix, cfg: GlobalConfig | None = None,
                  log=print) -> dict:
    """Stitch every frame of a scan folder.  Returns the output paths."""
    st = GlobalStitcher(cfg, log=log)
    st.load(Path(folder))
    st.register_all()
    paths = st.compose(str(out_prefix))
    st.write_reports(str(out_prefix), paths)
    paths["report"] = f"{out_prefix}_report.txt"
    paths["poses"] = f"{out_prefix}_poses.csv"
    return paths
