"""Tests for the live stitcher.

Run with ``python -m pytest image_stitching/live/tests -q`` or directly
``python image_stitching/live/tests/test_live.py``.

The synthetic tests pin down the behaviours that the recorded scans exposed;

* translation recovery from a known shift,
* rejection of a black / flat / defocused frame,
* rejection of a duplicate (stalled motor),
* holes being filled by soft frames without overwriting good pixels,
* the session surviving a frame the pipeline cannot process.

The optional real-data tests use ``from_scratch/test_images`` when present and
are skipped otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]      # image_stitching/
sys.path.insert(0, str(ROOT))

from live.config import LiveConfig                                      # noqa: E402
from live.mosaic import IncrementalMosaic                               # noqa: E402
from live.pipeline import LiveStitcher                                  # noqa: E402
from live.quality import assess_intrinsic, sharpness                    # noqa: E402
from live.registration import Registrar                                 # noqa: E402

SCAN_DIR = ROOT / "from_scratch" / "test_images"


# ── helpers ──────────────────────────────────────────────────────────────────
def texture(h=600, w=800, seed=7) -> np.ndarray:
    """A deterministic, richly textured fake blade surface."""
    rng = np.random.default_rng(seed)
    base = rng.normal(128, 22, (h, w)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 1.2)
    for _ in range(40):                                  # speckle + pits
        x, y = rng.integers(10, w - 10), rng.integers(10, h - 10)
        cv2.circle(base, (int(x), int(y)), int(rng.integers(2, 9)), 40, -1)
    for _ in range(6):                                   # long scratches
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(base, p1, p2, 200, int(rng.integers(1, 3)))
    base = np.clip(base, 0, 255).astype(np.uint8)
    return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)


def crop(img, dy, dx=0, h=400, w=600):
    return img[dy:dy + h, dx:dx + w].copy()


def no_log(*_a):
    pass


# ── registration ─────────────────────────────────────────────────────────────
def test_recovers_known_shift():
    img = texture()
    a, b = crop(img, 0), crop(img, 130, 20)
    cfg = LiveConfig()
    reg = Registrar(cfg)
    reg.prepare("a", a)
    reg.prepare("b", b)
    r = reg.register("a", "b")
    assert r is not None, "registration failed on a clean synthetic pair"
    dx, dy = r.shift
    assert abs(dx - 20) < 3 and abs(dy - 130) < 3, f"shift off: {(dx, dy)}"
    assert r.ncc > 0.9


def test_prior_rescues_when_matches_are_contradictory():
    """The prior must not be able to invent a transform that the pixels reject."""
    img = texture()
    cfg = LiveConfig()
    reg = Registrar(cfg)
    reg.prepare("a", crop(img, 0))
    reg.prepare("b", crop(img, 200))
    # a prior pointing the wrong way must NOT be trusted blindly
    r = reg.register("a", "b", np.array([0.0, -260.0], np.float32))
    assert r is None or r.ncc > cfg.min_ncc


# ── quality gate ─────────────────────────────────────────────────────────────
def test_black_and_flat_frames_rejected():
    cfg = LiveConfig()
    assert not assess_intrinsic(np.zeros((400, 600, 3), np.uint8), cfg).ok
    flat = np.full((400, 600, 3), 128, np.uint8)
    assert not assess_intrinsic(flat, cfg).ok


def test_defocus_is_visible_to_the_sharpness_measure():
    img = texture()
    soft = cv2.GaussianBlur(img, (0, 0), 5.0)
    assert sharpness(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) > \
        1.8 * sharpness(cv2.cvtColor(soft, cv2.COLOR_BGR2GRAY))


# ── mosaic ───────────────────────────────────────────────────────────────────
def test_mosaic_places_and_averages():
    img = texture()
    cfg = LiveConfig(flush_every=0)
    mo = IncrementalMosaic(cfg)
    mo.add(img, np.eye(3))
    T = np.array([[1, 0, 0], [0, 1, 300], [0, 0, 1]], float)
    res = mo.add(img, T)
    assert 0.3 < res.new_area < 0.7
    mosaic, coverage, lowq = mo.render()
    assert mosaic.shape[0] == 600 + 300 and mosaic.shape[1] == 800
    assert (coverage > 0).all() and not (lowq > 0).any()
    x0, y0, x1, y1 = mo.bbox()
    assert (x1 - x0, y1 - y0) == (800, 900)


def test_soft_frame_only_fills_holes():
    """A soft fill-in must never overwrite pixels that are already good."""
    img = texture()
    soft = cv2.GaussianBlur(img, (0, 0), 4.0)
    mo = IncrementalMosaic(LiveConfig(flush_every=0))
    mo.add(img, np.eye(3))
    T = np.array([[1, 0, 0], [0, 1, 300], [0, 0, 1]], float)
    res = mo.add(soft, T, empty_only=True, low_quality=True)
    mosaic, _, lowq = mo.render()
    # the soft frame contributed only the band it alone covers
    assert 0.0 < res.new_area <= 0.51
    covered_rows = (lowq > 0).any(axis=1)
    assert covered_rows.sum() > 0 and covered_rows[:600].sum() == 0


# ── pipeline ─────────────────────────────────────────────────────────────────
def _pipeline(tmp_path, **kw):
    return LiveStitcher(LiveConfig(flush_every=0, **kw),
                        out_prefix=str(tmp_path / "run"), log=no_log)


def test_pipeline_keeps_good_frames_and_drops_duplicates(tmp_path):
    img = texture()
    st = _pipeline(tmp_path)
    try:
        d1 = st.process(crop(img, 0), "a")
        d2 = st.process(crop(img, 150), "b")
        d3 = st.process(crop(img, 150), "c")          # stalled motor: duplicate
        assert d1.ok and d2.ok
        assert not d3.ok and "duplicate" in d3.reason
        assert st.mosaic.count == 2
    finally:
        st.stop()


def test_pipeline_drops_defocused_frame(tmp_path):
    img = texture()
    st = _pipeline(tmp_path)
    try:
        st.process(crop(img, 0), "a")
        d = st.process(cv2.GaussianBlur(crop(img, 150), (0, 0), 6.0), "b")
        assert not d.ok
        assert "focus" in d.reason or "soft" in d.reason
    finally:
        st.stop()


def test_pipeline_survives_a_broken_frame(tmp_path):
    img = texture()
    st = _pipeline(tmp_path)
    try:
        st.process(crop(img, 0), "a")
        # an all-black frame exercises the reject path, not an exception
        d = st.process(np.zeros((400, 600, 3), np.uint8), "bad")
        assert not d.ok
        d2 = st.process(crop(img, 150), "b")
        assert d2.ok, "pipeline stopped accepting frames after a bad one"
    finally:
        st.stop()


def test_submit_is_asynchronous_and_finalize_writes_outputs(tmp_path):
    img = texture()
    st = LiveStitcher(LiveConfig(flush_every=0), out_prefix=str(tmp_path / "run"),
                      log=no_log)
    for i, dy in enumerate((0, 150, 300)):
        st.submit(crop(img, dy), f"f{i}")
    assert st.wait_idle(30), "worker did not drain the queue"
    paths = st.finalize()
    assert Path(paths["png"]).exists() and Path(paths["coverage"]).exists()
    assert Path(str(tmp_path / "run_frames.csv")).exists()
    assert Path(str(tmp_path / "run_report.txt")).exists()


# ── frame geometry: clipped borders are not image ────────────────────────────
def test_clipped_border_is_excluded_from_the_valid_mask():
    from live.frames import analyse_frame
    img = texture()
    frame = crop(img, 0).copy()
    frame[:, int(frame.shape[1] * 0.85):] = 0          # dead right strip
    geo = analyse_frame(frame)
    assert geo.dead_frac > 0.05
    assert geo.valid[:, -5:].max() == 0, "dead border marked as image"
    assert geo.valid[:, :100].min() == 255, "live area cut away"
    assert geo.box[2] <= int(frame.shape[1] * 0.87), geo.box


def test_a_mostly_dead_frame_is_recognised_as_a_capture_fault():
    from live.frames import analyse_frame
    frame = crop(texture(), 0).copy()
    frame[:] = 0
    geo = analyse_frame(frame)
    assert geo.dead_frac > 0.9


def test_dark_parts_of_the_surface_are_not_cut_away():
    """Holes and shadow are 20-60 luma: they must stay part of the image."""
    from live.frames import analyse_frame
    frame = crop(texture(), 0).copy()
    geo = analyse_frame(frame)
    assert geo.dead_frac < 0.01, geo.dead_frac
    assert geo.valid.min() == 255


def test_blend_weight_decays_inside_the_valid_area():
    from live.frames import smooth_weight
    mask = np.zeros((120, 120), np.uint8)
    mask[10:-10, 10:-10] = 255
    w = smooth_weight(mask, feather_px=20)
    assert w[0, 0] == 0.0                      # outside: never blended
    assert 0.9 < w[60, 60] <= 1.0              # deep inside: full weight
    # and it is monotone on the way in, with no step at the feather limit
    profile = w[60, 10:40]
    assert np.all(np.diff(profile) >= -1e-6), profile
    assert (np.diff(profile, 2).__abs__() < 0.05).all(), "weight has a kink"


def test_illumination_is_flattened_without_touching_defects():
    """The fix for the rectangles: remove the frame's own light falloff."""
    from live.frames import flatten_illumination
    h, w = 300, 400
    yy, xx = np.mgrid[0:h, 0:w]
    rng = np.random.default_rng(7)
    flat = np.full((h, w), 120.0) + rng.normal(0, 12, (h, w))   # surface + grain
    flat = np.clip(np.dstack([flat] * 3), 20, 235)
    lit = np.clip(flat * (1.0 + 0.28 * (xx / w - 0.5))[:, :, None], 0, 255)
    lit = lit.astype(np.uint8)
    out = flatten_illumination(lit)
    # the trend across the frame is gone ...
    left_before = float(lit[:, :40].mean())
    right_before = float(lit[:, -40:].mean())
    left_after = float(out[:, :40].mean())
    right_after = float(out[:, -40:].mean())
    assert right_before / left_before > 1.2, (left_before, right_before)
    assert abs(right_after / left_after - 1.0) < 0.08, (left_after, right_after)
    # ... the overall level is kept ...
    assert abs(float(out.mean()) - float(lit.mean())) < 12
    # ... and fine detail survives
    assert float(np.std(out.astype(np.float32) - lit.astype(np.float32))) < 20
    detail_before = float(np.abs(np.diff(lit.astype(np.float32), axis=1)).mean())
    detail_after = float(np.abs(np.diff(out.astype(np.float32), axis=1)).mean())
    assert detail_after > 0.7 * detail_before, (detail_before, detail_after)


def test_flattening_ignores_a_dead_border():
    from live.frames import analyse_frame, flatten_illumination
    img = texture()
    frame = crop(img, 0).copy()
    frame[:, int(frame.shape[1] * 0.8):] = 0            # dead right strip
    geo = analyse_frame(frame)
    out = flatten_illumination(frame, geo.valid)
    live = geo.valid > 0
    # the black strip must not drag the profile, so the live area keeps its level
    assert abs(float(out[live].mean()) - float(frame[live].mean())) < 12


def _scan_with_clipped_border(tmp_path, name, clip):
    img = texture()
    st = LiveStitcher(LiveConfig(flush_every=0),
                      out_prefix=str(tmp_path / name), log=no_log)
    try:
        for dy in (0, 150, 300, 450):
            f = crop(img, dy).copy()
            if clip:
                f[:, int(f.shape[1] * 0.8):] = 0        # dead right strip
            st.submit(f, f"f{dy}", axis="Y")
        st.wait_idle(60)
        mosaic, cov, _lq = st.mosaic.render()
        return mosaic, cov
    finally:
        st.stop()


def test_a_clipped_border_does_not_change_the_mosaic(tmp_path):
    """The delivered artefact: clipped pixels lifted by the gain clip.

    The dead strip must be *masked*, not brightened: a scan whose frames lose
    20 % of their width has to stitch to the same picture as a clean scan of the
    same part (and must not stretch the canvas over the dead column).
    """
    clean, cov_clean = _scan_with_clipped_border(tmp_path, "clean", clip=False)
    clipped, cov_clip = _scan_with_clipped_border(tmp_path, "clip", clip=True)
    h = min(clean.shape[0], clipped.shape[0])
    w = min(clean.shape[1], clipped.shape[1])
    diff = np.abs(clean[:h, :w].astype(np.int16)
                  - clipped[:h, :w].astype(np.int16)).mean(axis=2)
    shared = (cov_clean[:h, :w] > 0) & (cov_clip[:h, :w] > 0)
    assert shared.mean() > 0.9, "the two scans do not cover the same part"
    assert diff[shared].mean() < 1.5, f"mean |diff| {diff[shared].mean():.1f}"
    assert diff[shared].max() < 40, f"max |diff| {diff[shared].max()}"
    # and the canvas is not widened by the dead column
    assert clipped.shape[1] <= clean.shape[1] * 0.95, (
        clean.shape, clipped.shape)


# ── stage prior: learn the rig's real motion signature ──────────────────────
def test_prior_learns_each_axis_separately():
    from live.prior import StagePrior
    p = StagePrior(min_samples=3)
    for i in range(6):
        p.observe("Y", (2.0 + i * 0.1, 250.0 + i))
        p.observe("X", (230.0 + i, -4.0))
    assert p.known("Y") and p.known("X")
    assert p.expected("Y")[1] > 200 and abs(p.expected("Y")[0]) < 10
    assert p.expected("X")[0] > 200 and abs(p.expected("X")[1]) < 10


def test_prior_does_not_veto_until_it_has_evidence():
    from live.prior import StagePrior
    p = StagePrior(min_samples=3)
    ok, why = p.check("Y", (500.0, -400.0))
    assert ok and why == ""


def test_prior_vetoes_a_step_that_is_off_its_axis():
    from live.prior import StagePrior
    p = StagePrior(min_samples=3)
    for _ in range(5):
        p.observe("Y", (3.0, 250.0))
    ok, why = p.check("Y", (3.0, 252.0))
    assert ok, why                                  # normal noise passes
    ok, why = p.check("Y", (250.0, 3.0))            # X-shaped step on a Y move
    assert not ok and "off_axis" in why, why


def test_prior_tracks_a_drifting_axis():
    """An open-loop actuator's step varies; the prior must follow, not ossify."""
    from live.prior import StagePrior
    p = StagePrior(min_samples=3, max_samples=10)
    for i in range(10):
        p.observe("Y", (0.0, 200.0 + 12 * i))
    assert p.expected("Y")[1] > 240, p.expected("Y")
    ok, _ = p.check("Y", (0.0, 320.0))
    assert ok


def test_pipeline_uses_the_axis_prior_to_place_a_serpentine_scan(tmp_path):
    """Y frames advance in Y; the X turn is placed as X, not forced into Y."""
    img = texture()
    st = _pipeline(tmp_path, axis_min_samples=2)
    try:
        for i, dy in enumerate((0, 150, 300, 450)):
            st.submit(crop(img, dy), f"y{i}", axis="Y")
        st.wait_idle(60)
        assert st.prior.known("Y"), st.prior.summary()
        step = st.prior.expected("Y")
        assert 100 < step[1] < 200, step
        assert abs(step[0]) < 40, step
    finally:
        st.stop()


# ── live preview (what the GUI polls) ────────────────────────────────────────
def test_preview_empty_returns_none(tmp_path):
    st = _pipeline(tmp_path)
    try:
        rgb, summary = st.preview(120)
        assert rgb is None
        assert summary["accepted"] == 0
    finally:
        st.stop()


def test_preview_returns_scaled_rgb_snapshot(tmp_path):
    img = texture()
    # make the frame warm (BGR: little blue) so a BGR/RGB mix-up is detectable
    warm = img.copy()
    warm[:, :, 0] = warm[:, :, 0] // 4
    st = _pipeline(tmp_path)
    try:
        st.process(crop(warm, 0), "a")
        st.process(crop(warm, 150), "b")
        rgb, summary = st.preview(120)
        assert rgb is not None and rgb.dtype == np.uint8 and rgb.ndim == 3
        assert rgb.shape[1] <= 120, rgb.shape
        assert summary["accepted"] >= 2, summary
        b, g, r = (int(v) for v in crop(warm, 0)[0, 0])
        assert abs(int(rgb[0, 0, 0]) - r) < 12, (rgb[0, 0], (r, g, b))
        assert abs(int(rgb[0, 0, 2]) - b) < 12, (rgb[0, 0], (r, g, b))
    finally:
        st.stop()


def test_preview_is_safe_while_frames_are_being_stitched(tmp_path):
    """The GUI polls on its own thread while the worker blends."""
    import threading
    img = texture()
    st = LiveStitcher(LiveConfig(flush_every=0), out_prefix=str(tmp_path / "run"),
                      log=no_log)
    errors, shapes = [], []

    def poll():
        for _ in range(25):
            try:
                rgb, _ = st.preview(60)
                if rgb is not None:
                    shapes.append(rgb.shape)
            except Exception as exc:            # noqa: BLE001
                errors.append(exc)

    try:
        poller = threading.Thread(target=poll)
        poller.start()
        for i, dy in enumerate((0, 150, 300, 450)):
            st.submit(crop(img, dy), f"f{i}")
        st.wait_idle(60)
        poller.join(10)
        assert not errors, errors
        assert all(s[1] <= 60 for s in shapes)
        assert st.mosaic.count >= 3
    finally:
        st.stop()


def test_flush_writes_a_snapshot_mid_scan(tmp_path):
    img = texture()
    st = _pipeline(tmp_path)
    try:
        st.process(crop(img, 0), "a")
        st.process(crop(img, 150), "b")
        paths = st.flush()
        assert Path(paths["png"]).exists()
        # the scan can keep going after a manual snapshot
        assert st.process(crop(img, 300), "c").ok
    finally:
        st.stop()


# ── real data (skipped when the sample scan is absent) ───────────────────────
def _real_pairs(n=6):
    files = sorted(SCAN_DIR.glob("col01_Ydown_[0-9]*.jpg"))
    return files[:n]


def test_real_pair_registration_matches_measured_motion():
    files = _real_pairs(4)
    if len(files) < 2:
        return "skipped: sample scan not available"
    cfg = LiveConfig()
    reg = Registrar(cfg)
    for p in files:
        reg.prepare(p.name, cv2.imread(str(p)))
    for prev, cur in zip(files, files[1:]):
        r = reg.register(prev.name, cur.name)
        assert r is not None
        dx, dy = r.shift
        # measured on this scan: one motor step moves 210-490 px, mostly in Y
        assert 150 < abs(dy) < 550, f"{cur.name}: implausible dy={dy}"
        assert abs(dx) < 120, f"{cur.name}: implausible dx={dx}"
        assert r.ncc > 0.8


if __name__ == "__main__":
    import tempfile

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
            else:
                fn()
            print(f"PASS {name}")
        except Exception as exc:                       # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
