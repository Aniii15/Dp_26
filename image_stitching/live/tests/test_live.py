"""Tests for the live stitcher.

Run with ``python -m pytest image_stitching/live/tests -q`` or directly
``python image_stitching/live/tests/test_live.py``.

The synthetic tests pin down the behaviours that the recorded scans exposed;

* translation recovery from a known shift,
* rejection of a black / flat / defocused frame,
* rejection of a duplicate (stalled motor),
* holes being filled by soft frames without overwriting good pixels,
* the session surviving a frame the pipeline cannot process.

The optional real-data tests use ``sample_scans/test_images`` when present and
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

SCAN_DIR = ROOT.parent / "sample_scans" / "test_images"


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
    cfg = LiveConfig(flush_every=0, lens_radius_frac=0)   # synthetic: no optics
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
    mo = IncrementalMosaic(LiveConfig(flush_every=0, lens_radius_frac=0))
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
    kw.setdefault("lens_radius_frac", 0)          # synthetic frames: no optics
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


# ── lens, scale, gating, dead reckoning ─────────────────────────────────────
def test_lens_mask_keeps_clipped_corners_out_of_the_mosaic():
    """This camera's right corners are black in every frame (outside the
    image circle).  They must not be blended - they made dark wedges."""
    img = texture(1080, 1920)
    yy, xx = np.mgrid[0:1080, 0:1920]
    img[np.hypot(xx - 960, yy - 540) > 0.56 * 1920] = 0      # optical clipping
    mo = IncrementalMosaic(LiveConfig(flush_every=0))         # default lens on
    mo.add(img, np.eye(3))
    mosaic, coverage, _ = mo.render()
    assert coverage[540, 960] > 0
    assert coverage[0, -1] == 0 and coverage[-1, -1] == 0, "clipped corner blended"
    # and nothing black leaks in from the rim
    inside = coverage > 0
    assert mosaic[inside].mean(axis=1).min() > 5


def test_accum_scale_really_shrinks_the_canvas():
    img = texture()
    full = IncrementalMosaic(LiveConfig(flush_every=0, lens_radius_frac=0))
    half = IncrementalMosaic(LiveConfig(flush_every=0, lens_radius_frac=0,
                                        accum_scale=0.5))
    T = np.array([[1, 0, 0], [0, 1, 300], [0, 0, 1]], float)
    for mo in (full, half):
        mo.add(img, np.eye(3))
        mo.add(img, T)
    a, _, _ = full.render()
    b, _, _ = half.render()
    assert abs(b.shape[0] - a.shape[0] / 2) <= 2 and abs(b.shape[1] - a.shape[1] / 2) <= 2
    assert half._wsum.size < 0.5 * full._wsum.size       # memory, not just pixels
    assert half.bbox() == full.bbox() or         all(abs(p - q) <= 2 for p, q in zip(half.bbox(), full.bbox()))


def test_bandpass_ncc_catches_a_wrong_shift_that_plain_ncc_accepts():
    """Measured on the real scans: a 77 px misregistration still scores plain
    NCC ~0.90 because illumination dominates.  Band-passed NCC must not."""
    img = texture(900, 900)
    yy, xx = np.mgrid[0:400, 0:600].astype(np.float32)
    shade = 0.35 + 0.65 * np.exp(-((xx - 300) ** 2 + (yy - 200) ** 2) / (2 * 260.0 ** 2))
    a = (crop(img, 0) * shade[:, :, None] * 0.6 + 60 * shade[:, :, None]).astype(np.uint8)
    b = (crop(img, 150) * shade[:, :, None] * 0.6 + 60 * shade[:, :, None]).astype(np.uint8)
    cfg = LiveConfig(work_width=600)
    reg = Registrar(cfg)
    reg.prepare("a", a)
    reg.prepare("b", b)
    r = reg.register("a", "b")
    assert r is not None and abs(r.shift[1] - 150) < 3
    assert r.ncc_hp > 0.5, r.ncc_hp
    from live.registration import highpass
    wrong = r.affine.copy()
    wrong[1, 2] += 40
    plain, _ = reg._verify(reg.work_gray("a"), reg.work_gray("b"), wrong)
    hp, _ = reg._verify(highpass(reg.work_gray("a")), highpass(reg.work_gray("b")),
                        wrong, centered=True)
    assert plain > 0.5, plain                  # plain NCC is fooled ...
    assert hp < cfg.min_ncc_hp, hp             # ... band-passed NCC is not


def test_unregisterable_frame_is_dead_reckoned_not_dropped(tmp_path):
    img = texture()
    other = texture(seed=99)                   # sharp, but unrelated content
    st = _pipeline(tmp_path)
    try:
        st.process(crop(img, 0), "a")
        st.process(crop(img, 100), "b")
        d = st.process(crop(other, 150), "c")
        assert d.ok and d.status == "filled", (d.status, d.reason)
        assert d.method == "dead_reckoned"
        assert abs(d.pose[1, 2] - 200) < 15, d.pose      # last pose + last step
        assert st.mosaic.disconnected == 0
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
