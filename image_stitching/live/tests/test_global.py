"""End-to-end test of the global stitcher on a synthetic scan with known truth.

The synthetic scan reproduces what the recorded scans showed:

* serpentine path (down pass, X step, up pass) with an open-loop actuator:
  every Y step is a different length and drifts sideways a little,
* autofocus frames that sit part of a step further along the previous move
  (the stage had not stopped when the frame before them was grabbed),
* optically clipped corners and vignetting,
* some motion-blurred frames.

Checks: nothing is dropped, every frame is placed within a few px of the truth,
and the mosaic matches the ground-truth surface (band-passed NCC).

Run: ``python image_stitching/live/tests/test_global.py``
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]      # image_stitching/
sys.path.insert(0, str(ROOT))

from live.global_stitch import GlobalConfig, GlobalStitcher, list_scan  # noqa: E402

FW, FH = 640, 360


def surface(h=2700, w=1100, seed=11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(128, 26, (h, w)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 1.5)
    for _ in range(260):
        x, y = int(rng.integers(10, w - 10)), int(rng.integers(10, h - 10))
        cv2.circle(base, (x, y), int(rng.integers(3, 12)), float(rng.integers(20, 60)), -1)
    for _ in range(40):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(base, p1, p2, 210, int(rng.integers(1, 3)))
    base = np.clip(base, 0, 255).astype(np.uint8)
    col = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR).astype(np.float32)
    col[:, :, 0] *= 0.8                        # a little colour
    return np.clip(col, 0, 255).astype(np.uint8)


def optics(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot(xx - w / 2, yy - h / 2) / (w / 2)
    vig = 1.0 - 0.18 * r ** 2
    out = frame.astype(np.float32) * vig[:, :, None]
    out[r > 1.0] = 0                            # clipped corners (radius = w/2)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_scan(folder: Path, seed: int = 5):
    """Write a scan folder; return {file name: true top-left (x, y)}."""
    rng = np.random.default_rng(seed)
    big = surface()
    truth: dict[str, tuple[float, float]] = {}
    t = [0]

    def save(tag, x, y, blur=0.0):
        xi, yi = int(round(x)), int(round(y))
        f = big[yi:yi + FH, xi:xi + FW].copy()
        assert f.shape[:2] == (FH, FW), (tag, xi, yi)      # stay on the surface
        if blur:
            f = cv2.GaussianBlur(f, (0, 0), blur)
        t[0] += 1
        hh, mm, ss = 20, t[0] // 60, t[0] % 60
        name = f"col01_{tag}_{hh:02d}{mm:02d}{ss:02d}.jpg"
        cv2.imwrite(str(folder / name), optics(f), [cv2.IMWRITE_JPEG_QUALITY, 95])
        truth[name] = (float(xi), float(yi))

    x, y = 60.0, 40.0
    save("Ydown_1", x, y)
    for i in range(2, 13):
        step = rng.uniform(110, 190)
        x += rng.uniform(-4, 4)
        y += step
        save(f"Ydown_{i}", x, y, blur=2.5 if i in (6, 7) else 0.0)
        if i % 4 == 0:                          # AF frame: the rest of the move
            save(f"Ydown{i}_AF", x + 6, y + 0.4 * step)
    x += 250
    save("Xr_move", x, y)
    save("Xr_AF", x + 3, y + 2)
    for i in range(1, 12):
        step = rng.uniform(110, 190)
        x += rng.uniform(-4, 4)
        y -= step
        if y < 0:
            break
        save(f"Yup_{i}", x, y, blur=2.0 if i == 5 else 0.0)
    return big, truth


def run(tmp: Path):
    big, truth = make_scan(tmp)
    cfg = GlobalConfig(work_width=640, label_scale=4, threads=4, lens_frames=60)
    st = GlobalStitcher(cfg, log=lambda *_: None)
    st.load(tmp)
    st.register_all()
    paths = st.compose(str(tmp / "out"))
    st.write_reports(str(tmp / "out"), paths)
    return big, truth, st, paths


def test_global_stitch_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        big, truth, st, paths = run(tmp)
        names = [f.path.name for f in st.frames]

        # 1. nothing dropped
        assert len(names) == len(truth), (len(names), len(truth))
        assert paths["frames_used"] >= 0.8 * len(names)

        # 2. every frame where it really is (relative to frame 0)
        gt = np.array([truth[n] for n in names])
        est = st.T - st.T[0] + gt[0]
        err = np.linalg.norm(est - gt, axis=1)
        assert err.max() < 4.0, {names[k]: round(float(e), 1)
                                 for k, e in enumerate(err) if e >= 4.0}
        assert not st.placed_by_prior, st.placed_by_prior
        rot = np.degrees(np.abs(st.RS[:, 0]))
        assert rot.max() < 0.3, rot.max()

        # 3. the mosaic is the surface (find the offset, compare band-passed)
        mosaic = cv2.imread(paths["png"])
        cov = cv2.imread(paths["coverage"], 0) > 0
        g = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY).astype(np.float32)
        G = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ph, pw = 300, 300
        cy, cx = g.shape[0] // 2 - ph // 2, g.shape[1] // 2 - pw // 2
        res = cv2.matchTemplate(G, g[cy:cy + ph, cx:cx + pw], cv2.TM_CCOEFF_NORMED)
        _, peak, _, loc = cv2.minMaxLoc(res)
        assert peak > 0.9, peak
        oy, ox = loc[1] - cy, loc[0] - cx
        h = min(g.shape[0], G.shape[0] - oy)
        w = min(g.shape[1], G.shape[1] - ox)
        a = g[:h, :w]
        b = G[oy:oy + h, ox:ox + w]
        m = cov[:h, :w]
        m = cv2.erode(m.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0

        def hp(x):
            return x - cv2.GaussianBlur(x, (0, 0), 6)
        x, y = hp(a)[m], hp(b)[m]
        ncc = float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))
        assert ncc > 0.8, ncc

        # 4. clipped corners never reach the mosaic: no black inside coverage
        lum = a[m]
        assert np.percentile(lum, 0.5) > 8, np.percentile(lum, 0.5)

        # 5. the move model learned the AF offset as its own move type
        assert "AF:Ydown" in st.motion
        assert st.motion["AF:Ydown"][0][1] > 30


def test_scan_order_is_capture_order():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _, truth = make_scan(tmp)
        frames = list_scan(tmp)
        names = [f.path.name for f in frames]
        assert names == list(truth.keys())
        moves = [f.move for f in frames]
        assert moves[0] == "Ydown" and "AF:Ydown" in moves and "Xr" in moves


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:                    # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failures else 0)
