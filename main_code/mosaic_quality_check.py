#!/usr/bin/env python3
"""End-to-end check: does the focus work actually make the *mosaic* sharper?

The unit tests in ``focus_tests.py`` prove the engine finds focus and that
stacking fuses a bracket.  This one ties it to the thing the user complained
about - the delivered mosaic - by simulating a whole scan of a *curved* blade
through the real ``LiveStitcher``:

* a synthetic surface whose depth varies across the field (a tilted blade),
* a Z stage and a lens whose blur grows with distance from focus,
* three capture strategies side by side:

``narrow``   what the old autofocus did: one hill climb inside a small window,
             committing to whatever Z it stopped at (the reported bug: focus
             lies outside the window, so a band is never sharp);
``wide``     the new region-wise scan, single Z per stop;
``stack``    the new scan plus a Z-bracket fused by ``compose_focus_stack``, with
             per-pixel confidence handed to the stitcher's blend.

Output is a per-region sharpness comparison of the three mosaics, plus the PNGs
so the result can be looked at.  Run:  python mosaic_quality_check.py [outdir]
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "image_stitching"))

import cv2                                      # noqa: E402
import numpy as np                              # noqa: E402

import focus_engine as fe                       # noqa: E402
from live import LiveConfig                     # noqa: E402
from live.mosaic import IncrementalMosaic       # noqa: E402

# ── the scene ────────────────────────────────────────────────────────────────
SURF_H, SURF_W = 2000, 1400        # the blade's front face, in surface pixels
FRAME_H, FRAME_W = 360, 480        # what the camera sees
STEP_Y = 240                       # Y advance per capture
COLS = 3                           # X columns
STEP_X = 360


def make_surface(seed: int = 11) -> np.ndarray:
    """High-detail blade face: grain, cooling holes in two rows, scratches."""
    rng = np.random.default_rng(seed)
    base = rng.normal(150, 26, (SURF_H, SURF_W)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 1.1)
    for y in range(60, SURF_H - 40, 70):               # two columns of holes
        for x in (SURF_W - 210, SURF_W - 120):
            cv2.ellipse(base, (x, y), (16, 9), 8, 0, 360, 55, -1)
            cv2.ellipse(base, (x, y), (16, 9), 8, 0, 360, 225, 2)
    for _ in range(120):                               # scratches
        x0, y0 = rng.integers(0, SURF_W), rng.integers(0, SURF_H)
        dx, dy = rng.integers(-60, 60), rng.integers(20, 120)
        cv2.line(base, (int(x0), int(y0)), (int(x0 + dx), int(y0 + dy)),
                 float(rng.integers(70, 105)), 1)
    img = np.dstack([base * 0.95, base * 0.99, base * 1.03])   # warm metal
    return np.clip(img, 0, 255).astype(np.uint8)


SURFACE = make_surface()


def depth_at(x: int, y: int) -> float:
    """Blade depth in Z-steps at a surface point - the curvature that breaks AF.

    Across the face (x) the surface recedes by ~1.8 mm; down the face (y) it is
    nearly flat, matching the top view: the front face is curved in X.
    """
    fx = x / SURF_W
    return -2400.0 * fx + 90.0 * np.sin(2 * np.pi * y / SURF_H)


def blur_view(patch: np.ndarray, z_error: np.ndarray) -> np.ndarray:
    """Defocus a patch with a per-pixel Z error (two discrete kernel passes)."""
    out = patch.astype(np.float32)
    level = np.clip((np.abs(z_error) / 70.0).astype(int), 0, 9)
    for k in range(1, 10):
        band = level == k
        if not band.any():
            continue
        bl = cv2.GaussianBlur(patch, (2 * k + 1, 2 * k + 1), 0).astype(np.float32)
        out = np.where(band[:, :, None], bl, out)
    return np.clip(out, 0, 255).astype(np.uint8)


class Stage:
    """Z stage + camera looking at the surface, with a real depth profile."""

    def __init__(self, col: int, row: int):
        self.y0 = row * STEP_Y
        self.x0 = col * STEP_X
        self.z = 0.0

    def move(self, steps: int) -> None:
        self.z += steps

    def capture(self) -> np.ndarray:
        patch = SURFACE[self.y0:self.y0 + FRAME_H, self.x0:self.x0 + FRAME_W]
        ys, xs = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
        depth = depth_at(self.x0 + xs, self.y0 + ys)
        return blur_view(patch, self.z - depth)


def stops():
    """Every capture position of the sweep, in scan order."""
    for c in range(COLS):
        for r in range(int(SURF_H / STEP_Y) - 1):
            yield c, r


# ── the three strategies ────────────────────────────────────────────────────

def af_narrow(stage: Stage) -> float:
    """The old behaviour: hill climb inside a small window, no bracketing.

    Reproduces the defect faithfully: the search covers +/-600 steps and, when
    nothing is sharp anywhere inside it, it keeps whichever end it reached -
    it has no way of knowing the true focus is outside.
    """
    z0 = stage.z
    best_z, best_score = z0, -1.0
    for dz in range(-500, 501, 100):
        stage.move(int(z0 + dz - stage.z))
        frame = stage.capture()
        score = fe.tile_metrics(frame, fe.FocusConfig()).sharpness.mean()
        if score > best_score:
            best_z, best_score = z0 + dz, score
    stage.move(int(best_z - stage.z))
    return stage.z


def af_wide(stage: Stage) -> tuple[fe.FocusResult, fe.AutoFocus]:
    cfg = fe.FocusConfig(settle_s=0, range_steps=6000, coarse_steps=200,
                         max_extensions=2, tiles_x=6, tiles_y=3)
    af = fe.AutoFocus(stage.move, stage.capture, cfg, log=lambda m: None,
                      sleep=lambda s: None, origin=stage.z)
    return af.run(), af


def af_stack(stage: Stage) -> tuple[np.ndarray, np.ndarray]:
    """Scan, then bracket around the chosen Z and fuse it into one sharp frame.

    One runner does both, so the bracket is positioned from the scan's own
    bookkeeping and its *measured* depth spread decides how wide to bracket.
    """
    cfg = fe.FocusConfig(settle_s=0, tiles_x=6, tiles_y=3)
    res, af = af_wide(stage)
    frames = af.bracket(af.position, samples=5)
    comp, conf = fe.compose_focus_stack([f for _, f in frames], cfg)
    return comp, conf


# ── mosaics ─────────────────────────────────────────────────────────────────

def build(strategy: str) -> tuple[np.ndarray, dict]:
    """Stitch every capture of the scan at its *known* pose.

    Registration is deliberately bypassed: the question here is the sharpness of
    the imagery, not ORB's ability to register a synthetic texture, and a wrong
    transform would only add ghosting that muddies the focus measurement.  The
    poses are exact translations, so all three mosaics show the same surface
    area and are directly comparable.
    """
    mosaic = IncrementalMosaic(LiveConfig(flush_every=0, accum_scale=1.0))
    kept = rejected = 0
    for c, r in stops():
        stage = Stage(c, r)
        if strategy == "narrow":
            af_narrow(stage)
            frame, weight = stage.capture(), None
        elif strategy == "wide":
            af_wide(stage)
            frame, weight = stage.capture(), None
        else:
            frame, weight = af_stack(stage)
        pose = np.array([[1, 0, stage.x0], [0, 1, stage.y0], [0, 0, 1]], float)
        # frames that share a surface area overwrite each other in the overlap,
        # exactly as they do in a real scan; the focus weight decides who wins
        try:
            mosaic.add(frame, pose, weight=weight)
            kept += 1
        except MemoryError:
            rejected += 1
    out, coverage, _lowq = mosaic.render()
    return out, {"kept": kept, "rejected": rejected, "coverage": coverage}


def sharpness_grid(mosaic: np.ndarray, rows: int = 3, cols: int = 12):
    """Per-region sharpness across the mosaic: the blurred band shows up here."""
    h, w = mosaic.shape[:2]
    out = np.zeros((rows, cols), np.float32)
    cfg = fe.FocusConfig()
    for j in range(rows):
        for i in range(cols):
            y0, y1 = int(h * j / rows), int(h * (j + 1) / rows)
            x0, x1 = int(w * i / cols), int(w * (i + 1) / cols)
            patch = mosaic[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            out[j, i] = fe.tile_metrics(patch, cfg).sharpness.mean()
    return out


def main() -> int:
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    outdir.mkdir(parents=True, exist_ok=True)
    results = {}
    for strategy in ("narrow", "wide", "stack"):
        print(f"building {strategy} mosaic ...", flush=True)
        mosaic, info = build(strategy)
        cv2.imwrite(str(outdir / f"mosaic_{strategy}.png"), mosaic)
        grid = sharpness_grid(mosaic)
        results[strategy] = (mosaic, grid, info)
        print(f"  {mosaic.shape[1]}x{mosaic.shape[0]} px, {info['kept']} placed, "
              f"{info['rejected']} over budget")

    print("\nregion sharpness (rows = across the blade, cols = down the scan)")
    print("           " + " ".join(f"{i:>7d}" for i in range(results["narrow"][1].shape[1])))
    for name, (_, grid, _s) in results.items():
        for j in range(grid.shape[0]):
            print(f"{name:>7s} r{j} " + " ".join(f"{v:7.3f}" for v in grid[j]))
        print()

    print("blurred-band indicator: regions below 40 % of the mosaic's own median")
    print("(lower is better; a visible blurred band shows up as a high percentage)")
    for name, (_, grid, info) in results.items():
        vals = grid[grid > 0]
        if not vals.size:
            continue
        med = float(np.median(vals))
        blurred = float((vals < 0.4 * med).mean() * 100)
        print(f"  {name:>7s}: median {med:7.3f}  worst {vals.min():7.3f}"
              f"  blurred regions {blurred:5.1f}%"
              f"  min/median {vals.min() / med:.3f}")
    print(f"\nPNGs written to {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
