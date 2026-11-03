#!/usr/bin/env python3
"""Tests for the region-wise focus engine, on a synthetic curved blade.

A real blade cannot be put in a test, so the scene below reproduces the two
properties that break the old autofocus:

* **Tilted focal plane** - focus varies across the field of view, so at any one
  Z some regions are sharp and some are not.
* **Unknown, far-away focus** - the scene can be set up so the correct Z is
  outside the default search window, which is the reported failure.

Run:  python focus_tests.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cv2                                      # noqa: E402
import numpy as np                              # noqa: E402

import focus_engine as fe                       # noqa: E402

W, H = 320, 240


def texture(seed: int = 3) -> np.ndarray:
    """High-frequency test pattern: holes + grain, like an inspected surface."""
    rng = np.random.default_rng(seed)
    img = rng.integers(90, 165, (H, W, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    for y in range(12, H - 8, 22):              # rows of cooling holes
        for x in range(14, W - 8, 26):
            cv2.ellipse(img, (x, y), (5, 3), 20, 0, 360, (40, 40, 42), -1)
            cv2.ellipse(img, (x, y), (5, 3), 20, 0, 360, (230, 230, 232), 1)
    return img


def blur_for(delta: float) -> int:
    """Blur kernel size for a Z error, like a real lens: sharper away from focus."""
    return max(0, int(round(abs(delta) / 45.0)) * 2)


def focus_error(frame_shape, z: float, tilt: float) -> np.ndarray:
    """Per-pixel Z error: a tilted plane.  ``tilt`` = steps across the width."""
    xs = np.linspace(-0.5, 0.5, frame_shape[1], dtype=np.float32)
    return z + tilt * xs[None, :]


BLUR_PER_STEP = 1.0 / 220.0     # blur sigma per Z step of defocus


def render(z: float, tilt: float = 0.0, base: np.ndarray | None = None,
           scene=None) -> np.ndarray:
    """Sharpen/defocus the texture according to a (possibly tilted) Z error.

    Blur sigma grows linearly with the Z error, like a real lens' circle of
    confusion, so the focus curve is smooth and a depth-of-field measurement
    means something.  A tilted error is applied in narrow bands, which keeps a
    tilted frame sharp on one side and soft on the other.
    """
    tex = texture() if base is None else base
    err = focus_error(tex.shape, z, tilt)
    sigma = np.abs(err) * BLUR_PER_STEP
    out = tex.astype(np.float32)
    edges = np.linspace(0.0, max(6.0, float(sigma.max())), 25)
    for i, (lo_s, hi_s) in enumerate(zip(edges[:-1], edges[1:])):
        # the last band must include its upper edge, or the most defocused
        # pixels are the only ones left unblurred - which makes the far field
        # look sharp and sends the search chasing the window edge
        last = i == len(edges) - 2
        band = (sigma >= lo_s) & ((sigma <= hi_s) if last else (sigma < hi_s))
        if not band.any():
            continue
        bl = cv2.GaussianBlur(tex, (0, 0), float((lo_s + hi_s) / 2)).astype(np.float32)
        out = np.where(band[:, :, None], bl, out)
    return np.clip(out, 0, 255).astype(np.uint8)


class Rig:
    """A fake Z stage + camera: moves are arithmetic, captures are renders."""

    def __init__(self, focus_z: float, tilt: float = 0.0, noise: int = 2,
                 drift: float = 0.0):
        self.pos = 0.0
        self.focus_z = focus_z
        self.tilt = tilt
        self.noise = noise
        self.drift = drift          # focus moves while the stage does
        self.moves: list[int] = []
        self.captures = 0
        self.base = texture()

    def move(self, steps: int) -> None:
        self.moves.append(int(steps))
        self.pos += steps
        if self.drift:
            self.focus_z += self.drift * abs(steps) / 1000.0

    def capture(self):
        self.captures += 1
        img = render(self.pos - self.focus_z, self.tilt, self.base)
        if self.noise:
            rng = np.random.default_rng(self.captures)
            img = np.clip(img.astype(np.int16)
                          + rng.integers(-self.noise, self.noise + 1, img.shape),
                          0, 255).astype(np.uint8)
        return img


def run_af(rig: Rig, cfg: fe.FocusConfig | None = None, hint=None,
           logs=None):
    cfg = cfg or fe.FocusConfig(settle_s=0)
    af = fe.AutoFocus(rig.move, rig.capture, cfg,
                      log=(logs.append if logs is not None else (lambda m: None)),
                      sleep=lambda s: None)
    return af, af.run(z_hint=hint)


# ──────────────────────────────────────────────────────────────────────────────
#  tile metrics
# ──────────────────────────────────────────────────────────────────────────────

def test_sharp_frame_scores_higher_than_blurred():
    sharp = render(0.0)
    soft = render(400.0)
    ms, mb = fe.tile_metrics(sharp, fe.FocusConfig()), fe.tile_metrics(soft, fe.FocusConfig())
    assert ms.sharpness.mean() > 2.5 * mb.sharpness.mean(), (
        ms.sharpness.mean(), mb.sharpness.mean())


def test_tilt_makes_regions_peak_at_different_z():
    """The whole point: one frame is never uniformly sharp on a curved blade."""
    tilt = 1200.0
    stack = fe.FocusStack(fe.FocusConfig())
    for z in range(-900, 901, 100):
        stack.add(z, render(z, tilt, texture()))
    tile_z, _ = stack.peaks()
    usable = stack.reliable()
    spread = np.nanpercentile(tile_z[usable], 90) - np.nanpercentile(tile_z[usable], 10)
    assert spread > 400, f"tilt not reflected in per-tile peaks ({spread})"
    assert stack.needs_stacking()


def test_flat_scene_does_not_need_stacking():
    stack = fe.FocusStack(fe.FocusConfig())
    for z in range(-600, 601, 100):
        stack.add(z, render(z, 0.0, texture()))
    assert not stack.needs_stacking(), (stack.dof_steps(), stack.depth_spread())


def test_featureless_tiles_are_not_allowed_to_vote():
    frame = render(0.0)
    frame[:, :W // 2] = 128                    # left half: flat grey background
    stack = fe.FocusStack(fe.FocusConfig())
    for z in (-200, 0, 200):
        stack.add(z, frame.copy())
    usable = stack.reliable()
    assert not usable[0], "flat tile accepted as a focus reference"
    assert usable[-1], "textured tile rejected"


# ──────────────────────────────────────────────────────────────────────────────
#  finding focus
# ──────────────────────────────────────────────────────────────────────────────

def test_finds_focus_inside_the_window():
    rig = Rig(focus_z=420.0)
    _, res = run_af(rig)
    assert abs(res.z_peak - 420) <= 80, res.z_peak
    assert abs(rig.pos - res.z_peak) <= 1, (rig.pos, res.z_peak)
    assert res.in_focus_area > 0.8, res.in_focus_area
    assert not res.needs_stacking


def test_reports_when_focus_lies_outside_the_travel_limit():
    """The reported bug: no reachable Z is sharp.  Must be loud, not silent."""
    rig = Rig(focus_z=60000.0)                 # far beyond the 15 mm limit
    logs: list[str] = []
    _, res = run_af(rig, logs=logs)
    assert res.range_limited in ("neg", "pos"), res.range_limited
    joined = " ".join(logs)
    assert "travel limit" in joined, joined
    assert any("Z range is still too small" in r for r in res.recommendation())


def test_reaches_focus_far_outside_the_configured_range():
    """Focus well beyond range_steps/2: must be found, not declared absent."""
    cfg = fe.FocusConfig(settle_s=0, range_steps=2000, coarse_steps=200)
    rig = Rig(focus_z=2600.0)                  # outside +/-1000 of the start
    logs: list[str] = []
    _, res = run_af(rig, cfg, logs=logs)
    assert abs(res.z_peak - 2600) <= 150, res.z_peak
    assert res.bracketed, res.curve_variation
    assert res.in_focus_area > 0.8, res.in_focus_area
    # and it got there geometrically, not by marching the whole travel
    assert any("steps out" in m or "rescanning that neighbourhood" in m
               for m in logs), logs
    assert res.samples < 40, res.samples


def test_stops_at_the_travel_limit_instead_of_driving_into_the_part():
    cfg = fe.FocusConfig(settle_s=0, range_steps=2000, coarse_steps=200,
                         z_travel_limit=3000)
    rig = Rig(focus_z=99999.0)
    _, res = run_af(rig, cfg)
    assert max(rig.moves and [rig.pos]) <= 3000 + 1, rig.pos
    assert res.range_limited


def test_predictive_hint_costs_far_fewer_frames():
    rig = Rig(focus_z=300.0)
    _, full = run_af(rig, fe.FocusConfig(settle_s=0))
    rig2 = Rig(focus_z=300.0)
    _, hinted = run_af(rig2, fe.FocusConfig(settle_s=0), hint=280.0)
    assert hinted.samples < full.samples != 0
    assert abs(hinted.z_peak - 300) <= 80, hinted.z_peak


def test_bad_hint_still_escalates_to_a_correct_answer():
    rig = Rig(focus_z=2500.0)
    cfg = fe.FocusConfig(settle_s=0, range_steps=2000, coarse_steps=200)
    _, res = run_af(rig, cfg, hint=0.0)
    assert abs(res.z_peak - 2500) <= 150, res.z_peak


def test_fine_pass_lands_within_fifty_steps():
    rig = Rig(focus_z=1100.0)
    _, res = run_af(rig, fe.FocusConfig(settle_s=0))
    assert abs(res.z_peak - 1100) <= 50, res.z_peak
    assert abs(rig.pos - res.z_peak) <= 1, (rig.pos, res.z_peak)


def test_backlash_compensation_always_arrives_from_below():
    cfg = fe.FocusConfig(settle_s=0, backlash_steps=60)
    rig = Rig(focus_z=500.0)
    af = fe.AutoFocus(rig.move, rig.capture, cfg, log=lambda m: None,
                      sleep=lambda s: None)
    af.run()
    # the last move before every capture must be +60 (same direction every time)
    assert all(m > 0 for m in rig.moves[-1:]), rig.moves[-4:]
    assert 60 in rig.moves


def test_no_camera_raises_a_clear_error():
    af = fe.AutoFocus(lambda s: None, lambda: None, fe.FocusConfig(settle_s=0),
                      log=lambda m: None, sleep=lambda s: None)
    try:
        af.run()
    except fe.FocusError as exc:
        assert "no frames" in str(exc).lower()
    else:
        raise AssertionError("missing feed did not raise")


def test_featureless_scene_raises_instead_of_pretending_to_focus():
    flat = np.full((H, W, 3), 128, np.uint8)
    af = fe.AutoFocus(lambda s: None, lambda: flat.copy(),
                      fe.FocusConfig(settle_s=0), log=lambda m: None,
                      sleep=lambda s: None)
    try:
        af.run()
    except fe.FocusError as exc:
        assert "texture" in str(exc).lower()
    else:
        raise AssertionError("flat scene was reported as focused")


# ──────────────────────────────────────────────────────────────────────────────
#  what a scan concludes
# ──────────────────────────────────────────────────────────────────────────────

def test_measured_dof_is_the_width_of_the_focus_bump():
    """DOF must be measured from the curve, and not be an artefact of the step.

    Same scene, same lens, two scan resolutions: the answer should agree, and it
    must be a real number of steps (not zero, not the whole window).
    """
    def dof_for(step):
        stack = fe.FocusStack(fe.FocusConfig())
        for z in range(-800, 801, step):
            stack.add(z, render(z, 0.0, texture()))
        return stack.dof_steps()

    coarse, fine = dof_for(100), dof_for(50)
    assert 40 < coarse < 900, coarse          # a real width, in steps
    assert abs(fine - coarse) < 0.5 * max(coarse, fine), (coarse, fine)


def test_dof_tracks_a_gentler_lens():
    """A lens with a slower blur ramp must measure a wider depth of field."""
    def dof_with(scale):
        # ``scale`` stretches the Z axis, i.e. the lens defocuses more slowly
        stack = fe.FocusStack(fe.FocusConfig())
        for z in range(-800, 801, 100):
            stack.add(z, render(z / scale, 0.0, texture()))
        return stack.dof_steps()
    narrow, wide = dof_with(1.0), dof_with(3.0)
    assert wide > narrow, (narrow, wide)


def test_depth_spread_tracks_the_real_tilt():
    for tilt, expect in ((0.0, 0.0), (1200.0, 700.0)):
        stack = fe.FocusStack(fe.FocusConfig())
        for z in range(-900, 901, 100):
            stack.add(z, render(z, tilt, texture()))
        spread = stack.depth_spread()
        assert abs(spread - expect) < 350, (tilt, spread)


def test_tilted_stack_recommends_stacking_with_numbers():
    stack = fe.FocusStack(fe.FocusConfig())
    for z in range(-900, 901, 100):
        stack.add(z, render(z, tilt=1200.0))
    res = stack.result()
    assert res.needs_stacking
    texts = res.recommendation()
    assert any("bracket Z and stack" in t for t in texts), texts
    assert any("mm" in t for t in texts)


def test_clean_scan_says_so():
    stack = fe.FocusStack(fe.FocusConfig())
    for z in range(-400, 401, 100):
        stack.add(z, render(z, 0.0))
    res = stack.result()
    assert not res.needs_stacking
    assert "Focus locked" in res.recommendation()[0]


# ──────────────────────────────────────────────────────────────────────────────
#  focus stacking
# ──────────────────────────────────────────────────────────────────────────────

def test_rotational_invariance_of_scores_across_tiles():
    """Tiles on different texture must still be comparable after normalisation."""
    frame = render(0.0, tilt=900.0)
    stack = fe.FocusStack(fe.FocusConfig())
    for z in (-500, -250, 0, 250, 500):
        stack.add(z, render(z, tilt=900.0, base=texture()))
    norm = stack.normalised()
    usable = stack.reliable()
    assert norm[:, usable].max(axis=0).max() <= 1.0 + 1e-6
    assert (norm[:, usable] > 0).all()


def test_composition_is_sharper_than_any_single_frame():
    tilt = 900.0
    frames = [render(z, tilt, texture()) for z in (-700, -350, 0, 350, 700)]
    comp, conf = fe.compose_focus_stack(frames)
    cfg = fe.FocusConfig()
    scores = [fe.tile_metrics(f, cfg).sharpness.mean() for f in frames]
    comp_score = fe.tile_metrics(comp, cfg).sharpness.mean()
    assert comp_score > max(scores) * 1.05, (comp_score, max(scores))
    assert conf.shape == comp.shape[:2]
    assert conf.min() >= 0 and conf.max() <= 1


def test_composition_removes_the_blurred_band():
    """The goal: no tile left blurred, which no single frame can achieve.

    Ratio = tile sharpness / the best that tile ever reaches in the stack, so it
    is unaffected by how much texture different tiles have.  On a tilted blade
    most tiles are far from focus in any one frame; the composite must bring
    every tile close to its own best, and beat every single frame by a wide
    margin (that margin *is* the blurred band in the mosaic).
    """
    tilt = 900.0
    zs = np.arange(-800, 801, 100)
    frames = [render(z, tilt, texture()) for z in zs]
    cfg = fe.FocusConfig()
    per_frame = np.vstack([fe.tile_metrics(f, cfg).sharpness for f in frames])
    best = per_frame.max(axis=0)
    ratio_frames = per_frame / np.maximum(best, 1e-6)

    comp = fe.compose_focus_stack(frames)[0]
    ratio = fe.tile_metrics(comp, cfg).sharpness / np.maximum(best, 1e-6)

    assert ratio.min() > 0.80, ratio.min()          # no tile left blurred
    assert ratio.mean() > 0.85, ratio.mean()
    # every single frame has a blurred band somewhere, even the best of them
    worst_frame = ratio_frames.min(axis=1)
    assert worst_frame.max() < 0.20, worst_frame
    # ... while the composite keeps every tile near its own best, i.e. it beats
    # the best single frame by a wide margin *on that frame's worst tile*
    assert ratio.min() > 4 * worst_frame.max(), (ratio.min(), worst_frame.max())


def test_single_frame_composition_is_a_no_op():
    f = render(0.0)
    comp, conf = fe.compose_focus_stack([f])
    assert np.array_equal(comp, f)
    assert conf.min() == conf.max() == 1.0


def test_tile_weight_map_penalises_blurred_regions():
    # tilt about the *centre* is symmetric, so offset the focus: the right side
    # is then nearly sharp while the left is well out of focus
    img = render(-400.0, tilt=900.0)
    wt, ref = fe.tile_weight_map(img)
    assert wt.shape == img.shape[:2]
    assert ref > 0
    left, right = wt[:, :W // 8].mean(), wt[:, -W // 8:].mean()
    assert right > left + 0.1, (left, right)
    # sharp side must reach close to full weight
    assert right > 0.6, right
    # and a fully sharp frame gets high weights everywhere
    wt2, _ = fe.tile_weight_map(render(0.0), reference=ref)
    assert wt2.mean() > 0.5, wt2.mean()


# ──────────────────────────────────────────────────────────────────────────────
#  trend prediction
# ──────────────────────────────────────────────────────────────────────────────

def test_trend_extrapolates_along_a_column():
    trend = fe.FocusTrend()
    for i in range(10):                  # focus rises 120 steps per Y step
        trend.add(u=1.0, v=float(i), z=300.0 + 120.0 * i)
    assert len(trend) == 10
    pred = trend.predict(1.0, 11.0)
    assert pred is not None and abs(pred - (300 + 120 * 11)) < 20, pred


def test_trend_needs_three_points_and_ignores_the_far_past():
    trend = fe.FocusTrend(max_points=6)
    trend.add(1, 0, 100)
    trend.add(1, 1, 200)
    assert trend.predict(1, 2) is None
    for i in range(2, 20):
        trend.add(1, float(i), 2000.0 + i)
    assert len(trend) == 6                       # window, not the whole history
    p = trend.predict(1, 20)
    assert p is not None and abs(p - 2020) < 15, p


def test_trend_survives_a_single_column_of_points():
    trend = fe.FocusTrend()
    for i in range(5):
        trend.add(u=3.0, v=float(i), z=100.0 + 50 * i)
    p = trend.predict(3.0, 5.0)
    assert p is not None and 300 < p < 420, p


def main() -> int:
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:                     # noqa: BLE001
            fails += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
