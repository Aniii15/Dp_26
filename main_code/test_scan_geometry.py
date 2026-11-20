"""Tests for scan_geometry - the closed loop on the sweep's step sizes.

Plain asserts, no test framework (same style as focus_tests.py), because the
machine that runs this has a spare Python and nothing else:

    python test_scan_geometry.py

Every number used here is measured, not invented: the Y samples and the X step
come from the two recorded scans in ``sample_scans/`` (1920x1080
frames), so a failure means the geometry maths no longer agrees with the data
the rig actually produced.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scan_geometry as sg

FRAME = (1080, 1920)          # (h, w) - every recorded scan
CHECKS = [0, 0]


def check(name: str, cond: bool, detail: str = "") -> None:
    CHECKS[0] += 1
    if cond:
        CHECKS[1] += 1
        print(f"  PASS {name}")
    else:
        print(f"  FAIL {name}" + (f"  [{detail}]" if detail else ""))


def near(a, b, tol=1e-6) -> bool:
    return a is not None and abs(a - b) <= tol


# ── 1. tag parsing: which captures are one single step apart ────────────────
def test_single_step():
    print("\nsingle_step() - only consecutive captures of one move count")
    # the plain case: two consecutive Y moves of a column
    check("Ydown_14 -> Ydown_15 is one Y step",
          sg.single_step("col01_Ydown_14_155152",
                         "col01_Ydown_15_155210") == "Y")
    # the column turn: Xr_move follows Ycor_AF directly in the sweep
    check("Ycor_AF -> Xr_move is one X step",
          sg.single_step("col01_Ycor_AF_154902",
                         "col01_Xr_move_155003") == "X")
    # ...and the pair the registrar cannot solve must not become one
    check("Xr_move -> Xr_AF is not a step (AF sits mid-move)",
          sg.single_step("col01_Xr_move_155003",
                         "col01_Xr_AF_155101") is None)
    check("Xr_AF -> Xr2_move is not a Y or X step",
          sg.single_step("col01_Xr_AF_155101",
                         "col01_Xr2_move_155230") is None)
    check("AF -> AF is not a step",
          sg.single_step("col01_Ydown4_AF_154740",
                         "col01_Ydown4_AF_154801") is None)
    # a dropped/failed capture makes the next displacement span two moves
    check("stride 2 (Ydown_14 -> Ydown_16) is rejected",
          sg.single_step("col01_Ydown_14_155152",
                         "col01_Ydown_16_155230") is None)
    check("first frame of a scan has no predecessor",
          sg.single_step(None, "col01_Ydown_1_150000") is None)
    # columns are their own samples: crossing into a new column is a new move
    check("Ydown_30 -> next column start is not one step",
          sg.single_step("col01_Ydown_30_160000",
                         "col02_Ydown_1_160100") is None)
    check("phase_of keeps the index and drops the timestamp",
          sg.phase_of("col02_Ydown_14_155152") == "Ydown_14")
    check("move_of names the move before the frame",
          sg.move_of("col02_Ydown_14_155152") == "Ydown")
    check("move_of: Xr_move", sg.move_of("col02_Xr_move_161010") == "Xr")
    check("move_of: an AF frame has no single move",
          sg.move_of("col02_Ycor_AF_161010") is None)


# ── 2. observations, and the outliers they must refuse ─────────────────────
def test_observe():
    print("\nobserve() - accepts steps, refuses mis-registrations")
    g = sg.ScanGeometry()
    check("a 323 px Y step is accepted", g.observe("Y", 9.0, 323.0, FRAME))
    check("measured_y_px ~= 323", near(g.measured_y_px, 323.0, 0.5),
          f"{g.measured_y_px}")
    # a locked-on cooling-hole row can produce a nonsense match; 900 px would be
    # 83 % of the frame height and would poison every later suggestion
    before = g.measured_y_px
    check("a 900 px Y sample is refused",
          not g.observe("Y", 5.0, 900.0, FRAME))
    check("...and does not move the estimate", near(g.measured_y_px, before),
          f"{g.measured_y_px}")
    check("a whole-frame displacement is refused",
          not g.observe("Y", 0.0, float(FRAME[0]), FRAME))
    check("a 900 px step is refused as impossible (> 75 % of the frame), even "
          "before it can be compared to the estimate",
          not sg.ScanGeometry().observe("Y", 0.0, 900.0, FRAME))
    check("a zero displacement is refused (dropped frame)",
          not g.observe("Y", 0.0, 0.0, FRAME))
    check("NaN is refused", not g.observe("Y", float("nan"), 300.0, FRAME))
    check("an unknown axis is refused", not g.observe("Z", 1.0, 1.0, FRAME))
    check("sample count tracks accepted samples", g.y_samples == 1,
          str(g.y_samples))

    # the X calibration: 452.6 px measured for a 0.4 mm command
    g2 = sg.ScanGeometry()
    check("X observed with moved_mm=0.4 sets px/mm",
          g2.observe("X", 452.6, -10.0, FRAME, moved_mm=0.4))
    check("px_per_mm ~= 1131", near(g2.px_per_mm, 1131.5, 1.0),
          f"{g2.px_per_mm}")
    # a 5000 px/mm reading means the command was not what was recorded
    g3 = sg.ScanGeometry()
    g3.observe("X", 100.0, 0.0, FRAME, moved_mm=0.02)
    check("an implausible px/mm (> 4000) is not stored",
          g3.px_per_mm is None, f"{g3.px_per_mm}")

    # the gate that makes AF frames harmless
    g4 = sg.ScanGeometry()
    g4.observe("Y", 4.0, 339.0, FRAME)
    # a stale anchor gives a displacement 10x too small; that is a wrong match
    check("a 30 px displacement is refused when 339 px is expected",
          not g4.observe("Y", 4.0, 30.0, FRAME))
    check("...while a plausible one still gets in (233-449 px measured)",
          g4.observe("Y", 4.0, 233.0, FRAME))


# ── 3. what the measurements mean ──────────────────────────────────────────
def test_overlap():
    print("\noverlap and the honest band")
    a = sg.ScanGeometry()
    a.measured_y_px = 323.0
    a.frame_hw = FRAME
    check("323 px of 1080 -> 70 % Y overlap",
          near(a.y_overlap(), 1 - 323 / 1080, 1e-9), f"{a.y_overlap():.4f}")
    b = sg.ScanGeometry()
    b.measured_y_px = 339.0
    b.frame_hw = FRAME
    check("339 px (scan B) -> 69 % Y overlap",
          near(b.y_overlap(), 1 - 339 / 1080, 1e-9), f"{b.y_overlap():.4f}")
    c = sg.ScanGeometry()
    c.measured_x_px = 452.6
    c.frame_hw = FRAME
    check("452.6 px of 1920 -> 76 % X overlap",
          near(c.x_overlap(), 1 - 452.6 / 1920, 1e-9), f"{c.x_overlap():.4f}")

    # the band is the argument for why an average-safe target can still gap
    d = sg.ScanGeometry()
    d.measured_y_px, d.y_spread_px = 323.0, 100.0
    lo, hi = d.y_overlap_range(1080)
    check("band low = worst case (largest step)", near(lo, 1 - 423 / 1080, 1e-9),
          f"{lo:.3f}")
    check("band high = best case (smallest step)", near(hi, 1 - 223 / 1080, 1e-9),
          f"{hi:.3f}")
    e = sg.ScanGeometry()
    check("no measurement -> no range", e.y_overlap_range(1080) is None)
    f = sg.ScanGeometry()
    f.measured_y_px, f.y_spread_px = 1075.0, 40.0     # nearly a whole frame
    lo2, hi2 = f.y_overlap_range(1080)
    check("the band is clipped to [0, 1]", 0.0 <= lo2 <= hi2 <= 1.0,
          f"{lo2},{hi2}")


# ── 4. what to change ──────────────────────────────────────────────────────
def test_suggestions():
    print("\nsuggested steps for the next scan")
    g = sg.ScanGeometry()
    g.px_per_mm = 1130.0
    g.frame_hw = FRAME
    x = g.suggested_x_step_mm(1920)
    check("50 % X overlap at 1130 px/mm -> ~0.85 mm",
          near(x, 1920 * 0.5 / 1130.0, 1e-6), f"{x:.3f}")
    check("...which is more than twice today's 0.40 mm", x > 2 * 0.40)
    g.target_x_overlap = 0.25
    check("a lower target asks for a longer step",
          g.suggested_x_step_mm(1920) > x)
    g.target_x_overlap = 0.50

    g.measured_y_px = 339.0
    t = g.suggested_y_time(1080)
    want = 1080 * 0.35
    check("Y pulse is scaled by want/measured",
          near(t, 1.20 * want / 339.0, 1e-6), f"{t:.3f}")
    g.measured_y_px = 250.0            # wants +51 %, the cap must bite
    t2 = g.suggested_y_time(1080)
    check("one observation may not stretch the pulse by more than 25 %",
          near(t2, 1.20 * 1.25, 1e-6), f"{t2:.3f}")
    # an already-correct step should not be moved
    g2 = sg.ScanGeometry()
    g2.measured_y_px = 1080 * 0.35
    check("a step that already hits the target is left alone",
          near(g2.suggested_y_time(1080), 1.20, 1e-6),
          f"{g2.suggested_y_time(1080)}")
    g3 = sg.ScanGeometry()
    check("no measurement -> no suggestion", g3.suggested_y_time(1080) is None)
    check("no px/mm -> no X suggestion", g3.suggested_x_step_mm(1920) is None)


def test_apply():
    print("\napply() - corrections stay small and leave a trail")
    g = sg.ScanGeometry()
    g.measured_y_px = 100.0            # wildly off, so the cap must act
    g.frame_hw = FRAME
    changes = g.apply(FRAME)
    check("apply reports what it changed", "y_time_s" in changes, str(changes))
    check("a single correction is capped at max_adjust (25 %)",
          near(g.y_time_s, 1.20 * 1.25, 1e-6), f"{g.y_time_s}")
    check("the log carries (old, new)", changes["y_time_s"][0] == 1.20,
          str(changes))

    g2 = sg.ScanGeometry(auto_tune=False)
    g2.measured_y_px = 100.0
    check("auto_tune off -> nothing changes", g2.apply(FRAME) == {})
    check("...and the command is untouched", near(g2.y_time_s, 1.20))

    g3 = sg.ScanGeometry()
    g3.px_per_mm = 1130.0
    ch = g3.apply(FRAME)
    check("X step is adopted and the return move follows it",
          near(g3.x_step_mm, 1920 * 0.5 / 1130.0, 1e-6) and
          near(g3.x_return_mm, g3.x_step_mm), str(ch))
    check("apply with no frame shape does nothing",
          sg.ScanGeometry().apply(None) == {})


# ── 5. the accurate loop: the global stitcher's motion models ──────────────
def test_absorb_motion():
    print("\nabsorb_motion() - calibrate from a finished scan")
    import numpy as np
    motion = {"Ydown": (np.array([4.0, 339.0]), 28.0, 31),
              "Xr": (np.array([452.0, -10.0]), 12.0, 2)}
    g = sg.ScanGeometry()
    got = g.absorb_motion(motion, FRAME, x_cmd_mm=0.4)
    check("Y is learned from the solve", near(g.measured_y_px, 339.0, 0.5),
          f"{g.measured_y_px}")
    check("sample count comes from the model's n", g.y_samples == 31,
          str(g.y_samples))
    check("X is learned from the solve", near(g.measured_x_px, 452.0, 0.5),
          f"{g.measured_x_px}")
    check("px/mm from the recorded command", near(g.px_per_mm, 1130.0, 1.0),
          f"{g.px_per_mm}")
    check("the returned dict is log-friendly",
          set(got) == {"y_px", "x_px", "px_per_mm"}, str(got))
    check("frame shape is recorded", g.frame_hw == FRAME, str(g.frame_hw))
    check("the step-to-step spread comes from the solver's sigma",
          near(g.y_spread_px, 28.0, 0.5), f"{g.y_spread_px}")

    # alpha 0.6 means one authoritative scan moves the estimate most of the way
    h = sg.ScanGeometry()
    h.measured_y_px = 400.0
    h.absorb_motion(motion, FRAME)
    check("a solve pulls the estimate towards it (alpha 0.6)",
          near(h.measured_y_px, 400 + 0.6 * (339 - 400), 0.5),
          f"{h.measured_y_px}")

    check("empty motion -> nothing learned",
          sg.ScanGeometry().absorb_motion({}, FRAME) == {})
    check("no frame shape -> nothing learned",
          sg.ScanGeometry().absorb_motion(motion, None) == {})
    # a motion model that is not a single step (a merged/duplicated move) is
    # refused instead of being averaged in
    bad = {"Ydown": (np.array([0.0, 5000.0]), 10.0, 3)}
    check("an impossible motion model is refused",
          sg.ScanGeometry().absorb_motion(bad, FRAME) == {})


# ── 6. persistence ────────────────────────────────────────────────────────
def test_roundtrip():
    print("\nJSON round-trip (the calibration must survive a restart)")
    g = sg.ScanGeometry()
    g.observe("Y", 5.0, 323.0, FRAME)
    g.observe("X", 452.6, -9.0, FRAME, moved_mm=0.4)
    g.target_y_overlap = 0.60
    d = g.to_dict()
    h = sg.ScanGeometry.from_dict(d)
    check("frame_hw comes back as a tuple of ints",
          h.frame_hw == FRAME and isinstance(h.frame_hw[0], int), str(h.frame_hw))
    check("measurements survive", near(h.measured_y_px, g.measured_y_px) and
          near(h.px_per_mm, g.px_per_mm))
    check("targets survive", near(h.target_y_overlap, 0.60))
    check("unknown keys are ignored (forward compatible)",
          sg.ScanGeometry.from_dict({"y_time_s": 1.5, "whatever": 3}).y_time_s
          == 1.5)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "scan_geometry.json")
        check("save() succeeds into a fresh directory", g.save(path))
        check("load() reads it back", sg.ScanGeometry.load(path) is not None)
        check("a missing file loads as None",
              sg.ScanGeometry.load(os.path.join(tmp, "nope.json")) is None)

        scan = os.path.join(tmp, "scan_20261103_203842")
        p = sg.write_scan_geometry(scan, g, extra={"frames": 176})
        check("geometry.json is written into the scan folder",
              p and os.path.exists(p), str(p))
        with open(p) as f:
            raw = json.load(f)
        check("the extra facts are in the record", raw.get("frames") == 176)
        rec = sg.read_scan_geometry(scan)
        check("the record reads back as a ScanGeometry",
              isinstance(rec, sg.ScanGeometry) and
              near(rec.x_step_mm, g.x_step_mm),
              repr(rec))
        check("a scan folder without a record reads as None",
              sg.read_scan_geometry(os.path.join(tmp, "empty")) is None)


# ── 7. the report section ─────────────────────────────────────────────────
def test_lines():
    print("\ngeometry_lines() - what goes into the report")
    import numpy as np
    g = sg.ScanGeometry()
    g.observe("Y", 4.0, 339.0, FRAME)
    g.observe("X", 452.0, -10.0, FRAME, moved_mm=0.4)
    text = "\n".join(sg.geometry_lines(
        g, {"Ydown": (np.array([4.0, 339.0]), 28.0, 31),
            "Xr": (np.array([452.0, -10.0]), 12.0, 2),
            "AF:Ydown": (np.array([13.7, 258.5]), 135.8, 8),
            "Ycor_up": (np.array([-0.2, -51.7]), 30.0, 1)}, FRAME))
    check("AF frames and corrections are counted, not tabulated as steps",
          "AF:Ydown" not in text and "Ycor_up" not in text, text)
    check("...and the report says how many were left out",
          "2 further move types" in text, text)
    for needle in ("Sweep geometry", "Ydown", "Xr", "overlap", "px/mm",
                   "recommended"):
        check(f"report mentions {needle!r}", needle in text)
    empty = "\n".join(sg.geometry_lines(sg.ScanGeometry()))
    check("an unmeasured rig still produces a readable section",
          "not measured" in empty or "no frame size" in empty, empty)
    check("describe() is a single line",
          "\n" not in sg.ScanGeometry().describe())


def main() -> int:
    for fn in (test_single_step, test_observe, test_overlap, test_suggestions,
               test_apply, test_absorb_motion, test_roundtrip, test_lines):
        fn()
    passed, total = CHECKS[1], CHECKS[0]
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
