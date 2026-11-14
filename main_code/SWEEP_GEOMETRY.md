# Sweep geometry: choosing the step sizes from measurement

The sweep moves the part under the camera along two axes and photographs it.
Everything downstream - the live mosaic, the global re-stitch, whether a hole is
counted once or twice - depends on **one** number per axis: how much of the
previous frame the next frame still contains, i.e. the **overlap**.

Until now that number was never known. Two constants in `_sweep_worker` decided
it:

```python
x_steps = mm_to_x_steps(0.4)     # 320 stepper pulses per column
y_duration_ms = 1200            # a 1.2 s pulse on the Y actuator
```

Neither is an overlap. The first is a distance that was never converted to
pixels; the second is a *time*, and time is not distance on a cam-driven
actuator. This document records what those constants actually produce on this
rig, and why the code now measures instead of assuming.

---

## 1. What the constants actually do

Measured with the same registrar the live stitcher uses (`Registrar.register`),
on the two recorded scans in `image_stitching/from_scratch` (1920x1080 frames):

### Y axis - fixed 1200 ms pulse

| scan | pairs | min \|dy\| | p10 | **median** | p90 | max | implied overlap |
|---|---|---|---|---|---|---|---|
| `test_images` | 30 / 31 | 3 | 215 | **323 px** | 473 | 669 | 38 % - 81 % |
| `scan_20261103_203842` | 31 / 31 | 233 | 247 | **339 px** | 433 | 449 | 58 % - 78 % |

Two things to see in that table, and they are not the same thing:

* the **median** is ~330 px of 1080, i.e. ~69 % overlap - which sounds fine;
* the **spread** is the problem. In the first scan one step moved 669 px (38 %
  overlap) and the next moved 3 px. A stitcher does not need overlap *on
  average*; it needs overlap *on every pair*. The 3 px pair is the signature of a
  step that did not happen, and the 669 px pair is the signature of a step that
  happened twice.

The global solver measures the same spread as a per-move `sigma` of 130-150 px,
which is the honest step-to-step band: **55 % - 81 % overlap** on a rig whose
average is 68 %. That is why `y_overlap_range()` exists and why the report prints
a band instead of a number.

### X axis - "0.4 mm" = 320 pulses per column

| scan | column | measured step | px / mm (nominal) | X overlap of 1920 px |
|---|---|---|---|---|
| `test_images` | col01 | 272.9 px | 682 | 86 % |
| `test_images` | col02 | 318.4 px | 796 | 83 % |
| `test_images` | col03 | 337.7 px | 844 | 82 % |
| `scan_20261103_203842` | col01 | 452.6 px | 1132 | 76 % |
| `scan_20261103_203842` | col02 | 400.4 px | 1001 | 79 % |

Two conclusions, both of which shaped the code:

1. **X was far more conservative than it needed to be.** At 78 % overlap each
   new column re-photographs four fifths of the previous one. The stitcher needs
   a fraction of that; the rest is pure scan time and pure storage.
2. **px/mm is not a constant.** 683 vs 1132 for the same nominal command is a
   37 % disagreement *between scans of the same rig*. Either the belt slipped,
   the nominal steps/mm is wrong, or the "0.4 mm" was never 0.4 mm. All three
   are possible and none can be distinguished afterwards - because the scan
   folder did not record what command produced it.

That is why every scan folder now gets a `geometry.json` next to
`capture_log.csv`:

```json
{
  "phase": "end",
  "y_time_s": 1.32, "y_speed": 70,
  "x_step_mm": 0.9, "x_return_mm": 0.9,
  "target_y_overlap": 0.65, "target_x_overlap": 0.5,
  "px_per_mm": 1064.2, "measured_y_px": 343.6, "measured_x_px": 425.7,
  "commands_at_start": "Y 1.20s @ PWM 70 | X 0.40 mm ...",
  "x_steps_per_mm": 800, "frame_hw": [1080, 1920]
}
```

`phase: "start"` is written when the sweep begins and `phase: "end"` when it
finishes, because auto-tune may change both steps mid-scan and a record that
only captured the intent would be a lie about the frames in the folder.

---

## 2. Why a closed loop is possible at all

Neither axis reports its position: the Y actuator is an open-loop PWM pulse and
the X stage is a stepper with no encoder. There is no sensor to read.

There is, however, a camera. **The image displacement between two frames is a
direct measurement of what the motor did**, it costs nothing extra (the stitcher
was already computing it for every frame), and it is available twice per scan:

| loop | source | when | alpha | trust |
|---|---|---|---|---|
| coarse | live `Decision.dx/dy` | during the scan | 0.35 | greedy, half-resolution preview |
| accurate | `GlobalStitcher.motion` | after the scan | 0.6 | every frame, globally solved |

Both are wired into `main_code/scan_geometry.py` (`ScanGeometry.observe` and
`ScanGeometry.absorb_motion`), a module with no Tk and no camera dependency, so
it is unit-tested against the recorded data (`test_scan_geometry.py`, 81 checks).

### The gate that makes the live loop safe: `single_step()`

Feeding raw displacements into a calibration is wrong, and the recorded data
shows exactly why: an `AF` frame is captured *part-way along* the move that
preceded it (`Xr_move -> Xr_AF` measures only 50.7 px, while the X move itself is
426 px). Blending those in inflates the spread enormously and biases the
estimate down.

`single_step(prev_tag, tag)` therefore returns an axis only when the two tags are
**consecutive captures of one move family**:

```
col01_Ydown_14_...  ->  col01_Ydown_15_...    "Y"    (one Y move)
col01_Ycor_AF_...   ->  col01_Xr_move_...     "X"    (the column turn)
col01_Ydown_14_...  ->  col01_Ydown_16_...    None   (a frame was dropped: 2 moves)
col01_Ydown4_AF_... ->  col01_Ydown4_AF_...   None   (no move at all)
col01_Xr_move_...   ->  col01_Xr_AF_...       None   (AF sits mid-move)
col01_Xr_AF_...     ->  col01_Xr2_move_...    None   (reverse move)
```

Everything else - a stale anchor, a merged duplicate, a cooling-hole row locking
onto its neighbour - is refused by `_plausible()`:

* not finite, or under 1 px: refused;
* **more than 75 % of the frame in one step**: refused (measured steps top out
  at 62 %; anything beyond that is a mis-registration, not a step);
* more than 3x away from the axis's current estimate: refused (a skipped frame
  or a wrong match);
* px/mm outside 150-4000: refused as a calibration.

Refused samples are **rejected, not smoothed**. An EWMA cannot recover from a
500 px outlier - it just takes twenty samples to walk back. And one observation
may move a command by at most `max_adjust = 0.25`, so no single frame can double
the step.

### What the loops actually change

```
suggested_y_time  = y_time_s * clamp((frame_h * (1 - target_y_overlap)) / measured_y_px,
                                     0.75, 1.25)
suggested_x_step  = frame_w * (1 - target_x_overlap) / px_per_mm
```

Targets: **65 % Y**, **50 % X**. X is kept deliberately higher than the strict
minimum because the column link is the only link between columns, so it carries
every bit of accumulated drift.

---

## 3. What the measurement says to do

From the real 176-frame scan (`scan_20261103_203842`), after the global solve:

```
$ python -c "... GlobalStitcher.load/register_all ... absorb_motion ..."

{'y_px': 343.6, 'x_px': 425.7, 'px_per_mm': 1064.2}
Y 344 px -> 68% overlap, want 65% (55%-81% step to step)  ·  X 426 px -> 78% overlap,
want 50%  ·  1064 px/mm (1.8 mm field)
recommended for the next scan: Y pulse 1.32 s, X step 0.90 mm
```

| | today | recommendation | why |
|---|---|---|---|
| Y pulse | 1.20 s | **1.32 s** | 344 px is 3 % short of the 378 px a 65 % target wants |
| X step | 0.40 mm (320 pulses) | **0.90 mm (720 pulses)** | 0.90 mm x 1064 px/mm = 958 px = 50 % overlap |
| Y frames per column | 10 for a 3-frame-tall part | 9 | 11 % fewer |
| columns for a 3.6 mm part | 6 | 3 | 50 % fewer |
| **captures for the same coverage** | 100 % | **~43 %** | fewer than half |

Note that the Y pulse moves in the *safing* direction: the loop asks for slightly
longer moves because the measured step is slightly short of target. It does not
chase the 38 %-overlap outliers - it corrects the mean, and the band is reported
so the operator knows the mean is not the whole story.

A caveat worth stating plainly: `px_per_mm` is the ratio of *measured pixels* to
*commanded millimetres*. If the nominal 800 steps/mm is wrong, then "0.90 mm" is
nominal too - the true quantity is the overlap, which is measured. The mm column
is a convenience for the operator, not a metrology claim.

---

## 4. How it is wired into the GUI

`main_code/main3.py`, Sweep tab, **SCAN GEOMETRY** card (above AUTO SWEEP):

* entries: Y pulse (s), Y PWM, X step (mm), target Y overlap (%), target X
  overlap (%), "Auto-tune from the measured step";
* a live advice line: `Y 344 px -> 68% overlap, want 65% (55%-81% step to step)
  ·  X 426 px -> 78% overlap, want 50% · 1064 px/mm (1.8 mm field)`;
* **Use recommendation** writes the suggestion into the entries;
* **Recalibrate** throws the calibration away and restores the defaults.

Data flow:

```
_save_sweep_image ─ first frame ─► self._frame_shape (needed for any overlap)
                    └─ _stitch_submit ─► LiveStitcher worker thread
                                             │ Decision(dx, dy, status)
                                             ▼
                              _on_stitch_decision  (stitcher thread!)
                                 └─ _scan_geo_observe ─ single_step ─ observe
      _sweep_worker: tune() after each autofocus and each column
                                 └─ apply() -> y_ms() re-read per move
      _start_global_stitch ─► stitch_folder ─► paths["motion"], paths["frame_shape"]
                                 └─ _scan_geo_from_motion ─ absorb_motion -> save
```

Three invariants, all of which are violated by the obvious implementation:

1. **The stitcher callback never blocks a scan.** `_on_stitch_decision` does the
   arithmetic inline (it is a few float operations) and wraps it in
   `except Exception: pass`. A geometry bug must cost a log line, never a sweep.
2. **The worker thread never touches a Tk variable.** `_scan_geo_sync()` reads
   the entries and is therefore called on the Tk thread, in `_start_autosweep`,
   *before* the worker starts; the worker only reads the plain dataclass. Results
   going the other way travel through `self._post(...)`.
3. **A correction applies to the move that follows it**, not the next column:
   `y_ms()` is re-read inside the move loop, while `x_advance` deliberately
   re-reads `x_steps` and records `self._last_x_cmd_mm` - the measurement is
   only attributable if the command that produced it is known.

`stitch_folder()` grew two additive return keys (`motion`, `frame_shape`) so the
GUI can calibrate from the solve without a second pass over 176 frames.

---

## 5. What this does *not* fix

* **The Y band.** 55 %-81 % is a mechanical property of an open-loop PWM pulse on
  this actuator. Sizing the pulse to the mean overlap makes the band narrower in
  *overlap* terms but does not narrow it in *pixels*. Fixing it properly needs a
  finer Y pitch (a lead screw) or a position sensor; shortening the stride is the
  only software lever, and the recorded stride test says stride 2 gives 38 %
  mean overlap with 161 inliers (usable) while stride 3 gives 23 % with 18
  (not usable). Stride 2 is therefore only worth trying on a scan where the
  measured `p10` allows it - verify per scan, do not hardcode it.
* **Backlash on the column turn.** `x_return_mm` exists for it but is only
  mirrored from `x_step_mm`; no measurement currently sets it.
* **The first scan after a rig change.** The loop needs a measurement to work
  from, so the first scan of a new lens/resolution runs on the last calibration
  and corrects as it goes. That is why the defaults (1.20 s / 0.40 mm) are
  conservative rather than optimal.

---

## 6. Files

| file | role |
|---|---|
| `main_code/scan_geometry.py` | the model: commands, measurements, suggestions, `geometry.json` I/O, report section. numpy + json only. |
| `main_code/test_scan_geometry.py` | 81 checks against the recorded measurements; plain `python test_scan_geometry.py`. |
| `main_code/main3.py` | SCAN GEOMETRY card, `_scan_geo_*` controller, `tune()` in `_sweep_worker`, hooks in `_on_stitch_decision` / `_start_global_stitch`. |
| `image_stitching/live/global_stitch.py` | `stitch_folder()` now returns `motion` and `frame_shape`. |
| `main_code/test_ui_headless.py` | drives the card headless: entries -> controller, live sample in, AF frame refused, solve absorbed, recommendation applied, record written. |
