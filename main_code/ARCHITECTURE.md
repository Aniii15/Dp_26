# Architecture: from a stepper pulse to a finished mosaic

A tour of this repository **bottom-up**: start at the metal, end at the mosaic.
Each section says *what the layer is made of*, *what it is for*, and - the part
that is easy to lose - *why it exists at all*, i.e. what breaks if you delete it.

Written for someone who has never seen the project. Tier marks are only a
suggestion of reading order:

* **[basics]** - the mental model, no code needed
* **[intermediate]** - the actual modules and their contract
* **[nuance]** - the specific bug each design decision is defending against

---

## 0. The problem, in one paragraph [basics]

A turbine blade is inspected by putting it under a microscope and photographing
it. One photograph shows ~1.8 mm across (`geo px_per_mm ≈ 1064`, 1920 px wide).
A blade is hundreds of millimetres long. So the part has to be *swept*: move,
photograph, move, photograph, in a serpentine pattern, and then glue the
photographs back together - a mosaic - so that a human (or a defect model) can
look at the whole surface at once.

Four things have to work, and they fight each other:

1. **Motion** - three motors plus a linear actuator, driven over WiFi by an
   ESP32, with no position feedback anywhere.
2. **Focus** - the surface is tilted and the lens has ~0.1 mm of depth of field,
   so most photographs are partly blurred unless Z is searched per frame.
3. **Coverage** - consecutive frames must overlap enough to be registered, with
   no gaps, without spending an hour on a 30 cm part.
4. **Reconstruction** - the overlap must be resolved into one seamless,
   correctly-exposed, sharp image, not a collage of visibly-pasted rectangles.

Speed is the constraint that makes all of this hard: the scan must not take
longer than the operator will tolerate, and nothing may ever be allowed to
*delay a motor move* - a late move means a bad frame, and a bad frame means a
hole in the mosaic.

---

## 1. The machine and the wire [basics]

`main_code/main3.py`, top of file:

```
M1  Z axis     1600 steps/mm   focus (up/down)
M2  rotation   1600 steps/rev  (unused by the sweep today)
M3  X axis      800 steps/mm   column advance
Y   linear actuator, BTS7960, PWM 70  row advance  (time-driven, no steps)
```

The ESP32 runs an HTTP server. Every command is a **GET with the parameters in
the URL** - no protocol, no framing, no acknowledgements:

```python
http_get(f"{BASE_URL}/move_x?steps=320&dir=1")   # tell the motor to go
http_get(f"{BASE_URL}/status")                   # ask if it is still going
```

`move_z`, `move_x`, `move_r`, `move_actuator`, `set_speed`, `emergency_stop`
are thin wrappers. `_wait_move_done()` is the important one:

> **[nuance]** The ESP32 does not say "done". It says "moving"/"not moving", and
> it only says that if you ask. So every move is followed by polling until the
> *camera* stops changing (`MotionDetector`, below). Timing the move by
> `time.sleep()` instead is the classic bug: a 15 % slower move leaves the
> camera in motion when the frame is grabbed, and the frame is blurred in a way
> no autofocus can fix.

---

## 2. Frames: what the camera gives, and what it costs [intermediate]

`CAMERA_INDEX = 0`, an HDMI capture card seen as a webcam. `_get_frame()` reads
`cv2.VideoCapture` and returns BGR `np.uint8`. Every frame in both recorded
scans is **1920x1080**.

Two derived measurements drive everything downstream:

* **`MotionDetector`** - counts changed pixels between frames; `is_moving()` /
  `is_still()` / `motion_since()`. Uses a threshold triple
  (`MOTION_THRESHOLD = 20000`, `FRAME_DIFF_THRESHOLD = 20`, `STILLNESS_FRAMES = 5`).
  This is the only closed loop the *motors* have.
* **`detect_boundaries()`** - threshold gray > `BOUNDARY_THRESHOLD` to separate
  blade from background, then report left/right/upper percentages. The sweep
  uses these percentages to know when a column is finished or the part ends, so
  the operator never types a number of rows.

**[nuance]** The left edge of the recorded frames is a *dark strip*, and it is
not a stitching defect: it is off-blade background. Boundary detection is what
tells the sweep that, and `LensModel` (later) is what stops that strip from
being baked into the mosaic as a black seam. Two different mechanisms, easy to
confuse when looking at a bad mosaic.

---

## 3. Focus: `main_code/focus_engine.py` (1301 lines) [intermediate]

The largest single idea in the repo: **focus is per-region, not per-frame.**

```
tile_metrics(frame)      →  per-tile Tenengrad/sharpness, on a grid
AutoFocus.search()       →  coarse 200-step walk → bracket the peak → fine 50-step
FocusStack               →  3 exposures fused by tile into one sharp frame
FocusTrend               →  extrapolate the peak along a column (predict the next Z)
tile_weight_map()        →  a 0..1 per-pixel "how much do I trust this pixel"
```

Why it is not `cv2.Laplacian(frame).var()` and a hill climb:

**[nuance]** A blade is *tilted*. A single frame can have the left third in focus
and the right third blurred - a frame-wide sharpness number peaks at a Z where
*nothing* is sharp. So the engine scores **tiles** and requires a tile to be
*featureless-aware*: a tile with no texture is not allowed to vote (it has no
peak, and its noise floor dominates the score). `FocusResult.recommendation()`
then reports "N % of tiles in focus" and whether stacking is needed.

**[nuance]** The band of blur that used to be visible in finished mosaics came
from exactly this: a region of an otherwise-acceptable frame was out of focus,
and the stitcher blended it at full strength because distance-from-edge was the
only weight it had. `tile_weight_map()` (and `STITCH_FOCUS_WEIGHTS = True`) hands
a per-pixel focus confidence into the blend, so the sharp neighbour wins the
overlap. `compose_focus_stack()` is the alternative path: fuse 3 exposures at the
same (x, y) and keep one all-sharp frame.

**[nuance]** `range_steps` is a **hard bound**, not a distance: `0` means the
whole travel, and a scan stops as soon as the peak is bracketed. It is what stops
a bad prediction from driving Z into the part. `FocusTrend.predict()` reduces the
cost further: an X advance changes surface depth by far less than one coarse
step, so the column-turn autofocus passes `hint=0.0` and skips the coarse walk.
(Passing no hint there made every column turn pay for a full search - measured as
the single biggest cost in a sweep.)

---

## 4. The sweep: `_sweep_worker` [intermediate]

Serpentine, boundary-guided:

```
for each column:
    Y down until the lower boundary is > 10 %       (Ydown, every i%4 → autofocus)
    Y corrective move up
    autofocus                                  → Ycor_AF
    X right one step, autofocus                → Xr_move, Xr_AF
    Y up until the upper boundary is > 20 %          (Yup, autofocus every 4)
    Y corrective move down
    autofocus                                  → Yup_AF
    X right one step, autofocus                → Xr2_move, Xr2_AF
    stop when the right boundary > 10 %
```

Every capture is saved as `colNN_<phase>_HHMMSS.jpg` **and** appended to
`capture_log.csv`.

**[nuance]** File names only carry `HH:MM:SS`, so two frames in the same second
(an AF frame and the next step) are ambiguous, and a scan across midnight is
worse. `capture_log.csv` exists because the global stitcher *needs* the true
capture order to know which frame is a neighbour of which. It is preferred over
name-sorting whenever it exists (`list_scan()`).

**[nuance]** Nothing the live stitcher rejects is ever deleted. The log lists
every saved frame; the deliverable (section 7) is rebuilt from all of them.

**Step sizes** are the subject of `SWEEP_GEOMETRY.md`. Summary: they used to be
two constants (320 X pulses, a 1200 ms Y pulse), measured now, closed-loop, from
the stitcher's own registration. Constants are in
`main_code/scan_geometry.py: ScanGeometry` and the numbers live in
`geometry.json` inside each scan folder.

---

## 5. The live stitch: `image_stitching/live/` [intermediate]

Seven modules, each doing one thing. Together they turn a growing folder of
frames into an incrementally-growing preview **without ever delaying the sweep**.

```
pipeline.py     LiveStitcher: submit() → queue → worker thread → blend
mosaic.py       IncrementalMosaic: canvas, placement, weighting, flush
registration.py Registrar: phase-correlation (+ high-pass) → shift + score
quality.py      sharpness / directional blur / black-frame rejection
lens.py         LensModel: radial validity mask + flat field, fitted from the scan
config.py       LiveConfig: every tunable, with defaults that matter
run_live.py     replay a folder through the pipeline (no hardware)
```

Why it is not `cv2.Stitcher_create()`: the earlier
`image_stitching/using_opencv_funcn/balde_stitch_inbuilt_funcn.py` was that
attempt (removed; it is in the git history). OpenCV's stitcher assumes *features* (SIFT/ORB) and
*rotational* camera motion. A blade at 100x has almost no features, and the
motion is a pure translation of a few hundred pixels - roughly 99 % of the frame
is identical between neighbours. Phase correlation on a high-passed image is
both cheaper and more accurate here.

`Decision` is the pipeline's verdict on each frame:

```python
status: "accepted" | "rejected" | "dropped" | "filled"
dx, dy   # measured displacement, full-res pixels  ← used by the scan geometry
ncc      # normalised cross-correlation of the match
overlap  # fraction of the frame that landed on existing canvas
```

**[nuance]** `submit()` must be **non-blocking**, and the pipeline must
**self-disable** on error: the sweep is the priority, and a stitching bug that
raises into the capture path would stop the machine mid-column. This is why the
GUI calls `_stitch_submit()` inside `try/except` and sets `self._stitcher = None`
on any failure - the sweep continues, mosaicking stops.

**[nuance]** `accum_scale = 0.5` for the live preview. A 30 cm part at 1064 px/mm
is ~319 000 x ~11 500 px ≈ 3.7 Gpx ≈ 11 GB at RGB8. Full-resolution accumulation
OOM-killed the sandbox during development (`max_canvas_px = 120e6` is the
guard). The live mosaic is a *preview*; the deliverable is section 7.

---

## 6. `lens.py` - the palest defect in the history of this project [nuance]

The HDMI card clips the corners of the field and vignettes the right edge by
~85 %. Early mosaics had **pale rectangles** where a clipped frame was blended
over good pixels. Two wrong fixes were tried and rejected:

* raising `exposure_gain_clip` - the pale block is not an exposure error;
* rejecting black-bordered frames - it throws away the frame's good 80 %.

`LensModel` is the right fix: fit a **radial validity mask** plus a flat field
*from the scan itself* (the scan contains the same surface at many positions, so
the per-pixel vignetting is recoverable), then mask invalid pixels out of the
blend instead of blending them. `LensModel.resized()` exists because the live
mosaic works at `accum_scale` while the global one works at full resolution.

---

## 7. The global stitch: `global_stitch.py` (1261 lines) [intermediate → nuance]

This is the **deliverable**. It runs after the sweep, on the whole folder, in the
background, and it is allowed to take minutes.

```
list_scan(folder)      → ordered ScanFrames (from capture_log.csv if present)
_contextualise()       → each frame's "move": Ydown/Yup/Xr, or AF:<prev move>
pairwise registration  → Edge(i, j, m) for every plausible neighbour pair
_learn_motion()        → per-move-type displacement model  {move: (median, σ, n)}
solve_graph()          → robust IRLS over *all* edges → one pose per frame
compose()              → Potts-MRF sharpest-pixel choice + global gain solve
write_reports()        → report.txt, poses.csv, jpg/tif
```

Four decisions that make it work where the live stitcher cannot:

1. **Every frame is used, and poses are solved globally.** The live stitcher is
   greedy: it places each frame against 2 anchors and never revisits. Drift over
   30 rows is unavoidable that way. `solve_graph()` distributes the residual over
   the whole graph, so a column's accumulated error is corrected by the links
   above and below it.
2. **A motion model per move type.** The rig repeats the same move hundreds of
   times, so the *statistics* of what a `Ydown` does are a prior: median ± σ.
   `_prior()` then rescues a pair whose registration contradicts its neighbours -
   **[nuance]** this is what stops a locked-on cooling-hole row (which looks
   identical to the row above it) from dragging a frame 300 px sideways.
3. **`_contextualise()` gives AF frames the move they sit inside** (`AF:Ydown`).
   An AF frame is captured part-way along the move before it, so its measured
   displacement (~265 px where the move is ~340 px) would otherwise look like a
   third, mysterious step size. Grouping it with the move it belongs to is what
   makes the motion table readable - and it is why `single_step()` refuses those
   pairs when calibrating (see `SWEEP_GEOMETRY.md`).
4. **Potts-MRF + gain solve in `compose()`.** Each canvas pixel is assigned to
   one contributing frame: sharpest wins, with a smoothness term so a pixel is
   not taken from a different neighbour than its own neighbour's (that is what
   produces visible seams). Exposure is then equalised by one global least-squares
   gain solve, not by per-frame normalisation at blend time.

**[nuance]** `_PASS_ORDER` encodes the serpentine order as a *rank*, not a
timestamp: the solver's search order and the graph's plausible-neighbour set both
depend on knowing that column 3 row 12 comes after column 3 row 11 and before
column 4 row 12, which time stamps alone do not reliably say.

---

## 8. The GUI: `main_code/main3.py` (2965 lines) [intermediate]

Single file, single window, five tabs (**tabs** were an explicit request):

| tab | contents |
|---|---|
| **Sweep** | SCAN GEOMETRY card (steps + measured overlap), AUTO SWEEP card, column/timer status |
| **Focus** | autofocus parameters, range cap, stacking, result panel (Z footprint, tiles in focus) |
| **Manual** | jog X/Z/R, actuator moves, autofocus-at-cursor with ROI |
| **Calibration** | go-to-starting-point, boundary thresholds, camera/connection status |
| **Log** | everything, severity-tagged |

Layers inside the class:

* `_build_ui()` → header, status bar, visuals (live feed | live mosaic), tabs;
* `_card()` / `_section()` → the visual grammar, so every panel looks the same;
* one `ttk.Notebook`, scrollable pages, `_post()` / `_poll_ui_queue()` for
  worker→UI traffic;
* `_sweep_worker`, `_gsp_worker`, `_feed_worker`, global-stitch `work()` run in
  **daemon threads**.

**[nuance]** Tk is not thread-safe. Every worker must reach the UI through
`self._post(callable)`, which appends to a `queue.Queue` that `_poll_ui_queue()`
drains on the Tk thread. `self.after(0, ...)` from a worker looks like it works
and then segfaults under load - `test_ui_headless.py` greps the source for
`after(0,` specifically to keep it from creeping back.

**[nuance]** `_SAFETY` (the emergency-stop guard) is built *before*
`emergency_stop` exists, and `Halted` is a distinct exception type that must
never be swallowed by a broad `except Exception`. A stop that gets absorbed by
error handling is worse than no stop.

---

## 9. Tests: how this gets verified without a machine [intermediate]

| file | what it proves | needs |
|---|---|---|
| `main_code/test_ui_headless.py` | the whole GUI builds and behaves, with a **stub Tk** | numpy |
| `main_code/focus_tests.py` | 35 focus checks against a synthetic tilted surface | numpy |
| `main_code/test_scan_geometry.py` | 81 checks on the measured step data | numpy |
| `main_code/mosaic_quality_check.py` | scored mosaics for AF/stitch variants | opencv |
| `image_stitching/live/tests/test_live.py` | the pipeline, incl. real recorded pairs | opencv |
| `image_stitching/live/tests/test_global.py` | global stitch end-to-end on a synthetic scan | opencv |

The headless stub is the load-bearing trick: `_install_stub_tk()` replaces the
`tkinter` module with objects that record widget construction and fire traces,
so 312 widgets can be built with no display, and a worker/UI-thread violation is
a test failure rather than a crash in front of an operator.

**[nuance]** Tests here are written as plain `assert`s and a `check()` helper, no
pytest, because the machine that runs the GUI has a bare Python. Keeping the test
suite runnable there is worth more than the framework.

---

## 10. Recorded scans: `sample_scans/` [nuance]

`test_images/` (207 frames), `scan_20261103_203842/` (176 frames) and
`scan_20261021_214947/` (80 frames) are **real recorded scans**. Every number in
`SWEEP_GEOMETRY.md` and every "measured" constant in this repo was derived from
them; without the frames none of the calibration could be re-derived or
regression-tested. They used to sit, duplicated, in `image_stitching/from_scratch/`
and `image_stitching/using_opencv_funcn/`; one copy of each now lives here.

The first stitcher, `from_scratch/blade_stitch.py` (name-sorted order, phase
correlation against the previous frame only, paste with a soft edge), was
removed; it is in the git history. It is *why* the current pipeline looks like
it does: chained pairwise registration drifts, so the next version solved
globally.

---

## 11. Data flow, end to end [basics]

```
      operator presses "Start sweep"
                  │
   ┌──────────────▼───────────────────────────────────────────────┐
   │ _start_autosweep  (Tk thread)                                │
   │   _scan_geo_sync()  ← step sizes from the SCAN GEOMETRY card  │
   │   thread ─► _sweep_worker                                     │
   └──────────────┬───────────────────────────────────────────────┘
                  │ per move: move → poll camera → autofocus → capture
                  ▼
   blade_images/scan_<ts>/colNN_phase_HHMMSS.jpg  +  capture_log.csv
                                            +  geometry.json
                  │                                  │
      _stitch_submit (non-blocking)                 │ after the sweep
                  ▼                                  ▼
   LiveStitcher worker                      _start_global_stitch (thread)
     register → accept/reject                 list_scan → edges → motion model
     blend into preview                       solve_graph → compose → report
        │        │                                      │
        │        └─ Decision(dx,dy) ─► _on_stitch_decision
        │                                   └─► scan_geometry.observe  (coarse loop)
        └─ live_<ts>.jpg                              ▼
                                             blade_mosaic/global_scan_<ts>.jpg
                                                      │
                                             absorb_motion (accurate loop)
                                                      ▼
                                          scan_geometry.json  ← next sweep's steps
```

---

## 12. Critique of the request itself

The prompt was: *"tune the sweep step sizes (Y between frames, X between columns),
closed-loop, measured from registration rather than guessed"*, plus *"a very
large bottom-up explanation of what each part of the code does and why it
exists"*, judged by *"the stitched image being usable"*, with live stitching that
*never slows the machine*.

What was good about it, and worth keeping in future requests:

* **It named the mechanism** ("measured from registration rather than guessed"),
  not just the symptom. That is what made a real fix possible instead of a third
  round of constant-twiddling.
* **It named the axes.** "Y between frames, X between columns" is unambiguous,
  where "the step size" alone would have been a guess.
* **It stated the constraint that must not be broken** (live stitching must not
  slow the machine) and the success criterion (a usable image). Constraints and
  criteria are what let a plan be rejected.

Where it left work on the table:

1. **No failure was described, only a knob.** "Tune the step sizes" does not say
   *which* defect is being fixed: gaps between rows, blurry seams, too slow,
   too much storage? Those want opposite changes - and on this rig the honest
   answer turned out to be "Y is roughly right but wildly variable, X is 4x more
   conservative than needed", which is only visible if you know what you are
   looking at.
2. **No acceptance test.** "The stitched image must be usable" is the right
   *human* criterion and the wrong *engineering* one: it cannot be evaluated
   without the machine. Re-expressing it as "no pair of consecutive frames below
   55 % overlap, ≤ 5 % of canvas from a single frame, seam step below X" makes it
   checkable here and keeps the human criterion intact.
3. **Two very different jobs in one request** - a numerical tuning loop and a
   reference document. They pull in opposite directions: the first wants the
   smallest, most verifiable change; the second wants the widest view. Splitting
   them costs the user one more message and produces better work in both.
4. **Missing rig facts that only the operator has**: the real Y actuator's
   response to a PWM pulse (is it linear in time?), whether the claimed 800
   steps/mm is true, the true depth envelope of the blade. Any of those stated
   up front would have removed a paragraph of hedging from the result.
5. **No statement of what has already been tried.** On a rig like this, "we tried
   X and it did not work" is worth more than a page of requirements - it is what
   keeps an agent from re-proposing the same thing.
6. The phrase "do the same thing as the last prompt, again" was doing a lot of
   silent work: it referred to work that no longer exists in the repository (the
   branch was reset), in a conversation whose context had been compacted. In that
   situation the strongest prompt is a *fresh* one that restates the goal and the
   constraints - which is what this branch effectively received.

What was unusually well specified, and should stay that way: **tabs**, the
autofocus invariants (no blurred band, `Range cap` as a hard bound, `0` = whole
travel), the "consecutive captures are neighbours" stitching constraint, and
"commit only to the specified branch". Each of those is a decision an agent
cannot infer correctly from the code.

---

## 13. Where to look first when something is wrong [basics]

| symptom | first place |
|---|---|
| black or pale bands in the mosaic | `lens.py` validity mask, then `focus_weight` in `_save_sweep_image` |
| visible rectangular patches / seams | `global_stitch.compose()` (Potts-MRF), `seam_step` in the report |
| a column is smeared sideways | `_learn_motion` / `_prior` - a frame was placed against a repeated pattern |
| mosaic has a hole | `quality.assess` (was a frame rejected?) and the `capture_log.csv` order |
| gaps between rows | `SWEEP_GEOMETRY.md` - overlap band, `y_overlap_range()` |
| scan much slower than expected | `FOCUS_CONFIG.range_steps` and the column-turn `hint=0.0` |
| GUI freezes during a sweep | a `self.after()` or a widget touched from a worker |
