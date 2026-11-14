# Dp_26 — what every part of the code does, and why it exists

A turbine/blade micro-inspection machine. A DSLR-grade sensor behind an HDMI
capture card looks at a rusty blade through a macro lens; steppers and a linear
actuator move the part; software drives the motion, keeps the image sharp, and
stitches the frames into one picture you can actually inspect.

This document is the map. It is organised **bottom-up**: the physical thing
first, then the layer that talks to it, then the layer that decides, then the
layer you look at. Every section says what the code does *and* what it is
defending against, because nearly every line in this repo is a scar from a
specific failure.

```
                    ┌──────────────────────────────────────────┐
  physical world    │  blade · stage · motors · camera · lens  │
                    └───────────────────┬──────────────────────┘
                                        │ HTTP GET over WiFi
                    ┌───────────────────▼──────────────────────┐
  transport         │  main3.http_get / move_* / SafetyStop    │
                    └───────────────────┬──────────────────────┘
                    ┌───────────────────▼──────────────────────┐
  control           │  main3.MotorControlApp._sweep_worker     │
                    │  focus_engine.AutoFocus                  │
                    │  scan_geometry.ScanGeometry              │
                    └───────────┬───────────────┬──────────────┘
                    ┌───────────▼──────┐  ┌─────▼───────────────┐
  perception        │ live/LiveStitcher│  │ live/quality,frames │
                    └───────────┬──────┘  └─────────────────────┘
                    ┌───────────▼──────────────────────────────┐
  artefact          │  mosaic.png · coverage · low_quality ·   │
                    │  frames.csv · report.txt                 │
                    └──────────────────────────────────────────┘
```

---

# Part 0 · The one-paragraph version

`main3.py` is the operator's window on the machine: it holds the camera feed,
drives four motor endpoints over HTTP, and runs the sweep. `focus_engine.py`
answers *"where in Z is the part sharp, and is one Z enough?"*. `live/` answers
*"where does this frame belong in the picture, and does it deserve to be
blended?"*. `scan_geometry.py` answers *"how big should a step be?"*.
`test_*.py` and `*_tests.py` are how all of that is checked without touching the
machine. Every module exists because a specific, measurable failure happened
first; the docstrings cite the measurements.

---

# Part 1 · `main_code/main3.py` (≈2 900 lines) — the operator's machine

Everything is one file, on purpose: it is the thing the operator launches
(`python main3.py`), it must survive a missing GPU, a missing camera and a
missing network, and it is the only place that knows all four motors.

## 1.1 Constants — the physical model

```python
Z_STEPS_PER_MM = 1600     X_STEPS_PER_MM = 800      R_STEPS_PER_REV = 1600
CAMERA_INDEX  = 1         ESP32_IP = "192.168.4.1"
```

**Why**: these are the only place the physical world is described. Everything
else works in steps or pixels and converts here (`mm_to_z_steps`,
`mm_to_x_steps`, `degrees_to_r_steps`). One axis is a stepper with a known
steps/mm; **Y is a PWM linear actuator with no position at all** — which is why
Y is commanded in *milliseconds* and why `scan_geometry.py` exists.

`MOTION_THRESHOLD` / `STILLNESS_FRAMES` / `FRAME_DIFF_THRESHOLD` feed
`MotionDetector`: closed-loop "has the stage actually stopped?" by frame
differencing. **Why**: the actuator has no endstop and no feedback, so "is it
still moving" can only be answered by looking.

## 1.2 The transport layer — `http_get` and friends

```python
http_get("/z", {"steps": n, "dir": d})     move_z / move_x / move_r
http_get("/actuator", {"dir","time","speed"})   move_actuator   (the Y axis)
http_get("/speed", {"us": us})             set_speed
http_get("/stop")                          emergency_stop
http_get("/ping")                          ping_esp32
```

**Why HTTP GET to an ESP32 in AP mode**: zero setup on the operator's laptop, no
daemon, no serial port. The cost is that every command burns 5-50 ms and can
time out, which is exactly why motion is never fired from the Tk thread.

**The subtlety**: `move_z` takes *unsigned* steps + a direction flag. Passing a
signed value with `dir=0` sends a negative step count to the firmware and the
motor does something surprising. That bug is commented in `_focus_runner`.

## 1.3 `SafetyStop` — the emergency stop (new)

A **latch**, not a request. `trip()` sets an event and bumps a generation
counter; `Halted` (an exception, not a flag) unwinds whatever is running.

**Why it had to change**: the old handler set the autosweep flag and fired one
`/stop`. The sweep might be inside a 1.5 s actuator move, an autofocus bracket or
one of fourteen `time.sleep(0.5)` calls, so the machine kept moving for seconds.
Worse, a queued command could still land *after* the stop.

Four defences, in order: latch → every wait becomes `haltable_sleep()` →
every motion wrapper and motion endpoint is refused while latched → `/stop` is
retried four times in its own thread until the firmware acknowledges (and the log
shouts if it never does). `_set_motion_controls` disables motion controls by
*walking the widget tree*, so a control added later cannot be forgotten.
See `SAFETY.md`.

## 1.4 `MotorControlApp` — the window

Tk, one class, five tabs (`Sweep`, `Focus`, `Manual`, `Calibration`, `Log`).
Layout is tabs because a single scrolling page had grown unusable.

Threading rules that the whole file obeys, and that `test_ui_headless.py`
enforces:

* the **feed thread** (`_feed_worker`) owns the camera and only publishes
  `self._latest_frame` under `_frame_lock`;
* the **sweep thread** (`_sweep_worker_inner`) owns motion;
* **Tk is touched only on the Tk thread.** Worker threads hand callables over
  through `_post()`; `_poll_ui_queue` drains them 20×/s. Calling `self.after()`
  from a worker is *the* classic Tk crash and there is an explicit test that
  greps for it.

| method group | what it does | why it exists |
|---|---|---|
| `_build_*`, `_card`, `_section`, `_make_*` | widget plumbing | one visual language, no duplicated styling |
| `_poll_ui_queue`, `_post` | cross-thread hand-off | see above |
| `_feed_*`, `_get_frame` | camera | the only consumer of `cv2.VideoCapture` |
| `_wait_motion_stop` | poll `MotionDetector` | "move done" without an encoder |
| `_stitch_*`, `_on_stitch_decision`, `_mosaic_render`, `_mosaic_ui_tick` | live mosaic panel | the operator must see the mosaic *while* scanning, not after |
| `_focus_cfg`, `_focus_runner`, `_autofocus*` | bridge to `focus_engine` | keeps physics out of the engine, keeps Tk out of the search |
| `_sweep_worker_inner` | the serpentine scan | see §1.7 |
| `_gsp_worker` | "go to starting point" | calibration move sequence |
| `_log*` | log queue + colour tags | every decision is auditable |

`init_ai` / `load_siamese_model` / `predict_focus_change` / `train_siamese_pair`
are the **optional** PyTorch Siamese path: it says "is B better focused than A".
It is optional by design (`HAS_TORCH`); the region engine replaced it for
sweeps because a pairwise opinion cannot tell you how far to move.

`detect_boundaries` / `draw_boundaries`: the blade is bright, the background is
black, so a column/row profile with a threshold gives the blade edges as
percentages. **Why it matters**: the sweep is *boundary-triggered*, not
step-counted — it turns around when the lower edge is 10 % into the frame. That
absorbs a mis-mounted part.

## 1.5 The sweep, in detail

```
column start
  └─ y_sweep("down", lower edge)        move Y, wait, save, AF every 4th move
  └─ y_correct("up")                    back off the overshoot
  └─ autofocus_y                        focus + (optionally) Z-stack
  └─ x_advance("Xr")                    one column right, AF, boundary check
  └─ y_sweep("up", upper edge)          ... serpentine ...
  └─ y_correct("down"); autofocus; x_advance
```

Each `save()` writes a JPEG **and** pushes the frame to the live stitcher. The
phase tag (`Ydown_12`, `Xr_move`, `Ycor_up`) is what tells the stitcher which
axis was commanded — see Part 3.

**Why it looks like this**: the actuator has no encoder, so the Y step is a fixed
pulse; the boundary detector decides when the column is finished; the corrective
move exists because the overshoot is unpredictable. Every `haltable_sleep(0.5)`
after a move is there because a capture during settle is motion-blurred, and a
motion-blurred frame is worse than a missing one (it either fails registration or
pollutes the mosaic).

## 1.6 `_focus_*` — the bridge

`_focus_cfg()` builds a `FocusConfig` from the Tk variables, `_focus_runner()`
builds an `AutoFocus` wired to `move_z` and `_get_frame`, `_autofocus()` runs it
and reports. `_capture_focus_stack` + `_focus_weight` are the part that made the
mosaic sharp: after a Z-bracket the stack is fused **and** a per-pixel focus
weight is handed to the stitcher, so a soft region loses the blend to a sharp
neighbour instead of being baked in permanently.

---

# Part 2 · `main_code/focus_engine.py` (≈1 250 lines) — where is it sharp?

This module knows nothing about Tk or HTTP: it takes `move(delta)`,
`capture()` and `wait()` as callables. That is what makes 32 hardware-free tests
possible.

## 2.1 Why per-tile, not per-frame

A blade is curved. One frame cannot be sharp everywhere: the top of the frame may
be 0.1 mm nearer than the bottom. A single global sharpness number then reports
"the best compromise", which is the **blurred band** the operator kept seeing.

So: `tile_bounds` cuts the frame into a grid; `tile_metrics` scores each tile
(`sharpness` = exposure-normalised Tenengrad, plus contrast and brightness).
`FocusStack` holds the scores of every Z sample and is the object that answers
the real questions:

* `quality` — mean over *usable* tiles: the search signal;
* `area_in_focus` — fraction of tile area within 90 % of its own best: the thing
  you actually want maximised;
* `peaks` — per-tile best Z (sub-step, parabola);
* `dof_steps` — the *measured* depth of field: how wide a Z span a tile stays
  sharp over. Not assumed, measured, because it changes with lens and aperture;
* `depth_spread` — p90−p10 of the per-tile best Z: how much depth the field spans;
* `needs_stacking` — `depth_spread > dof`. **This one number decides whether a
  single frame is even possible.** That is the whole insight of the module.
* `normalised` — each tile divided by its own best, so a dark corner and a bright
  centre become comparable. Without it the search steers toward the brightest
  region regardless of focus.

`reliable` excludes featureless and blown tiles: a blank tile's "sharpness" is
noise, and letting it vote is how autofocus walks off the part.

## 2.2 The search: `AutoFocus.run()`

`range_steps` is a **hard bound** on the whole search, not a distance to walk
(`_search_window`, `_in_window`). `0` means "the whole travel". **Why**: the
operator's complaint was that autofocus searched a range nobody needed. Now:

1. **gallop** (only when there is no usable hint) — double the offset each round
   until the score falls, so a focus far outside the nominal window is still
   found;
2. **bracket walk** from the best sample, coarse steps, stop as soon as the peak
   is *inside* the searched window (`bracketed`). Not "walk the window then pick
   the max" — that pays for samples that cannot change the answer;
3. **fine pass** at `fine_steps` around the peak;
4. `refine_peak` fits a parabola for a sub-step answer;
5. `_check_budget` / `max_samples` stop a runaway scan;
6. `z_travel_limit` refuses to drive into the part.

Measured effect (captures per scan): 37 → 9 near focus, → 15 at a 0.94 mm cap,
→ 27 → 13 when the hint is wrong. `AUTOFOCUS.md` has the table.

The other dead ends are documented in that file because they are easy to
re-invent: absolute contrast gate, per-tile normalisation steering, unbounded
normalisation, over-bracketing, fixed-window scanning, gating refine on capture
order.

## 2.3 `FocusTrend` — the scan knows where it is

Focus moves smoothly as the sweep walks the part. `FocusTrend` fits
(predicted Z from column index and row index), so the next stop starts with a
`hint` instead of a full search. `_escalate` keeps it honest: if the peak sits on
the window edge, the next window moves there. A wrong hint costs samples, not
correctness — there is a test for exactly that.

## 2.4 `compose_focus_stack`, `tile_weight_map`

`compose_focus_stack` merges a Z-bracket into one frame that is sharp everywhere;
`tile_weight_map` produces the per-pixel 0..1 confidence the stitcher uses to
decide *who wins a pixel*. Blow-up is bilinear so the weight map has no blocky
edges of its own. Feather σ = tile/8: a bigger σ washes out the composite.

---

# Part 3 · `image_stitching/live/` — where does this frame belong?

The streaming stitcher. Frames are assessed, registered and blended as they
arrive, in a worker thread, so it never delays the motors.

## 3.1 `pipeline.py` — the decision maker

```
analyse_frame → quality gate → register against anchors → axis veto
   → pose fusion → sharpness / new-area gates → blend or reject
```

* **`Node`** — an accepted frame kept as a registration anchor (features + pose).
* **`Decision`** — the auditable verdict for one frame: `status`, `reason`,
  `dx/dy`, `ncc`, `overlap`, `new_area`, `gain`, `pose`, `ms`. It is what the
  GUI shows, what the CSV records and what `scan_geometry` learns from.
* **`LiveStitcher.submit()`** — non-blocking queue with backpressure; dropped
  frames are counted, never block the capture loop.
* **`_anchor_candidates` / `_nearest_nodes`** — the temporal predecessor is tried
  first; if the chain broke (a column turn, a rejected frame) the nearest *placed*
  frames are searched, which is how a new column links to the old one.
* **`reject(keep_as_anchor=True)`** — a duplicate is not blended but *is* kept in
  the anchor chain. Without this the next frame is matched across two motor
  steps, overlap halves, matching collapses and the scan cascades into
  "disconnected bands" (observed: 104 bands on real data).
* **`_fuse_poses`** — average the global pose implied by every successful anchor,
  so one bad pair cannot drag the canvas.
* **`park_disconnected`** — a frame with no geometric link starts a *labeled* new
  band instead of inventing a transform or being thrown away.

## 3.2 `registration.py` — bounded feature matching

ORB (or SIFT) → ratio test → RANSAC → NCC verification, with the matches
constrained to a **prior band** around the expected displacement. **Why the
prior**: the blade has a periodic row of cooling holes. Phase correlation and
ECC both lock onto whichever hole period is nearest the guess, so the answer can
be off by one pitch (~50 px) while still showing `cc = 0.98` — a silent
misregistration that shows up as doubled holes. The prior band and the
photometric NCC check (on real overlap only) are what keep it honest. ECC polish
is only run when ORB is already mediocre, because on good pairs it changes
nothing and costs a lot.

## 3.3 `prior.py` — what the stage can physically do

`StagePrior` learns, per commanded axis, the observed displacement (mean, spread,
rotation) from frames already verified. It then (a) narrows the registration
prior and (b) **vetoes** a match that contradicts the commanded axis *before* the
frame is blended. A serpentine sweep is never axis-aligned — the rig is not
square, the actuator pushes at a slight angle — so the signature is *learned*
rather than assumed, and the prior stays permissive until it has
`axis_min_samples` observations.

## 3.4 `frames.py` — what is image, and how is it lit

Three measurements taken before anything is blended:

* `analyse_frame` → **validity mask**. A clipped border/corner reads near-black;
  treated as image, the 2× exposure gain lifts it to mid-grey and it becomes a
  pale rectangle per frame. Dead pixels are excluded from every statistic.
* `flatten_illumination` → the frame's **own** smooth brightness surface (robust
  4th-order fit on block medians), divided out. Light falloff is common to all
  frames, so no amount of matching between them can remove it — matching only
  ever sees a strip. Extrapolating that strip is what drew brightness
  *rectangles* in the delivered mosaics.
* `smooth_weight` → C1 blend weight (smoothstep^γ). The old
  `min(distanceTransform, feather)` has a gradient kink that draws a line along
  the seam.

## 3.5 `mosaic.py` — the accumulator

Float32 RGB accumulator + weight sum, grown on demand, cropped on render.
Per frame: warp, mask, scalar exposure gain against the canvas, feathered
additive blend. `render()` returns `(mosaic, coverage, low_quality)`.
`probe_new_area` answers "does this frame add anything?" before paying for a
blend. `max_canvas_px` is a **safety valve, not a limit to raise casually**:
crossing it means peak memory (16 bytes/px) or, worse, the OOM killer.

## 3.6 `quality.py`, `config.py`

`quality.py`: cheap intrinsic gates (exposure, contrast, sharpness, directional
blur) that need no registration, plus `focus_ratio` — sharpness of the current
frame relative to the previous one *on identical pixels*, which removes content
bias. `config.py` is the tuning surface, and every threshold there cites the
measurement it came from. It is the first file to read when a frame is being
rejected for a reason you disagree with.

## 3.7 `run_live.py`, `purge.py`, `tests/`

* `run_live.py` — the offline/CLI entry point: `--replay DIR` to validate tuning
  on recorded scans without the machine, `--camera N` for live. `camera_loop`
  only submits after the image stops changing, which removes motion-blurred
  captures at the source.
* `purge.py` — finds frames that cannot help a stitch (near-duplicates,
  out-of-focus, low overlap) and optionally deletes them. This is the "what to
  delete" tooling from the original analysis.
* `tests/test_live.py` — 27 cases including a **real-data** registration test and
  a regression test that a scan with a clipped border stitches to the same
  picture as a clean one.
* `tests/repro_patchy_mosaic.py` — the synthetic reproducer that measures the
  "patched up" artefact against a ground-truth stitch, before vs after.

---

# Part 4 · `main_code/scan_geometry.py` — how big should a step be? (new)

Both sweep axes were commanded blind: Y in *milliseconds* of PWM (measured
result: 208-669 px for the same command), X in *millimetres* (0.4 mm → 81 %
overlap, i.e. four times more columns than the stitching justified).

`ScanGeometry` closes the loop with the only encoder available — the camera:

* `observe(axis, dx, dy, frame_shape, moved_mm)` → EWMA of the achieved pixel
  step, plus a free `px_per_mm` calibration from the X move;
* `y_overlap` / `x_overlap` / `columns_needed` → what those numbers mean;
* `suggested_y_time` / `suggested_x_step_mm` → the command that would hit the
  target overlap, clamped to ±25 % so one bad registration cannot double a step;
* `advice()` → the one line the operator reads.

Everything is arithmetic on measurements: no Tk, no camera, no global state, so
it is unit-tested. `SWEEP_GEOMETRY.md` has the numbers and the caveats.

---

# Part 5 · Tests — how we know without the machine

| file | what it protects | run |
|---|---|---|
| `main_code/focus_tests.py` | 32 cases: the focus search, Z cap, trend, stacking, DOF measurement, composition. Uses `Rig`, a fake stage + lens, so the "right answer" is known. | `python focus_tests.py` |
| `main_code/test_ui_headless.py` | builds the **whole Tk window** against a stub tkinter: every tab, every tick handler, the mosaic panel, the estop behaviour, the scan-geometry panel, and a grep-test that no worker calls `after()` | `python test_ui_headless.py` |
| `main_code/mosaic_quality_check.py` | end-to-end: does the focus work make the *mosaic* sharper? Reports blurred-region fraction for narrow / wide / stacked autofocus (41.7 % / 2.8 % / 0.0 %) | `python mosaic_quality_check.py` |
| `image_stitching/live/tests/test_live.py` | 27 cases: shift recovery, gates, duplicates, hole filling, resilience, async finalisation, validity mask, flat field, axis prior, real-data registration | `python live/tests/test_live.py` |
| `image_stitching/live/tests/repro_patchy_mosaic.py` | the artefact reproducer: error vs a clean stitch, seam step, pale-block fraction | `python live/tests/repro_patchy_mosaic.py` |

**Why these and not more**: each one encodes a failure that actually happened.
The moment a test becomes "check that the code does what the code does", it is
cost without protection.

---

# Part 6 · What is dead, what is legacy, what to be suspicious of

* `image_stitching/from_scratch/blade_stitch.py` — the original batch stitcher.
  Kept as the reference implementation and as the source of the measurements.
  **Do not use it**: it takes 16 m 42 s on the recorded folder, then dies on the
  canvas safety limit, and its ECC polish slides a correct ORB answer by 90 px
  while still reporting `cc > 0.5`.
* `image_stitching/using_opencv_funcn/` — a 42-line `cv2.Stitcher` experiment.
  Educational only; `cv2.Stitcher` assumes a rotating camera and has no notion of
  a prior.
* The Siamese CNN path in `main3.py` — still useful for small, near-focus
  corrections and it is trained in-house, but a pairwise "better/worse" opinion
  cannot say *how far* to move, which is why the region engine does the sweeps.
* `_autofocus_cnn` — the original coarse-to-fine hill climb. Kept as a fallback
  when `focus_engine.py` is absent.
* Known weak spots (all documented in the READMEs): the X step and the Y step are
  now measured but still open-loop *commanded*; pose fusion is translation-only;
  `px_per_mm` assumes a fronto-parallel surface; a column with no overlap is
  reported, not repaired.

---

# Part 7 · The reading order

If you are picking this up cold, read in this order — each step answers a
question the previous one raised:

1. `main_code/main3.py` §constants + `_sweep_worker_inner` — what the machine
   physically does.
2. `focus_engine.py` — `FocusStack.needs_stacking` next to `AutoFocus.run`: why
   one frame is not enough and how the Z search is bounded.
3. `image_stitching/live/pipeline.py::_process` — the single function that decides
   the fate of every frame. Everything else in `live/` is a helper it calls.
4. `live/frames.py` and `live/mosaic.py` — what makes the picture look right.
5. `live/config.py` — every threshold, each with the measurement behind it.
6. `scan_geometry.py` + `SWEEP_GEOMETRY.md` — how the two step sizes are chosen.
7. The test files, in the same order.
