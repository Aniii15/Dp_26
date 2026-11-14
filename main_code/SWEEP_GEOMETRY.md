# Sweep geometry: choosing the Y step and the X step

Two numbers decide whether a sweep is a usable mosaic: how far the part moves
between adjacent frames (**Y**), and how far it moves between adjacent columns
(**X**). Both are now in the GUI (`Sweep` tab → **SCAN GEOMETRY**) and both are
corrected from measurement.

---

## 1. Boil it down: what is actually being controlled

```
pulse width / step count  ──►  stage travel  ──►  image displacement  ──►  overlap
        (the knob)                 (µm)                (px)                 (%)
```

Only the left end is a knob, and the metric that matters - **overlap** - is at
the right end. The intermediate arrows used to be invisible:

| axis | command | what it actually did (measured on the recorded scan) |
|---|---|---|
| Y | PWM **1200 ms** at speed 70 | moved the image **208 - 669 px** (frame is 1080 px), i.e. **38 - 81 % overlap**. A 3.2× spread from the same command: supply sag, friction, load. |
| X | **320 steps = 0.4 mm** | moved the image **~363 px** of a 1920 px frame = **81 % overlap**. Every column re-imaged 4/5 of the previous one. |

So the machine was spending 4-5× more captures than the stitcher needed, and the
Y step it *thought* it was taking was off by up to 3× in either direction.

### The camera is the encoder you already have

There is no position readout on either axis, so this looks unclosable. It isn't:
the stitcher measures the real frame-to-frame displacement of every accepted
frame (`Decision.dx/dy`, full-resolution pixels). That is a direct measurement of
what the motor did, it is free, and it was already being written to the CSV.

Two things fall out:

* **A scale calibration.** One X move of a known step count moves the image a
  measured number of pixels → `px_per_mm = 363 / 0.4 = 908 px/mm` (a 2.1 mm field
  of view). Any overlap target now converts to a command.
* **A closed loop.** Actuator travel is proportional to pulse time, so
  `new_time = old_time × target_px / measured_px` converges. Clamped to ±25 %
  per correction so one mis-registered frame (a locked-on hole row) cannot double
  the step.

---

## 2. What the numbers say you can do

Simulated with `scan_geometry.ScanGeometry` against the recorded col01 Y steps:

| Target Y overlap | converged pulse | frames per 10 000 px column |
|---|---|---|
| 65 % (default) | 1.62 s | 26 |
| 50 % | 2.06 s | 19 |
| 40 % | 2.47 s | 15 |

and for columns (4 mm strip, 908 px/mm):

| Target X overlap | step | columns |
|---|---|---|
| 81 % (what 0.4 mm did) | 0.4 mm | 6 |
| 50 % (default) | 1.06 mm | 3 |

Combined, that is roughly **one capture in three** for the same coverage.

Why 40 % Y is safe *here*: measured on the recorded pairs, stride-2 (38 %
overlap) still registered 30/30 with median NCC 0.980 and 161 inliers.

Why X stays at 50 %: the horizontal overlap is the **only** link between columns,
so it carries all accumulated drift. A column costs one X move either way; the
saving is in having fewer columns, which 50 % already delivers.

---

## 3. The nuances that decide whether this works in practice

1. **Overlap is not uniform.** The actuator's 3.2× spread means a 40 % *target*
   gives 38-81 % *actual*. The loop controls the mean. If you need a guarantee,
   either raise the target or accept the occasional gap; the report already says
   when a frame landed `unregistered`.

2. **A step you cannot see is a step you cannot trust.** Around the column turn
   there are three moves in a row (`Ycor` correction, `X`, then AF), and the
   measurement of each is polluted by the others. The controller only samples
   frames whose phase tag matches the axis (`Ydown*`, `Yup*` → Y; `Xr*` → X).

3. **The pixel scale is not constant across the part.** If the blade is tilted
   with respect to the image plane, `px_per_mm` changes with depth. The EWMA
   tracks it; a sudden change is logged as an auto-tune correction, which is also
   how you would detect a slipping coupler.

4. **Turning points get double-covered.** The sweep is boundary-triggered, so the
   last frame of a column overshoots the edge; the `Ycor` move then re-covers it.
   That is deliberate (it is what makes the turn stitcheable) but it means the
   *last* two frames of a column are not one step apart. They are excluded from
   the measurement for the same reason as (2).

5. **Autofocus cadence interacts with the step.** `AF after 4 Y moves` is scale
   free; if you halve the number of frames by doubling the step, you also halve
   the number of AF stops, and the focus trend gets sparser. `AUTOFOCUS.md`
   covers the trade; the sweep logs the AF footprint per stop.

---

## 4. Using it

`Sweep → SCAN GEOMETRY`:

| field | meaning |
|---|---|
| Y step (pulse time) | seconds of actuator drive per Y move |
| Y actuator PWM | drive strength (speed 10-255) |
| X step | millimetres of stepper travel per column |
| Target Y overlap | % of the frame shared with the previous frame |
| Target X overlap | % shared with the previous column |
| Auto-tune | adopt measured corrections between columns (and every 4th Y move) |
| Use recommendation | compute both commands from the current measurement |

Under the fields, one line tells you what is actually happening:

```
Y step 314 px → 71% overlap (want 65%) · X step 363 px → 81% (want 50%) · 908 px/mm (2.1 mm field)
```

Until the first column advance there is no X measurement, so the scale is
unknown and the line says so - it is not guessing.

### Reading the log

```
[SWEEP] geometry: Y 1.20s @ PWM 70 | X 0.40 mm | scale 908 px/mm | measured Y 314 px
[SWEEP]   Y 1200 ms per move, X 320 steps (0.40 mm)
[SWEEP] auto-tune x_step_mm: 0.4 -> 0.53 (Y step 314 px → 71% overlap ...)
```

A correction on *every* column means something is drifting (supply, belt,
friction). No corrections after the first two is the normal, healthy case.

---

## 5. What this does not do (honest limits)

* **It cannot make the actuator repeatable.** It corrects the mean, not
  frame-to-frame scatter. A real fix is a lead screw with an encoder, or
  triggering the capture from measured overlap instead of from a fixed move.
* **`px_per_mm` assumes a flat, fronto-parallel surface.** On a strongly curved
  blade it is an average; good enough for choosing a step, not for metrology.
* **The X measurement needs one successful column link.** If a column turn has
  no overlap at all, no calibration is learned from it and the report says
  `disconnected band` - that is the signal that the step is too big.
* **Stepper travel is open loop too.** `mm_to_x_steps` assumes 800 steps/mm. If
  the micro-step setting is changed on the driver, the calibrated `px_per_mm`
  will drift and the auto-tune will show it as a slow correction.
