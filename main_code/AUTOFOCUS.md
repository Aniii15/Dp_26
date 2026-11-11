# Autofocus for a curved blade

## What was wrong

The delivered mosaic has a band where **no** detail is resolvable. That band is
not a stitching artefact — it is a band of the *blade* that was never in focus —
and the old autofocus could not have discovered that, for four reasons.

**1. It asked the wrong question, about one region.** `_autofocus` hill-climbed
Z one step at a time, asking the Siamese CNN "is B sharper than A?" on a single
middle-third crop. On a curved blade different regions of the frame are at
different depths (your top view: the surface recedes across the field of view),
so at any one Z some regions are sharp and some are not. A single scalar
preference cannot express "sharp here, soft there", so it cannot optimise for
*the blade* rather than *a patch*.

**2. The searched range was fixed and the result was never checked against it.**
Every Z position in the loop is `current ± step`, with a maximum of 20 steps of
the coarsest size. If the true focus is outside that span, every nearby position
looks equally bad, `prob < 0.4` fires, and the code logs **"Peak at current
position"** and continues. Nothing ever asked *"is my best sample at the edge of
what I searched?"* A focus that could not be reached was indistinguishable from a
focus that had been found. That is the reported bug.

**3. A hill climb needs a hill.** When the start point is far from focus the
curve is flat; the CNN returns ~0.5 (its untrained value), and the loop's
`prob < 0.4` test is a coin toss. Worse, on a flat curve the loop happily climbs
20 × 400 steps (5 mm) in whichever direction it happened to start.

**4. The stitcher then embedded the blur.** Frames were weighted only by distance
from their own edge, so an out-of-focus region of an acceptable frame was blended
at full strength wherever it landed on empty canvas. The *second* frame to cover
that area did not correct it — the first blur was already there, and the mosaic
averaged towards white haze.

## What replaces it

`main_code/focus_engine.py` — no hardware, no PyTorch, fully testable.

### Score regions, not frames
The frame is split into a tile grid (6×4 by default) and each tile gets an
exposure-normalised Tenengrad score, `mean(|∇I|²) / mean(I)²` — the same formula
as `image_stitching/live/quality.sharpness`, so numbers mean the same thing in
both places. Each tile is then normalised by *its own* best score over the
Z-stack, which makes tiles with different texture comparable: "0.95" means "95 %
of the best this patch can ever look". Tiles with no texture (background,
blown-out specular) are excluded from voting, judged relative to the best tile in
view rather than by an absolute threshold — a blurred blade still has 3× the
luma-banding of a blank wall.

### Focus on the left 40 % of the view (`roi`)
`FocusConfig.roi = (x0, y0, x1, y1)` (fractions) lays the tile grid over that
rectangle only, so the peak, the area-in-focus and the depth spread all describe
that region. The sweep uses `FOCUS_CONFIG["roi"] = (0.0, 0.0, 0.40, 1.0)`,
shown as a yellow box on the live feed.

Why: measured over a whole scan, sharpness falls 5-10x from the left-centre of
the frame to the right edge in *every* frame - the blade curves away from the
lens on the right, so no single Z makes the frame sharp. Optimising the whole
frame trades the sharp left for a half-sharp middle. The right side is covered
again one X step later by the *left* part of the next strip, and the global
stitcher picks the sharpest view of every spot, so a reliably sharp left band
per frame is what the mosaic needs.

Stacking (`compose_focus_stack`) and the stitch weights (`tile_weight_map`)
ignore the ROI and always use the whole frame. Tests:
`test_roi_*` in `focus_tests.py` (left vs right ROI land ±0.3·tilt apart on a
tilted synthetic blade, exactly where those regions are sharp).

### The search: gallop → bracket → refine
Three stages, each of which stops as soon as it knows enough:

1. **Gallop** (only when there is no usable prediction). Sample at ±`coarse_steps`,
   then ±2×, ±4×… so the whole travel is reachable in ~8 captures per side
   instead of 120. It stops at the first sample that is actually sharp
   (`min_peak_sharpness`) — resolution only matters once the neighbourhood is known.
2. **Bracket walk.** From that point, step outward at `coarse_steps` and suspend
   each side the moment its own samples fall `bracket_margin` below the best
   score seen. A peak that is bracketed by worse samples on both sides is
   *found*; anything else is not. The walk is allowed to keep climbing past the
   gallop's hit, because a sample that is merely "sharp" is not the peak.
3. **Refine.** A fine sweep (`fine_steps`) over ±`coarse_steps`/2 around the
   winner, then a parabola through the three best samples for sub-step accuracy.

The gallop is skipped when a prediction exists, because the trend along the scan
path is normally good to a few hundred steps.

| situation | what happens |
|---|---|
| peak bracketed by worse samples on both sides | reported as locked, with the numbers |
| walk reaches the `Range cap` without a bracket | *"stopped at the Range cap (±1.88 mm, 6000 steps) without a bracketed peak — raise Range if the focus is further away, or 0 for the whole travel"* |
| walk reaches `z_travel_limit` | *"travel limit reached … the true focus is outside the reachable range"* |
| the gallop stepped over a narrow focus cone | detected by the *spread* of its scores (5-10×, not noise) and rescanned |
| no signal anywhere (spread ≈ noise) | *"nothing in view is in focus at any reachable Z"* |

A peak only counts when it stands clear of **both** ends of the window *and* its
absolute sharpness is real. On a hopelessly defocused field the curve wobbles by
±30 % from noise alone; without the second test that wobble is "the peak".

### Optimise the area in focus
The chosen Z maximises the fraction of usable blade area whose tiles are within
90 % of their own best. When several regions cannot be sharp together it says so
and quantifies it:

```
z=+900 steps (+0.56 mm), area in focus 33%, DOF 0.09 mm, spread 0.30 mm, STACKING NEEDED
  • Only 33% of the blade is in focus at this Z. …
  • Depth spread 439 steps (0.27 mm) exceeds the measured depth of field 145
    steps (0.09 mm): no single Z can make the whole field sharp — bracket Z and stack.
```

The depth of field is **measured** from the scan itself (median Z span a tile
stays above 90 % of its peak, interpolated), so no lens data is needed.

### Predict, so a wide search stays affordable
`FocusTrend` fits `z = f(column, row)` over previous stops. With a hint the next
stop searches a ±300-step window instead of the whole travel, and escalates
immediately if the peak lands on an edge — so a wrong prediction costs a few
captures, never accuracy.

### Stack when depth exceeds depth of field
If `depth_spread > 0.75 × DOF` the engine captures a 5-exposure bracket around
the peak and fuses it: each tile takes its pixels from the frame where that tile
is sharpest, weights decaying exponentially with the shortfall, feathered inside
one tile to hide the seams. The composite also carries a per-pixel confidence
map.

### Teach the stitcher who is sharp
`LiveStitcher.submit(..., weight=...)` accepts that confidence (or
`focus_engine.tile_weight_map` for a single frame). It multiplies the feathered
edge weight, so in an overlap **the sharp neighbour wins the pixel** and a
blurred region of an otherwise useful frame can no longer be baked in. This is
what stops a soft first frame from owning a patch of the mosaic forever.

## Measured on a synthetic curved blade

`mosaic_quality_check.py` simulates a full scan (3 columns × 5 stops) of a
surface that recedes 2.4 mm across the field, through the real
`IncrementalMosaic`, and measures how many regions of the finished mosaic are
blurred (below 40 % of the mosaic's own median sharpness):

| strategy | median sharpness | worst region | blurred regions |
|---|---|---|---|
| narrow window, commit (what the old code did) | 0.006 | 0.001 | **41.7 %** |
| wide region-wise scan, single Z | 0.015 | 0.002 | 8.3 % |
| wide scan + Z-stack + stitcher weights | **0.029** | **0.012** | **0.0 %** |

The middle column is the honest answer to "why not just widen the range": it
helps, but 2.4 mm of surface curvature cannot be flattened by any single Z.
`/home/user/out/focus_final/compare_band.png` is the visual of the same crop
from the three mosaics.

## Parameters — what each one does, and what it costs

Every Z quantity is in motor steps on M1: **1600 steps = 1 mm**, so 200 steps =
0.125 mm. Cost is per focus scan, i.e. per stop where the sweep calls autofocus.

| Parameter (UI) | Config key | Default | What it does | Cost if you change it |
|---|---|---|---|---|
| `Range cap` | `range_steps` | 6000 (±1.9 mm) | **Hard cap** on the whole search, either side of where the scan starts. `0` = use the whole `Travel limit`. Not a distance travelled — a scan stops as soon as the peak is bracketed | Sets the worst case *and* the bound: tighten it to the blade's real depth envelope and nothing outside that envelope is ever sampled |
| `Predict window` | `predictive_window` | 600 | Width of the window confirmed when a prediction exists (±300). Skipped when it fails | Smaller = faster when the prediction is right, more fall-backs when it is wrong |
| `C` (coarse) | `coarse_steps` | 200 (0.125 mm) | Step of the bracket walk, and the resolution to which focus is bracketed | Captures per side ≈ distance/C. 300 instead of 200 is ~35 % fewer captures and 0.19 mm resolution |
| `F` (fine) | `fine_steps` | 50 (31 µm) | Step of the refinement pass | ~4-6 captures regardless; larger steps = less accurate final Z |
| `D` (settle) | `settle_s` | 0.20 s | Wait after each Z move before grabbing the frame. **Every capture pays this** | The single biggest per-capture cost. 0.12 s instead of 0.20 s saves ~40 % of the wall time — as long as the stage really has stopped |
| `Backlash` | `backlash_steps` | 0 (off) | Approach every target from below before sampling | Doubles the moves (not captures) when enabled |
| `Travel limit` | `z_travel_limit` | 24000 (15 mm) | Hard safety stop for the whole search | Safety only |
| — | `max_samples` | 60 | Hard cap on captures in one scan | Bounds the worst case |
| — | `in_focus_ratio` | 0.90 | A tile counts as "in focus" at 90 % of its own best. Drives the area-in-focus objective and the measured depth of field | Lower = more permissive = fewer stops flagged as needing a stack |
| — | `min_peak_sharpness` | 0.01 | Absolute sharpness below which nothing counts as focused. 0.3-0.7 = sharp, 0.0006 = hopelessly defocused (measured) | Raise if a blurred frame is ever accepted; lower on very low-contrast parts |
| — | `bracket_margin` | 0.05 | How far a sample must fall below the best before its side of the walk is closed | Smaller = walks further on noise; larger = may stop before the real peak |
| — | `gallop_signal` | 2.5 | Score spread that distinguishes "focus cone stepped over" from "nothing is focusable" | Rarely needs changing |
| — | `tiles_x` / `tiles_y` | 6 × 4 | Region grid used to score the field | More tiles = better spatial diagnosis, slightly more CPU per capture |

Module-level switches in `main3.py`:

| Switch | Default | Effect |
|---|---|---|
| `FOCUS_MODE` | `"region"` | `"cnn"` restores the old hill climb |
| `FOCUS_STACKING` | `True` | Bracket + fuse when the depth spread exceeds the depth of field |
| `FOCUS_STACK_SAMPLES` | 3 | Exposures per stack. 3 costs 2 extra captures per stop and fuses about as well as 5 |
| `STITCH_FOCUS_WEIGHTS` | `True` | Give the stitcher a per-pixel focus weight (cheap; leave on) |

## Making autofocus faster

Measured on the synthetic rig, captures per scan (each capture ≈ move + settle +
grab):

| scenario | before | now |
|---|---|---|
| focus already near the stage position | 37 | 9 |
| focus 0.94 mm away | 37 | 15 |
| focus 1.5 mm away | 37 | 25 |
| predicted focus correct (every X advance, every later Y stop) | 11 | 7 |
| predicted focus 0.5 mm wrong | 27 | 13 |

What to change, in the order that pays:

1. **Set `Range cap` to the blade's depth envelope.** Use **📏 Measure depth of
   field** once, read the reported *spread*, and set `Range cap` to roughly
   2 × spread (the cap is ±, so ±spread is enough with margin). On a blade that
   spans 1 mm of depth, `Range cap = 3000` makes wandering further physically
   impossible instead of merely unlikely.
2. **Make sure a prediction is used.** A full search only happens at the first
   stop of a column, or after a big jump. If the log says
   `with no hint` repeatedly, the trend has too few points — that is the
   expensive path.
3. **`D` (settle) down.** 0.20 → 0.12 s removes 0.08 s × every capture. Check it
   by eye once: if frames come out smeared, put it back up.
4. **`C` (coarse) up.** 200 → 300 steps (0.19 mm) is still finer than the depth
   of field of most setups and cuts the walk by a third.
5. **`Predict window` down** (600 → 400) if the predictions are good.
6. **`FOCUS_STACK_SAMPLES` 3** (already the default) and `FOCUS_STACKING` off if
   you would rather re-shoot the soft stops than pay 2 captures per stop.
7. **Fewer autofocus calls per column**, which is a sweep decision rather than an
   autofocus one: the sweep currently focuses after every 4th Y move plus at each
   column turn. Focusing after every 8th Y move halves the cost.

Use the **preset buttons** in the Focus tab: *Fast (sweeps)* = 4000/300/100/0.12,
*Balanced* = 6000/200/50/0.20 (defaults), *Accurate (single shot)* =
8000/120/40/0.30.

### "It searches a much bigger Z range than needed"

Until this was addressed there was **no knob for that**: `Range` bounded only the
bracket walk, while the gallop that hunts for focus was bounded solely by
`Travel limit` (15 mm by default). So a scan could and did wander far outside the
part's depth envelope. Now:

* **`Range cap` is a hard bound on everything** — gallop, walk and refinement.
  Nothing outside `start ± Range cap/2` is ever sampled, which is enforced against
  an absolute window anchored at the start of the scan (bounding each walk by an
  offset from its *own* centre was not the same thing: a gallop could start 1 mm
  out and the walk then add its own allowance on top).
* **`0` means "whole travel"**, for when the envelope is genuinely unknown.
* When the cap is what stopped the search it says so, in operator units:
  *"stopped at the Range cap (±0.62 mm, 2000 steps) without a bracketed peak —
  raise Range if the focus is further away, or 0 for the whole travel"*.
* The result panel now reports the **footprint** of every scan:
  `searched ±0.38 mm in Z, stage travelled 1.70 mm over 12 captures (cap ±0.94 mm)`.
  That turns "it searches too much range" into a number you can act on.

If a *predicted* stop still shows a full-range search, the trend has too few
points (`len(trend) < 3`); that only happens on the first stop of a column.

Two related things worth knowing:

* The number in the log is **Z steps** (1600/mm), and it describes focus travel,
  not the X axis. An X column step is 0.4 mm of sideways motion.
* An X advance used to trigger a full-range Z search: `x_advance()` called
  autofocus with `hint=None`, so every column turn walked the whole Z travel even
  though the surface depth changes by far less than one coarse step over 0.4 mm of
  X. It now passes `hint=0.0` — "focus is where you left it" — the difference
  between 37 and 7 captures per column turn.

## Using it

Focus tab → **Autofocus**:

* `Engine` — *Region-wise (recommended)* or *CNN hill climb (legacy)*.
* `Range cap` — hard limit on the whole Z search (±Range/2 from the start);
  `0` searches the whole travel.
* `Predict window` — width confirmed around a predicted focus.
* `Travel limit` — refuse to drive Z further than this (safety).
* `Backlash` — set to ~2× the measured backlash; the stage is then always
  approached from below, which matters for a 31 µm fine pass on a worn stage.
* `C/M/F/D` — coarse / medium / fine step and settle delay.
* `Preset` — *Fast (sweeps)* = ±0.94 mm cap / 0.19 mm walk / 0.12 s settle;
  *Balanced* = ±1.9 mm / 0.125 mm / 0.20 s;
  *Accurate (single shot)* = whole travel / 75 µm / 0.30 s.
* **🎯 Auto-Focus Z** — one scan at the current position.
* **📏 Measure depth of field** — a single scan that just reports DOF and the
  blade's depth spread. Run it once per blade type; the numbers tell you whether
  you need to close the aperture or enable stacking.

`FOCUS_STACKING`, `FOCUS_STACK_SAMPLES` and `STITCH_FOCUS_WEIGHTS` at the top of
`main3.py` switch the stacking and weighting on/off. The sweep uses all of it
automatically: trend hints between stops, stacking when a stop needs it, and
focus weights on every frame handed to the stitcher.

## Tests

```
python main_code/focus_tests.py          # 27 tests, synthetic blade + fake rig
python main_code/mosaic_quality_check.py # the table above, + PNGs
python main_code/test_ui_headless.py     # whole GUI against a stub tkinter,
                                         # incl. the autofocus path end to end
python image_stitching/live/tests/test_live.py   # 15 stitcher tests
```

The focus tests cover: tiles scoring sharp > blurred; a tilt producing per-tile
peaks 400+ steps apart; featureless tiles excluded from voting; focus found
inside the window; focus *outside* the travel limit reported loudly; window
extension when the peak is on an edge; the travel limit respected; a predictive
hint costing far fewer frames but a *wrong* hint still escalating to the right
answer; fine pass within 50 steps; backlash always arriving from below; a missing
camera and a featureless scene raising instead of pretending; DOF and depth
spread tracking their synthetic ground truth; a fused stack beating every single
frame; and the trend extrapolating along a column.

## Known limits

* **Open-loop Z.** Positions are motor steps from the scan's origin; there is no
  encoder. Backlash (`backlash_steps`) is the mitigation, and `refine_peak`
  avoids re-approaching from the far side.
* **Negative-parallax regions.** If the blade curves so steeply that neighbouring
  tiles need Z 1 mm apart, one stack of five exposures cannot cover it at 0.9
  aperture-stop steps — increase `FOCUS_STACK_SAMPLES` or stop the blade down.
* **The CNN is still there.** It is a fair tie-breaker near focus but it cannot
  find focus from far away; leave the engine on *Region-wise* for sweeps.
* **First frame of a column** has no trend point, so it pays for a full-range
  search (~20-40 captures ≈ 10-20 s at 0.25 s settle). Every later stop is
  cheap.
