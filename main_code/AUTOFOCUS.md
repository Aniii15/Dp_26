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

### Search a range, then refuse to lie about it
The scan covers the whole configured travel in both directions at 0.125 mm, then:

| situation | what happens |
|---|---|
| best sample is at an edge of the window | the window is extended and rescanned (`max_extensions`), following the sharper side |
| still climbing when extensions run out | reported: *"gave up after 2 extensions … more than 5.3 mm from the start"* |
| extension would exceed `z_travel_limit` | reported: *"peak is at the edge and the travel limit is reached"* |
| curve never rises above `min_peak_sharpness` | reported: *"no bracketed focus peak … nothing in view was ever in focus"* |
| peak bracketed by worse samples on both sides | reported as locked, with the numbers |

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

## Using it

Focus tab → **Autofocus**:

* `Engine` — *Region-wise (recommended)* or *CNN hill climb (legacy)*.
* `Range` — width of one search window, in Z steps (1600 steps = 1 mm).
* `Extensions` — how many times the window may grow.
* `Travel limit` — refuse to drive Z further than this (safety).
* `Backlash` — set to ~2× the measured backlash; the stage is then always
  approached from below, which matters for a 31 µm fine pass on a worn stage.
* `C/M/F/D` — coarse / medium / fine step and settle delay.
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
