# Live (on-the-fly) blade stitching

Streaming replacement for the capture-then-stitch batch flow. Frames are
assessed, registered and blended **as they are captured**, and frames that
cannot help a stitch (blurry, duplicate, unregisterable) are rejected on the
spot instead of being written to disk and cleaned up later.

```
capture ──► quality gate ──► bounded registration ──► accept? ──► feather blend
 (camera or   black/flat/        ORB + motion prior      overlap,      into growing
  replay)      gross defocus     + NCC verification      new area      canvas (+QC mask)
```

---

## 1. What the current flow actually costs (measured on this repo's data)

All numbers below come from `image_stitching/from_scratch/test_images`
(3 columns, 207 frames, 1920×1080) with the code in this repository.

| Finding | Measurement | Consequence |
|---|---|---|
| `blade_stitch.py` on the full folder **produces nothing** | 16 m 42 s, then `RuntimeError: Mosaic canvas exceeds safety limit` | Scan is lost; there is no partial output |
| Its registration is wrong but confident | phase correlation is defeated by the periodic cooling-hole rows; ECC polish then slid a *correct* ORB solution `(-4.7, 214.5)` → `(-90, 280)` while still reporting `cc > 0.5` | Silent misregistration → doubled holes in the mosaic |
| `cc` is not a geometry check | `cc` 0.88–0.99 on pairs whose true motion is 210–490 px | The diagnostics file cannot be trusted as a QC gate |
| Cost per pair | phase-correlate + full-res ECC: **28 s/pair** → 16 min for 207 frames | Impossible to run live |
| `cv2.Laplacian().var()` for blur | 13–14 for **both** a defocused and a sharp frame (dark vignette dominates) | The `MIN_FEATURES`/Laplacian style filter keeps blurry frames |
| Frames are far from uniformly spaced | one motor step = **210–490 px** (13 %–62 % overlap), occasionally 670 px (5 % overlap) | Neighbour-only chaining is fragile; overlap must be measured, not assumed |
| Real duplicates in the data | 4 frames are (near) pixel-identical to their `*_AF_*` neighbour, e.g. `col01_Ydown_9` == `col01_Ydown8_AF` (mean abs diff 2.8/255) | ~2–3 % of the archive is wasted bytes and wasted stitch time |
| Autofocus reference frames | 8–24 % of all saved frames; sharper-neighbour ratio 0.42–0.70 | They never help a stitch and cost disk + time |
| Chain drift | down-chain travel 10 391 px vs up-chain 10 515 px for the same physical distance → **1.2 % loop-closure error** | ~120 px of shear over one column if you only chain pairwise |

Good news that made a fast implementation possible: ORB + RANSAC +
photometric verification gives 55–1500 inliers and **40–130 ms per pair**, and
its answers are self-consistent (the 1.2 % loop-closure above).

---

## 2. Improvements, ordered by impact

### A. Fix registration before anything else (done, `live/registration.py`)
* ORB features at 1000 px work width (2–4× faster than full res, same accuracy).
* RANSAC affine, **bounded by the commanded motor step** (`prior_tol_px` +
  `prior_tol_rel`), which is what breaks the hole-period ambiguity.
* The prior only *disambiguates*: a full-match model is always computed too,
  and if the prior-constrained model is not corroborated by better NCC it is
  discarded. A blind prior would hide real motion (the actuator is open loop
  and does stall).
* Every model is **verified photometrically**: NCC over the actual overlap plus
  a physical-plausibility test on the step size. A wrong-but-confident model
  cannot pass.
* ECC is used only as an optional sub-pixel polish, only when it *improves*
  the measured NCC.

### B. Judge blur where it matters: on the overlap (done, `live/quality.py`)
Absolute sharpness is useless here — it varies **10× within one pass** (mean
luma drifts 145 → 30) and the objective is soft at the field edges in *every*
frame. What works:

* **Exposure-normalised Tenengrad** (`Σ|∇I|² / mean(I)²`): separated
  pre-autofocus frames (ratio < 1) from focused ones (230+) in the raw data.
* **Same-pixels comparison**: warp the new frame onto the previously accepted
  frame and compare sharpness *inside the overlap only*. The ratio then means
  "is this frame worse than what I already have?" — measured distribution:
  median 1.01, p5 0.58, p1 0.40, so the threshold is **0.72**.
* **A frame that will not register at all** is itself evidence of blur: on this
  data every "no registration" pair was a defocused stretch.

### C. Rejecting frames must not create holes (done, `live/pipeline.py`)
Simply deleting soft frames leaves gaps in the mosaic. Policy (`blur_policy`):

* a frame that is sharp **and** adds coverage → blended normally;
* a frame that is *softer* than its neighbour → blended **only into pixels that
  are still empty**, and every such pixel is written to
  `<prefix>_low_quality.png` for QC. It can never degrade good imagery;
* a frame that is soft **and** redundant → dropped;
* a frame that is soft **and** cannot register → dropped as a capture fault,
  not stitched into a separate band.

Duplicates are dropped but **kept as pose anchors** — otherwise the next frame
is matched across two motor steps, the overlap halves, matching collapses, and
the scan cascades into dozens of "disconnected" bands (observed: 104 bands
before this fix, 0 after).

### D. Never lose a scan (done, `live/mosaic.py`)
Incremental blending with a partial mosaic written every N accepted frames
(`--flush-every`). Kill the process mid-scan and you still have a mosaic.
Per-frame blending is limited to the frame's destination ROI (~40 ms) instead
of the whole canvas (200–440 ms), and the exposure gain is estimated on a
subsampled grid.

### E. Drift control (done, `live/pipeline.py`)
Each frame is registered against the previous frame **and** older spatially
overlapping anchors (`anchors=2`), and the implied global poses are averaged.
On a column change or a skipped frame, the pipeline retries against the nearest
frames already placed (`fallback_anchors`) — a column boundary no longer
becomes a separate mosaic.

### F. Capture-side changes (recommended, not yet in `main3.py`)
1. **Capture only when still.** The GUI already waits for the motion detector,
   but `_get_frame()` returns whatever the feed thread last decoded — up to
   ~80 ms stale, i.e. possibly a frame from *during* the move. Take a fresh
   grab for the saved frame, or use the settle gate in `run_live.py --camera`.
2. **Stop saving `*_AF_*` frames.** They are autofocus references taken while Z
   is still settling; measured sharper-neighbour ratio 0.42–0.70 and they are
   exact duplicates of the next frame in several cases. Keep them only if you
   need an audit trail, in a separate folder.
3. **Make the step overlap-aware.** The sweep moves a fixed actuator duration,
   which produces 5 %–62 % overlap depending on speed and load. Now that a
   shift can be measured in ~40 ms, move → measure → capture when
   `1 - shift/frame_height ≥ 0.35`. Two lines in `_sweep_worker` and a much
   more uniform mosaic; it also removes the 5 %-overlap frames that currently
   have to be dropped.
4. **Overlap the columns.** The X step (0.4 mm) sometimes produced column pairs
   with almost no shared content — no overlap, no link. Use ~⅓ frame width.
5. **Fix the exposure per frame** (LED/lens) instead of letting mean luma drift
   145 → 30 across a column; the blender compensates, but matching and defect
   visibility both improve with a stable exposure.
6. **Store PNG or JPEG q95 for the kept frames only** — the current sweep writes
   every frame at default quality, including the ones that end up rejected.

---

## 3. What to delete (and how)

Two tools, same criteria:

```bash
# online: run_live.py copies accepted frames to <out>_kept/ and rejected ones
# to <out>_rejected/<reason>__<name>.jpg for audit
python live/run_live.py --replay ../from_scratch/test_images --out runs/col01

# offline audit of an existing folder (dry run by default - deletes nothing)
python live/purge.py ../from_scratch/test_images
python live/purge.py ../from_scratch/test_images --move-to ../_trash   # or --delete
python live/purge.py ../from_scratch/test_images --drop-af            # + AF frames
```

Verdicts: `unreadable`, `underexposed`, `overexposed`, `no_contrast`,
`soft(ratio=…)`, `unmatchable`, `duplicate_of:<frame>`,
`autofocus_reference` (with `--drop-af`).

On the sample data the dry run flags **19 of 207** frames (7 unmatchable
defocused frames in `col02`, 8 soft frames, 4 duplicates) and touches nothing.
Every verdict, with its metrics, lands in `purge_report.csv`.

> Blur thresholds are per-pair, not absolute. Do not lower `--blur-ratio` below
> ~0.6: the p5 of frame-to-frame sharpness in this data is 0.58 from
> illumination drift alone, and you would start deleting good frames.

---

## 4. Using it

### Standalone (replay or camera)

```bash
cd image_stitching

# replay a finished scan, keeping only usable frames
python live/run_live.py --replay from_scratch/test_images --pattern 'col01_*' \
       --out runs/col01 --flush-every 0

# live capture card; submit only when the scene has been still for 0.8 s
python live/run_live.py --camera 1 --out runs/live --settle 0.8 --preview
```

Outputs, all prefixed with `--out`:

| file | contents |
|---|---|
| `_mosaic.png` / `.jpg` | stitched result |
| `_coverage.png` | which pixels are backed by real imagery |
| `_low_quality.png` | pixels that came only from soft fill-in frames |
| `_frames.csv` | one row per frame: keep/drop, reason, dx/dy, NCC, overlap, new area, sharpness, gain, pose_x/pose_y, ms |
| `_report.txt` | summary, per-column verdict (`ok` / `CHECK`), plus separate lists of frames that are redundant (safe to delete) and frames that must be re-captured |
| `_partial_*` | last flushed partial mosaic |

### Embedded in the sweep GUI
`main_code/main3.py` now submits each saved sweep frame to the stitcher
(`LIVE_STITCH = True`, config in `LIVE_STITCH_CONFIG`) and finalises the mosaic
at the end of the sweep into `main_code/blade_mosaic/`. It is non-blocking and
self-disabling: if the stitching module is missing or raises, the scan
continues unchanged. The growing mosaic is shown **inside the main window**
(next to the live feed) with its keep/drop counters; closing the window
mid-sweep still writes the mosaic. Expected log:

```
[LIVE] KEEP col01_Ydown_1_154429.jpg  ok              ncc=0.00 ov=0.00 sharp=1.00  284ms
[LIVE] KEEP col01_Ydown_2_154431.jpg  ok              ncc=0.98 ov=0.61 sharp=0.86  691ms
[LIVE] DROP col01_Ydown_9_154458.jpg  duplicate(...)  ncc=1.00 ov=1.00 sharp=1.02  174ms
```

### Focus weighting (why a blurred band no longer ends up in the mosaic)
`submit(..., weight=w)` takes an optional per-pixel confidence map in frame
coordinates. It multiplies the feathered edge weight, so in an overlap the sharp
frame wins the pixel instead of the first frame to arrive keeping it. The maps
come from `main_code/focus_engine.py` (`tile_weight_map` for one frame,
`compose_focus_stack` for a fused Z-bracket); see `main_code/AUTOFOCUS.md` for
why that is necessary on a curved blade and what it changed (blurred regions of
the finished mosaic: 42 % → 0 % on a synthetic 2.4 mm curved surface).

### Live preview (what the GUI shows)
```python
rgb, summary = st.preview(max_width=520)   # safe to call from any thread
```
`preview()` takes the same lock the blend worker uses, so it can never observe a
half-blended canvas, and it renders the current mosaic (cropped to content,
downscaled for display) as RGB for a Tk/`PhotoImage` widget. `summary()` carries
`accepted / filled / rejected / disconnected / mean_ncc / canvas /
low_quality_pct`, which is exactly what the GUI's mosaic panel prints.
`st.flush()` writes a snapshot mid-scan without stopping anything, so "save now"
during a sweep is just a button.

The GUI polls `preview()` about once a second **from a worker thread** and hands
the result back to Tk with `after(0, ...)`: rendering the canvas is ~180 ms, far
too long to run inside the event loop, and the sweep must never wait on it.

### Library
```python
from live import LiveConfig, LiveStitcher

st = LiveStitcher(LiveConfig(), out_prefix="runs/blade")
st.submit(frame_bgr, tag="col01_Ydown_1")   # non-blocking
...
paths = st.finalize()
```

---

## 5. Measured result

`col01` (85 frames, `*_AF_*` included, nothing pre-filtered):

| | old `blade_stitch.py` | `live/` |
|---|---|---|
| time | 16 m 42 s over the whole folder, then crashed | 30 s for the column (296 ms/frame) |
| output | none | 12 389 × 2 749 mosaic + coverage + low-quality masks |
| frames stitched | – | 43 (+8 soft fill-ins) |
| duplicates rejected | – | 31 |
| disconnected bands | – | 0 |
| mean NCC of kept frames | – | 0.959 |
| partial output on crash | no | yes (`--flush-every`) |

Trajectory check from the per-frame log (`pose_y` column): the down pass walks
0 → 10 314 px, the up pass returns to ≈ −600 px, then the column steps in x
(341 → 750). That is the serpentine the sweep is supposed to make, which is a
much stronger correctness signal than any pairwise score.

`col02` is the interesting counter-example: an out-of-focus stretch
(sharpness 0.04–0.06, 49–220 ORB keypoints, versus 0.1–1.4 and 250–4000 for
good frames) breaks the chain in the middle of the column. The pipeline:

* drops the worst of those frames as `out_of_focus` / `unregisterable_soft`,
* stitches the rest, starting a clearly labelled new band where no overlap
  exists,
* and prints `col02: stitched in 2 disconnected band(s)` plus
  `CHECK` in the per-column table, which is the instruction to re-scan it.

That is the honest outcome - the data has no geometric link across that
stretch, so any tool that produces one continuous mosaic there is guessing.

Operating notes for whole-part scans: three columns at full resolution need
~95 Mpx of canvas, above the 60 Mpx default because the OOM killer, not a
catchable exception, is what fails first on a 4–8 GB machine. Run
`--accum-scale 0.6` (~34 Mpx) for the whole part, or stitch one column per run
(what a sweep naturally produces).

## 6. Known limitations / next steps

* **Pose fusion is translation-only** (rotation/scale are taken from the best
  anchor). Fine here — measured rotation is < 0.15° — but revisit for a camera
  that is re-fixtured mid-scan. A global bundle adjustment over the pose graph
  is the natural next step if drift ever shows up in the QC mask.
* **Column linking assumes overlap.** With the current 0.4 mm step that is not
  guaranteed; the pipeline then honestly reports separate bands instead of
  inventing a link.
* **No radiometric model.** Per-frame gain is a single clipped scalar; for
  quantitative defect measurement, flat-field correction would be better than
  seam blending.
* **`--settle` gating is time-based.** A proper trigger would evaluate
  `sharpness` and frame-diff inside a short window; `MotionDetector` in
  `main3.py` already has the frame-diff half of that.
* **Tests** (`live/tests/test_live.py`, 11 cases) cover shift recovery, quality
  gates, duplicate rejection, hole-filling, resilience to a bad frame, async
  finalisation and one real-data registration test. Run:
  `python live/tests/test_live.py`.
