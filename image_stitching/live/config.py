"""Tuning knobs for the live (on-the-fly) blade stitcher.

Every threshold here is either measured from the production scans in
``sample_scans/test_images`` or has a comment explaining what
it guards against.  Values are deliberately conservative: it is far cheaper to
reject one extra frame than to blend a bad transform into a defect mosaic.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LiveConfig:
    # ── Feature registration ────────────────────────────────────────────────
    work_width: int = 1000
    """Width (px) images are scaled to for feature extraction/registration.

    1920 px frames register ~4x slower with no accuracy benefit; the final
    blend still uses the full-resolution frame.
    """
    feature: str = "orb"
    """``orb`` (fast, ~40 ms/frame) or ``sift`` (slower, more robust to scale)."""
    n_features: int = 4000
    polish_width: int = 600
    """Resolution for the optional ECC sub-pixel polish (0 disables).  ECC is
    only ever accepted when it improves the measured NCC, because on these
    frames the illumination vignette pulls it off by a whole hole pitch."""
    ratio_test: float = 0.75
    ransac_px: float = 2.5
    min_inliers: int = 15
    """Minimum RANSAC inliers.  Good adjacent pairs give 55-1500 here, so 15 is
    permissive; the NCC gate below is what actually rejects wrong models."""
    prior_tol_px: float = 60.0
    """Base tolerance of the expected-step band.  Matches further than this
    from the predicted displacement are treated as outliers - this is what
    kills the false locks on the periodic cooling-hole pattern."""
    prior_tol_rel: float = 0.30
    """Extra tolerance proportional to the predicted step size.  When frames
    were rejected the predicted step is a *sum* of several moves and is much
    less certain, so the band has to widen with it."""
    loose_ratio_retry: float = 0.12
    """If the strict ratio test produces no usable model, retry with
    ``ratio_test + this`` (capped at 0.9) and half the inlier floor.

    Measured need: a pair spanning two motor steps (after one frame was
    rejected) yielded only 8 strict matches, but 31 with ratio 0.85 and the
    correct shift at NCC 0.95.  Without the retry the whole rest of the column
    came out as 'unregistered'."""
    retry_without_prior: bool = True
    """If the bounded attempt fails, retry unbounded and let the NCC gate
    decide.  Needed after a burst of rejected frames, where the predicted step
    can be off by more than the band."""
    polish_below_ncc: float = 0.92
    """Run the (slow) ECC sub-pixel polish only when the ORB result is already
    mediocre.  Measured: on good pairs the polish changes nothing."""

    # ── Acceptance / rejection ──────────────────────────────────────────────
    park_disconnected: bool = True
    """When a frame cannot be registered against anything already stitched,
    start a new band with it and keep stitching from there.

    The alternative (rejecting forever) throws away the rest of the column: on
    the recorded col02 a single 49-keypoint frame broke the chain and the
    following ten good frames were lost.  A parked frame never gets a made-up
    transform - it becomes the origin of its own band, 64 px to the right of the
    existing content - and the report lists it, so the gap is visible.  Bounded
    by ``max_disconnected_bands``."""
    max_disconnected_bands: int = 8
    """If parking is enabled, stop after this many bands."""

    dead_reckon: bool = True
    """A frame that cannot be registered is placed where the motion model says
    it is (last pose + predicted step) and blended as a flagged hole-filler,
    instead of being dropped or parked in a separate band.  Nothing is lost:
    the global stitch after the scan re-registers every frame anyway."""

    soft_unregisterable_ratio: float = 0.65
    """If a frame fails to register *and* is this much softer than the last
    accepted frame, it is a capture fault: drop it instead of starting a new
    mosaic band for it."""
    min_overlap: float = 0.18
    """Fraction of the frame that must overlap already-accepted content."""
    min_ncc: float = 0.55
    """Photometric correlation of the warped frame with its neighbour.

    Correct adjacent pairs measure 0.74-0.99 here (0.74 at ~13% overlap).
    Caveat measured later: on this surface a model that is *wrong by 77 px*
    still scores ~0.90, because vignetting and the slow brightness gradient
    dominate plain NCC.  ``min_ncc_hp`` is the gate that actually works."""
    min_ncc_hp: float = 0.15
    """Band-passed NCC (``Registration.ncc_hp``) a registration must reach.
    Wrong models score <= 0.07 (p90), right ones 0.36-0.85 (median)."""
    min_new_area: float = 0.10
    """A frame covering <10% uncovered area is a duplicate (stalled motor)."""
    max_step_frac: float = 0.85
    """Reject transforms that imply a jump larger than this fraction of the
    frame size - physically impossible for one motor step."""

    # ── Blur / exposure gate ────────────────────────────────────────────────
    blur_ratio: float = 0.72
    """Keep the frame only if its (exposure-normalised) Tenengrad sharpness is
    at least this fraction of the previously accepted frame, measured on the
    *overlap region only*.  Same-pixels comparison removes content bias."""
    min_sharpness: float = 1.5e-3
    """Absolute floor on normalised Tenengrad; below this nothing is in focus.

    Only catches gross cases (lens cap, total defocus): across one scan the
    absolute value legitimately varies 10x with illumination, which is why the
    relative gates below exist.
    """
    pre_blur_ratio: float = 0.30
    """Coarse pre-gate against the previously accepted frame's absolute
    sharpness.  Deliberately lax.

    Measured frame-to-frame ratios over the recorded scans: median 1.01,
    p5 = 0.58, p1 = 0.40 - the low tail is normal illumination drift, so a
    tighter threshold would throw away good frames.  0.30 only catches gross
    defocus, which matters because such a frame yields too few ORB matches to
    register at all and used to be misfiled as a disconnected band."""
    max_clipped_low: float = 0.55
    max_clipped_high: float = 0.30
    """Reject frames that are mostly black (shutter/lighting fault) or blown."""
    min_contrast: float = 4.0
    """Standard deviation of luma; flat frames carry no stitching information."""

    max_cached_anchors: int = 400
    """Anchors whose features stay in memory for the fallback search.

    Each cached anchor costs ~0.6 MB (1000 px grey + 4000 ORB descriptors), so
    400 anchors ~ 240 MB.  Keeping the whole scan searchable is what lets a new
    column be linked to the previous one instead of being dumped in a separate
    band.
    """

    # ── Rejected-frame policy ───────────────────────────────────────────────
    blur_policy: str = "fill"
    """What to do with a frame that registered correctly but is blurrier than
    its neighbour:

    ``reject`` - drop it.  Leaves a hole; only use if the capture loop
                 guarantees a retake.
    ``fill``   - blend it *only* into canvas pixels that are still empty, so
                 it never degrades good imagery, and record those pixels in a
                 low-quality mask for QC.  Default: an empty hole in a defect
                 mosaic is worse than a soft patch that is flagged.
    """
    fill_min_new_area: float = 0.02
    """A blurry frame that adds less than this much new coverage is dropped."""

    # ── Drift control ───────────────────────────────────────────────────────
    anchors: int = 2
    """Register each new frame against this many previously accepted frames
    (the most recent one plus spatially overlapping older ones) and average
    the implied global poses.  Bounds accumulated drift."""
    anchor_overlap: float = 0.30
    fallback_anchors: int = 8
    """When the temporal predecessor cannot be matched (column change, skipped
    frames, motor jump), retry against the nearest frames already placed,
    nearest first.  Without this a column change cascades into dozens of
    'disconnected' bands."""

    # ── Blending ────────────────────────────────────────────────────────────
    feather_px: int = 160
    """Blend ramp at frame edges.  64 px made every frame edge a visible step
    whenever its neighbour had a lower focus weight."""
    lens_radius_frac: float = 0.56
    """Image-circle radius as a fraction of the frame width (0 = no lens mask).
    The top-right and bottom-right corners of this camera are black in every
    frame; without the mask they were blended as dark wedges."""
    exposure_gain_clip: tuple[float, float] = (0.75, 1.35)
    output_scale: float = 1.0
    """Scale of the saved mosaic (0.5 halves memory 4x for live previews)."""
    max_canvas_px: float = 60e6
    """Safety valve: refuse to allocate a canvas bigger than this.

    Peak memory is roughly ``max_canvas_px * 16`` bytes (float32 RGB
    accumulator + float32 weights + the uint8 render), i.e. ~1 GB at 60 Mpx,
    plus the same amount again while the canvas grows.  Exceeding the limit is
    not recoverable by catching an exception on a small machine - the OOM killer
    takes the process - so the default is deliberately conservative.

    A full three-column scan of the recorded part needs ~95 Mpx at full
    resolution: run it with ``accum_scale=0.6`` (~34 Mpx, ~0.5 GB) or raise this
    deliberately on a workstation with the memory to match.
    """
    accum_scale: float = 1.0
    """Resolution the mosaic is accumulated at (1.0 = capture resolution).

    Registration always runs on full-resolution frames; only the blend is
    scaled.  0.5 cuts canvas memory 4x for very large parts."""

    # ── Runtime ─────────────────────────────────────────────────────────────
    flush_every: int = 10
    """Write a partial mosaic every N accepted frames so a crash/cancel never
    loses the whole scan."""
    queue_size: int = 8
    save_kept: bool = True
    """Copy accepted frames next to the mosaic (audit trail)."""
    save_rejected: bool = True
    """Write rejected frames to a folder named after the rejection reason, so a
    bad threshold is obvious from the filenames instead of being invisible."""
